package com.hypenex.vcddetector

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Debug
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * On-device detection, model-agnostic: decode -> resample to adapter.spec.sampleRate -> adapter
 * validation/windows -> one ONNX session (sequential windows, batch 1) -> adapter scoring.
 * All model work runs on one background thread; installation runs on its own thread.
 * No network use except installModel().
 */
class DetectorEngine(private val context: Context, val adapter: DetectorAdapter = Adapters.default) {
  private val spec get() = adapter.spec
  val store = ModelStore(context.filesDir, adapter.spec.model, adapter.spec.id)
  private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "vcd-inference") }
  private val installer = Executors.newSingleThreadExecutor { r -> Thread(r, "vcd-install") }
  @Volatile private var session: OnnxWindowSession? = null
  private var lastLoad: Map<String, Any?>? = null

  /** External app dir: readable/writable by adb without run-as, private to the app otherwise. */
  private val extDir: File get() = context.getExternalFilesDir(null) ?: context.filesDir
  val referenceDir: File get() = File(extDir, "reference")
  private val dumpRoot: File get() = File(extDir, "dumps")

  fun submit(task: () -> Any?, onOk: (Any?) -> Unit, onErr: (Throwable) -> Unit) {
    worker.execute {
      try { onOk(task()) } catch (t: Throwable) { onErr(t) }
    }
  }

  @Volatile private var installQueued = false

  fun startInstall(progress: (Map<String, Any?>) -> Unit) {
    if (store.installing || installQueued) return
    installQueued = true // visible to getStatus() immediately, before the install thread starts
    installer.execute {
      try {
        store.install { phase, done, total ->
          progress(mapOf("phase" to phase, "doneBytes" to done, "totalBytes" to total))
        }
      } catch (t: Throwable) {
        progress(mapOf("phase" to "error", "error" to (t.message ?: t.javaClass.simpleName)))
      } finally {
        installQueued = false
      }
    }
  }

  // ---- model ---------------------------------------------------------------------
  fun status(): Map<String, Any?> = store.status() + mapOf(
    "installing" to (store.installing || installQueued),
    "loaded" to (session != null),
    "lastLoad" to lastLoad,
    "referenceAvailable" to File(referenceDir, "manifest.json").exists(),
    "attribution" to spec.attribution,
    "adapter" to spec.toMap(),
  )

  fun threads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

  fun load(): Map<String, Any?> {
    session?.let { return mapOf("alreadyLoaded" to true) + (lastLoad ?: emptyMap()) }
    if (!store.isVerified()) throw IllegalStateException("model not installed (or not verified)")
    val before = memory()
    val s = OnnxWindowSession.open(store.modelFile, threads())
    session = s
    val after = memory()
    lastLoad = mapOf("loadMs" to s.loadMs, "settings" to s.settings, "memoryBefore" to before, "memoryAfter" to after)
    return lastLoad!!
  }

  fun unload() {
    session?.close(); session = null
  }

  private fun requireSession(): Pair<OnnxWindowSession, Double?> {
    val existing = session
    if (existing != null) return Pair(existing, null)
    val info = load()
    return Pair(session!!, info["loadMs"] as Double)
  }

  // ---- analysis ------------------------------------------------------------------
  fun analyseUri(uri: Uri, mode: String, compare: Boolean, dump: Boolean): Map<String, Any?> {
    val t0 = System.nanoTime()
    val sha = sha256Uri(uri)
    val decoded = try {
      AudioDecode.decode(context, uri)
    } catch (e: AudioDecode.Unreadable) {
      return rejected(mode, e.message ?: "unreadable audio", t0, mapOf("sha256" to sha))
    }
    val tDecode = ms(t0)
    return analyseDecoded(decoded, mode, t0, tDecode, mapOf("sha256" to sha), compare, dump)
  }

  /** In-memory float32 PCM (interleaved): the entry point for remote audio windows from in-app calls later. */
  fun analysePcm(pcmLe: ByteArray, sampleRate: Int, channels: Int, mode: String): Map<String, Any?> {
    val t0 = System.nanoTime()
    val fb = ByteBuffer.wrap(pcmLe).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val inter = FloatArray(fb.remaining()).also { fb.get(it) }
    if (channels < 1 || sampleRate <= 0) throw IllegalArgumentException("bad sampleRate/channels")
    if (inter.isEmpty()) return rejected(mode, "empty audio (0 samples)", t0, emptyMap())
    if (inter.any { !it.isFinite() }) return rejected(mode, "audio contains NaN/Inf samples", t0, emptyMap())
    val frames = inter.size / channels
    val mono = if (channels == 1) inter else FloatArray(frames) { f ->
      var s = 0.0; for (c in 0 until channels) s += inter[f * channels + c]; (s / channels).toFloat()
    }
    val d = Decoded(sampleRate, channels, frames, mono, "pcm")
    return analyseDecoded(d, mode, t0, ms(t0), mapOf("in_memory" to true), compare = false, dump = false)
  }

  private fun analyseDecoded(d: Decoded, mode: String, t0: Long, tDecode: Double, source: Map<String, Any?>,
                             compare: Boolean, dump: Boolean): Map<String, Any?> {
    require(mode in spec.preprocessingModes) { "unknown preprocessing mode '$mode' for adapter ${spec.id}" }
    val tr = System.nanoTime()
    val x16 = Resampler.resample(d.mono, d.sampleRate, spec.sampleRate)
    val tResample = ms(tr)
    val audio = linkedMapOf<String, Any?>("native_sample_rate" to d.sampleRate, "channels" to d.channels,
      "native_frames" to d.frames, "decoder" to d.decoder, "decoder_notes" to d.notes)
    val tp = System.nanoTime()
    val v = try {
      adapter.validate(x16)
    } catch (e: AudioRejected) {
      return rejected(mode, e.message ?: "rejected", t0, source, audio)
    }
    val (wins, starts) = adapter.windows(v.x, mode)
    val tPre = ms(tp)

    val (s, loadMs) = requireSession()
    val windowOut = ArrayList<Map<String, Any?>>()
    val scores = DoubleArray(wins.size)
    val logitsAll = ArrayList<FloatArray>()
    val tm = System.nanoTime()
    for ((i, w) in wins.withIndex()) {
      val tw = System.nanoTime()
      val logits = s.run(w)
      val p = adapter.windowScore(logits)
      scores[i] = p
      logitsAll.add(logits)
      windowOut.add(mapOf("start_s" to starts[i].toDouble() / spec.sampleRate, "logits" to logits.map { it.toDouble() },
        "synthetic_score" to p, "ms" to ms(tw)))
    }
    val tModel = ms(tm)
    val mean = adapter.aggregate(scores)
    val result = linkedMapOf<String, Any?>(
      "status" to "ok",
      "prediction" to adapter.label(mean),
      "synthetic_score" to mean,
      "synthetic_score_max" to scores.max(),
      "score_is_calibrated" to spec.scoreCalibrated,
      "decision_threshold" to spec.decisionThreshold,
      "threshold_is_tuned" to spec.thresholdTuned,
      "adapter" to mapOf("id" to spec.id, "version" to spec.version, "model_revision" to spec.model.revision),
      "reject_reason" to null,
      "duration_s" to v.info["duration_s"],
      "analysed_duration_s" to v.info["analysed_duration_s"],
      "truncated" to v.info["truncated"],
      "audio" to (audio + v.info.filterKeys { it in setOf("peak_dbfs", "rms_dbfs", "active_s", "clipping_frac") }),
      "windows" to windowOut,
      "preprocessing" to mode,
      "timing_ms" to mapOf("decode" to tDecode, "resample" to tResample, "preprocess" to tPre,
        "model_load" to loadMs, "model_total" to tModel, "per_window_mean" to tModel / wins.size, "total" to ms(t0)),
      "memory" to memory(),
      "session" to s.settings,
      "source" to source,
      "on_device" to true,
    )
    val sha = source["sha256"] as? String
    if (dump && sha != null) result["dump_dir"] = dump(sha, d.mono, x16, mode, wins, result)
    if (compare && sha != null) result["comparison"] = compareWithReference(sha, d, x16, mode, wins, logitsAll, mean)
    return result
  }

  private fun rejected(mode: String, reason: String, t0: Long, source: Map<String, Any?>,
                       audio: Map<String, Any?>? = null) = mapOf(
    "status" to "rejected", "prediction" to null, "synthetic_score" to null, "synthetic_score_max" to null,
    "score_is_calibrated" to spec.scoreCalibrated, "decision_threshold" to spec.decisionThreshold,
    "adapter" to mapOf("id" to spec.id, "version" to spec.version, "model_revision" to spec.model.revision),
    "reject_reason" to reason, "preprocessing" to mode, "windows" to emptyList<Any>(), "audio" to audio,
    "timing_ms" to mapOf("total" to ms(t0)), "source" to source, "on_device" to true,
  )

  // ---- milestone 1: exported Colab tensors through the on-device runtime ----------------
  fun runReferenceTensors(maxWindows: Int): Map<String, Any?> {
    val manifest = readManifest() ?: throw IllegalStateException("no reference export at ${referenceDir.absolutePath}")
    val (s, loadMs) = requireSession()
    val clips = manifest.getJSONObject("clips")
    val rows = ArrayList<Map<String, Any?>>()
    var worst = 0.0
    var run = 0
    for (cid in clips.keys()) {
      val modes = clips.getJSONObject(cid).getJSONObject("modes")
      for (mode in modes.keys()) {
        val m = modes.getJSONObject(mode)
        val wins = readF32(m.getJSONObject("windows"))
        val refLogits = m.getJSONArray("logits")
        val n = refLogits.length()
        val per = wins.size / n
        for (i in 0 until n) {
          if (run >= maxWindows) break
          val w = wins.copyOfRange(i * per, (i + 1) * per)
          val t = System.nanoTime()
          val got = s.run(w)
          val dt = ms(t)
          val ref = refLogits.getJSONArray(i)
          val diff = maxOf(abs(got[0] - ref.getDouble(0)), abs(got[1] - ref.getDouble(1)))
          worst = maxOf(worst, diff)
          rows.add(mapOf("clip" to cid, "mode" to mode, "window" to i, "device_logits" to got.map { it.toDouble() },
            "colab_logits" to listOf(ref.getDouble(0), ref.getDouble(1)), "max_abs_logit_diff" to diff,
            "device_score" to adapter.windowScore(got), "colab_score" to m.getJSONArray("scores").getDouble(i), "ms" to dt))
          run++
        }
      }
    }
    return mapOf("windows_run" to run, "max_abs_logit_diff" to worst, "model_load_ms" to loadMs, "rows" to rows,
      "memory" to memory(), "session" to s.settings)
  }

  // ---- reference comparison (milestone 2) -------------------------------------------
  private fun readManifest(): JSONObject? {
    val f = File(referenceDir, "manifest.json")
    return if (f.exists()) JSONObject(f.readText()) else null
  }

  private fun readF32(spec: JSONObject): FloatArray {
    val f = File(referenceDir, spec.getString("file"))
    val bytes = f.readBytes()
    val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return FloatArray(fb.remaining()).also { fb.get(it) }
  }

  private fun diff(a: FloatArray, b: FloatArray, lag: Int = 0): Map<String, Any?> {
    var maxAbs = 0.0; var ss = 0.0; var refSs = 0.0; var n = 0
    for (i in a.indices) {
      val j = i + lag
      if (j < 0 || j >= b.size) continue
      val d = (a[i] - b[j]).toDouble()
      maxAbs = maxOf(maxAbs, abs(d)); ss += d * d; refSs += b[j].toDouble() * b[j]; n++
    }
    return mapOf("device_len" to a.size, "colab_len" to b.size, "overlap" to n, "lag" to lag, "max_abs_diff" to maxAbs,
      "rms_diff" to (if (n > 0) sqrt(ss / n) else null), "snr_db" to (if (ss > 0) 10 * kotlin.math.log10(refSs / ss) else null))
  }

  /** Best integer lag (device[i] ~ colab[i+lag]) within +-maxLag using the first 2 s. */
  private fun bestLag(a: FloatArray, b: FloatArray, maxLag: Int, span: Int): Int {
    var best = 0; var bestScore = Double.NEGATIVE_INFINITY
    val n = minOf(span, a.size)
    for (lag in -maxLag..maxLag) {
      var acc = 0.0
      for (i in 0 until n) { val j = i + lag; if (j in b.indices) acc += a[i].toDouble() * b[j] }
      if (acc > bestScore) { bestScore = acc; best = lag }
    }
    return best
  }

  private fun compareWithReference(sha: String, d: Decoded, x16: FloatArray, mode: String, wins: List<FloatArray>,
                                   logits: List<FloatArray>, mean: Double): Map<String, Any?> {
    val manifest = readManifest() ?: return mapOf("available" to false, "reason" to "no reference export on device")
    val clips = manifest.getJSONObject("clips")
    val cid = clips.keys().asSequence().firstOrNull { clips.getJSONObject(it).getString("source_sha256") == sha }
      ?: return mapOf("available" to false, "reason" to "picked file is not one of the reference clips (sha256 $sha)")
    val c = clips.getJSONObject(cid)
    val refNative = readF32(c.getJSONObject("native_mono"))
    val refX16 = readF32(c.getJSONObject("x16"))
    val lagNative = bestLag(d.mono, refNative, 4096, minOf(2 * d.sampleRate, d.mono.size))
    val m = c.getJSONObject("modes").getJSONObject(mode)
    val refWins = readF32(m.getJSONObject("windows"))
    val refLogits = m.getJSONArray("logits")
    val perWin = ArrayList<Map<String, Any?>>()
    for (i in wins.indices) {
      if (i >= refLogits.length()) break
      val rw = refWins.copyOfRange(i * spec.windowSamples, (i + 1) * spec.windowSamples)
      val ref = refLogits.getJSONArray(i)
      perWin.add(diff(wins[i], rw) + mapOf("device_logits" to logits[i].map { it.toDouble() },
        "colab_logits" to listOf(ref.getDouble(0), ref.getDouble(1))))
    }
    return mapOf(
      "available" to true, "clip" to cid, "mode" to mode,
      "native_rate" to mapOf("device" to d.sampleRate, "colab" to c.getInt("native_sr")),
      "native_mono" to diff(d.mono, refNative), "native_mono_best_lag" to diff(d.mono, refNative, lagNative),
      "x16" to diff(x16, refX16),
      "window_count" to mapOf("device" to wins.size, "colab" to refLogits.length()),
      "windows" to perWin,
      "clip_score" to mapOf("device" to mean, "colab" to m.getDouble("clip_score_mean"), "abs_diff" to abs(mean - m.getDouble("clip_score_mean"))),
    )
  }

  private fun dump(sha: String, native: FloatArray, x16: FloatArray, mode: String, wins: List<FloatArray>,
                   result: Map<String, Any?>): String {
    val dir = File(dumpRoot, sha.take(12)).apply { mkdirs() }
    fun write(name: String, arrays: List<FloatArray>) {
      FileOutputStream(File(dir, name)).channel.use { ch ->
        for (a in arrays) {
          val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
          bb.asFloatBuffer().put(a); ch.write(bb)
        }
      }
    }
    write("native_mono.f32", listOf(native)); write("x16.f32", listOf(x16)); write("windows_$mode.f32", wins)
    File(dir, "result_$mode.json").writeText(JSONObject(result.filterKeys { it != "memory" }).toString(1))
    return dir.absolutePath
  }

  // ---- helpers -------------------------------------------------------------------------
  private fun sha256Uri(uri: Uri): String {
    val md = MessageDigest.getInstance("SHA-256")
    context.contentResolver.openInputStream(uri)?.use { input ->
      val buf = ByteArray(1 shl 16)
      while (true) { val n = input.read(buf); if (n < 0) break; md.update(buf, 0, n) }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
  }

  fun memory(): Map<String, Any?> {
    val status = try { File("/proc/self/status").readLines() } catch (_: Exception) { emptyList() }
    fun kb(key: String) = status.firstOrNull { it.startsWith("$key:") }?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    // getProcessMemoryInfo() is rate-limited for apps and returned stale values (58 MB after a 1.4 GB load),
    // so PSS is measured externally with `adb shell dumpsys meminfo`; RSS/HWM from /proc are reliable.
    val pss: Long? = null
    return mapOf("rss_mb" to kb("VmRSS")?.div(1024.0), "peak_rss_mb" to kb("VmHWM")?.div(1024.0),
      "pss_mb" to pss, "native_heap_mb" to Debug.getNativeHeapAllocatedSize() / 1048576.0,
      "device_total_mb" to mi.totalMem / 1048576.0, "device_avail_mb" to mi.availMem / 1048576.0,
      "low_memory" to mi.lowMemory)
  }

  private fun ms(t: Long) = (System.nanoTime() - t) / 1e6

  @Suppress("unused")
  private fun JSONArray.toDoubles() = DoubleArray(length()) { getDouble(it) }
}
