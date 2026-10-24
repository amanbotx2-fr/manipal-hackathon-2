package com.hypenex.vcddetector

import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Downloads an adapter's pinned model once into app-private storage (filesDir), resumable,
 * sha256-verified. Directory: models/<adapterId>-<rev7>/ (Dhwani: models/dhwani-ee01871/).
 */
class ModelStore(filesDir: File, private val artifact: ModelArtifact, adapterId: String) {
  private val dir = File(filesDir, "models/$adapterId-${artifact.revision.take(7)}")
  val modelFile = File(dir, artifact.file)
  private val partFile = File(dir, artifact.file + ".part")
  private val marker = File(dir, artifact.file + ".sha256_ok")
  private val cancelled = AtomicBoolean(false)
  @Volatile var installing = false
    private set

  fun status(): Map<String, Any?> = mapOf(
    "installed" to isVerified(),
    "installing" to installing,
    "modelPath" to modelFile.absolutePath,
    "modelBytes" to (if (modelFile.exists()) modelFile.length() else 0L),
    "partialBytes" to (if (partFile.exists()) partFile.length() else 0L),
    "expectedBytes" to artifact.bytes,
    "sha256" to artifact.sha256,
    "revision" to artifact.revision,
    "freeBytes" to dir.parentFile?.let { it.mkdirs(); it.usableSpace },
  )

  fun isVerified(): Boolean =
    modelFile.exists() && modelFile.length() == artifact.bytes && marker.exists() &&
      marker.readText() == stamp(modelFile)

  private fun stamp(f: File) = "${f.length()}:${f.lastModified()}:${artifact.sha256}"

  fun cancel() = cancelled.set(true)

  fun delete() {
    marker.delete(); modelFile.delete(); partFile.delete()
  }

  /**
   * Blocking; call off the UI thread. progress(phase, done, total).
   * Phases: "downloading", "verifying", "done".
   */
  fun install(progress: (String, Long, Long) -> Unit) {
    if (isVerified()) { progress("done", artifact.bytes, artifact.bytes); return }
    installing = true
    cancelled.set(false)
    try {
      dir.mkdirs()
      if (modelFile.exists() && !marker.exists()) {
        // Complete file without a marker (e.g. interrupted verification): verify it again.
        if (modelFile.length() == artifact.bytes) modelFile.renameTo(partFile) else modelFile.delete()
      }
      val need = artifact.bytes - (if (partFile.exists()) partFile.length() else 0L)
      if (need > 0 && dir.usableSpace < need + 64L * 1024 * 1024) {
        throw IOException("not enough device storage: need ${need / 1048576} MB + 64 MB headroom, have ${dir.usableSpace / 1048576} MB")
      }
      Log.i(TAG, "install: starting, have ${if (partFile.exists()) partFile.length() else 0} bytes")
      download(progress)
      Log.i(TAG, "install: download complete, verifying sha256")
      progress("verifying", 0, artifact.bytes)
      val sha = sha256(partFile) { done -> progress("verifying", done, artifact.bytes) }
      if (sha != artifact.sha256) {
        partFile.delete()
        throw IOException("sha256 mismatch ($sha); partial file deleted, retry the download")
      }
      if (!partFile.renameTo(modelFile)) throw IOException("could not finalise model file")
      marker.writeText(stamp(modelFile))
      Log.i(TAG, "install: verified and finalised ${modelFile.absolutePath}")
      progress("done", artifact.bytes, artifact.bytes)
    } finally {
      installing = false
    }
  }

  private fun download(progress: (String, Long, Long) -> Unit) {
    var attempts = 0
    while (true) {
      val have = if (partFile.exists()) partFile.length() else 0L
      if (have == artifact.bytes) return
      if (have > artifact.bytes) { partFile.delete(); continue }
      if (cancelled.get()) throw IOException("cancelled")
      try {
        fetchFrom(have, progress)
        attempts = 0
      } catch (e: IOException) {
        Log.w(TAG, "download: ${e.message}; attempt $attempts")
        if (cancelled.get() || ++attempts >= 5) throw e
        Thread.sleep(2000L * attempts) // transient network error: resume from the .part file
      }
    }
  }

  private fun fetchFrom(offset: Long, progress: (String, Long, Long) -> Unit) {
    val conn = (URL(artifact.url).openConnection() as HttpURLConnection).apply {
      instanceFollowRedirects = true
      connectTimeout = 30_000
      readTimeout = 60_000
      if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
    }
    try {
      val code = conn.responseCode
      Log.i(TAG, "download: offset=$offset HTTP $code host=${conn.url.host}")
      val append = when {
        offset > 0 && code == 206 -> true
        code == 200 -> false // server ignored Range: restart from zero
        else -> throw IOException("HTTP $code from model host")
      }
      var done = if (append) offset else 0L
      conn.inputStream.use { input ->
        FileOutputStream(partFile, append).use { out ->
          val buf = ByteArray(1 shl 20)
          var lastReport = 0L
          while (true) {
            if (cancelled.get()) throw IOException("cancelled")
            val n = input.read(buf)
            if (n < 0) break
            out.write(buf, 0, n)
            done += n
            val now = System.currentTimeMillis()
            if (now - lastReport > 300) { progress("downloading", done, artifact.bytes); lastReport = now }
          }
          out.fd.sync()
        }
      }
      progress("downloading", done, artifact.bytes)
      if (done != artifact.bytes) throw IOException("connection ended at $done of ${artifact.bytes} bytes")
    } finally {
      conn.disconnect()
    }
  }

  companion object {
    private const val TAG = "VcdModelStore"

    fun sha256(f: File, onProgress: (Long) -> Unit = {}): String {
      val md = MessageDigest.getInstance("SHA-256")
      FileInputStream(f).use { input ->
        val buf = ByteArray(1 shl 20)
        var done = 0L
        var last = 0L
        while (true) {
          val n = input.read(buf)
          if (n < 0) break
          md.update(buf, 0, n)
          done += n
          val now = System.currentTimeMillis()
          if (now - last > 300) { onProgress(done); last = now }
        }
      }
      return md.digest().joinToString("") { "%02x".format(it) }
    }
  }
}
