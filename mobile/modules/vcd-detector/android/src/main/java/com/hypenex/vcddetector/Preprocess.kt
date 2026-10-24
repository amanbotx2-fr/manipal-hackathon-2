package com.hypenex.vcddetector

import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToLong
import kotlin.math.sqrt

/**
 * Dhwani-specific preprocessing, used by DhwaniAdapter. Port of src/vcd/dhwani_infer.py
 * (validation, windowing, preprocessing modes, scoring).
 * Keep in sync with that file; constants and rounding are copied deliberately.
 */
object Preprocess {
  const val SR = 16000
  const val WIN = 64000
  const val HOP = 32000
  const val MIN_DUR_S = 2.0
  const val MAX_DUR_S = 10.0
  const val PEAK_TARGET_DBFS = -3.0
  const val SILENT_PEAK_DBFS = -50.0
  const val ACTIVE_FRAME_S = 0.025
  const val ACTIVE_REL_DB = 35.0
  const val ACTIVE_ABS_DBFS = -55.0
  const val MIN_ACTIVE_S = 1.0
  const val SYNTHETIC_INDEX = 1
  const val DECISION_THRESHOLD = 0.5
  val MODES = listOf("train_peak", "train_zscore", "api_replica")

  private fun dbfs(v: Double) = 20 * log10(max(v, 1e-12))
  private fun round(v: Double, dp: Int): Double {
    val f = 10.0.pow(dp); return (v * f).roundToLong() / f
  }

  fun stats(x: FloatArray): Map<String, Double> {
    var peak = 0.0; var sumSq = 0.0; var clip = 0
    for (v in x) {
      val a = kotlin.math.abs(v.toDouble())
      if (a > peak) peak = a
      sumSq += v.toDouble() * v
      if (a >= 0.999) clip++
    }
    val rms = if (x.isNotEmpty()) sqrt(sumSq / x.size) else 0.0
    val flen = (ACTIVE_FRAME_S * SR).toInt()
    val nfr = x.size / flen
    var activeS = 0.0
    if (nfr > 0) {
      val fdb = DoubleArray(nfr)
      for (f in 0 until nfr) {
        var s = 0.0
        for (i in f * flen until (f + 1) * flen) s += x[i].toDouble() * x[i]
        fdb[f] = 10 * log10(s / flen + 1e-20)
      }
      val top = fdb.max()
      activeS = fdb.count { it > top - ACTIVE_REL_DB && it > ACTIVE_ABS_DBFS } * ACTIVE_FRAME_S
    }
    return mapOf(
      "peak_dbfs" to round(dbfs(peak), 2),
      "rms_dbfs" to round(dbfs(rms), 2),
      "active_s" to round(activeS, 3),
      "clipping_frac" to round(if (x.isNotEmpty()) clip.toDouble() / x.size else 0.0, 5),
    )
  }

  /** 2-10 s policy and silence gates; returns the analysed audio (first 10 s if longer). */
  fun validate(x16: FloatArray): ValidatedAudio {
    val dur = x16.size.toDouble() / SR
    val info = linkedMapOf<String, Any>("duration_s" to round(dur, 3), "truncated" to false)
    if (dur < MIN_DUR_S) throw AudioRejected("too short: ${"%.2f".format(dur)} s < $MIN_DUR_S s minimum")
    var x = x16
    if (dur > MAX_DUR_S) { x = x16.copyOf((MAX_DUR_S * SR).toInt()); info["truncated"] = true }
    info["analysed_duration_s"] = round(x.size.toDouble() / SR, 3)
    val st = stats(x)
    info.putAll(st)
    if (st["peak_dbfs"]!! < SILENT_PEAK_DBFS) throw AudioRejected("silent: peak ${st["peak_dbfs"]} dBFS < $SILENT_PEAK_DBFS dBFS")
    if (st["active_s"]!! < MIN_ACTIVE_S) {
      throw AudioRejected("mostly silent: ${"%.2f".format(st["active_s"])} s active signal < $MIN_ACTIVE_S s")
    }
    return ValidatedAudio(x, info)
  }

  fun windowStarts(n: Int): List<Int> {
    if (n <= WIN) return listOf(0)
    val starts = (0..(n - WIN) step HOP).toMutableList()
    if (starts.last() + WIN < n) starts.add(n - WIN)
    return starts
  }

  /** float32 mean/variance computed in double, as a close stand-in for numpy's float32 reductions. */
  private fun zscore(x: FloatArray): FloatArray {
    var s = 0.0
    for (v in x) s += v
    val mean = (s / x.size).toFloat()
    var ss = 0.0
    for (v in x) { val d = (v - mean).toDouble(); ss += d * d }
    val denom = sqrt((ss / x.size).toFloat() + 1e-5f)
    return FloatArray(x.size) { (x[it] - mean) / denom }
  }

  fun makeWindows(x16: FloatArray, mode: String): Pair<List<FloatArray>, List<Int>> {
    return when (mode) {
      "train_peak", "train_zscore" -> {
        val x = if (mode == "train_peak") {
          var peak = 0f
          for (v in x16) { val a = kotlin.math.abs(v); if (a > peak) peak = a }
          val scale = (10.0.pow(PEAK_TARGET_DBFS / 20) / (peak + 1e-6)).toFloat()
          FloatArray(x16.size) { x16[it] * scale }
        } else zscore(x16)
        val starts = windowStarts(x.size)
        Pair(starts.map { s -> FloatArray(WIN).also { w -> x.copyInto(w, 0, s, minOf(s + WIN, x.size)) } }, starts)
      }
      "api_replica" -> {
        val w = FloatArray(48000).also { x16.copyInto(it, 0, 0, minOf(48000, x16.size)) }
        val z = zscore(w)
        Pair(listOf(FloatArray(WIN).also { z.copyInto(it, 0, 0, 48000) }), listOf(0))
      }
      else -> throw IllegalArgumentException("unknown preprocessing mode '$mode'; use one of $MODES")
    }
  }

  fun softmax(logits: FloatArray): DoubleArray {
    val m = logits.max().toDouble()
    val e = logits.map { exp(it - m) }
    val s = e.sum()
    return e.map { it / s }.toDoubleArray()
  }

  /** Clip score = mean of window synthetic scores; label uses the 4-dp rounded mean like Python. */
  fun label(meanScore: Double): String =
    if (round(meanScore, 4) >= DECISION_THRESHOLD) "synthetic" else "genuine"
}
