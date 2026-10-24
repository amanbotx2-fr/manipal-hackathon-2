package com.hypenex.vcdcalling

import com.hypenex.vcddetector.DetectorEngine

/**
 * Scores the remote participant's audio during a call. Keeps a rolling buffer of the last 10 s of
 * 16 kHz mono PCM delivered by WebRtcCallView (bridge thread, 320 samples per call) and, every
 * `hop` samples, submits the latest `window` samples to the shared DetectorEngine on its own
 * inference thread. If a window is still being scored when the next is due, the new one is
 * skipped (counted as dropped), never queued. Local microphone audio is never scored.
 */
class LiveAnalyser(
  private val engine: DetectorEngine,
  private val emit: (Map<String, Any?>) -> Unit,
) {
  private val sr = engine.adapter.spec.sampleRate
  private val window = engine.adapter.spec.windowSamples
  private val hop = engine.adapter.spec.hopSamples
  private val mode = engine.adapter.spec.defaultMode
  private val cap = sr * 10
  private val ring = FloatArray(cap)
  private var written = 0L
  private var lastScoredAt = 0L
  private val lock = Any()
  @Volatile private var busy = false
  @Volatile private var closed = false
  private var windows = 0
  private var dropped = 0

  fun push(samples: FloatArray) {
    if (closed) return
    var due = false
    synchronized(lock) {
      for (s in samples) { ring[(written % cap).toInt()] = s; written++ }
      if (written >= window && written - lastScoredAt >= hop) { due = true; lastScoredAt = written }
    }
    if (!due) return
    if (busy) { synchronized(lock) { dropped++ }; return }
    busy = true
    val snap = synchronized(lock) { snapshot(window) }
    val audioEndS = written.toDouble() / sr
    engine.submit(
      { engine.analysePcmFloats(snap, sr, 1, mode) },
      { r ->
        busy = false
        if (closed) return@submit
        val idx: Int; val drop: Int
        synchronized(lock) { windows++; idx = windows; drop = dropped }
        @Suppress("UNCHECKED_CAST")
        emit((r as Map<String, Any?>) + mapOf("window_index" to idx, "dropped" to drop,
          "audio_end_s" to audioEndS, "window_s" to window.toDouble() / sr, "kind" to "window"))
      },
      { t ->
        busy = false
        if (!closed) emit(mapOf("kind" to "error", "status" to "error",
          "message" to "${t.javaClass.simpleName}: ${t.message}", "audio_end_s" to audioEndS))
      },
    )
  }

  private fun snapshot(n: Int): FloatArray {
    val out = FloatArray(n)
    val start = written - n
    for (i in 0 until n) out[i] = ring[((start + i) % cap).toInt()]
    return out
  }

  fun stats(): Map<String, Any?> = synchronized(lock) {
    mapOf("audio_s" to written.toDouble() / sr, "windows" to windows, "dropped" to dropped, "busy" to busy)
  }

  fun close() { closed = true }
}
