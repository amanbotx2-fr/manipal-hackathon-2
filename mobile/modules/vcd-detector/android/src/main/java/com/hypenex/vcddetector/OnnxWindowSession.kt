package com.hypenex.vcddetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Must match the onnxruntime-android dependency in android/build.gradle. */
const val ORT_VERSION = "1.31.0"

/**
 * One ONNX Runtime session running one [1, n] float32 window at a time (model-agnostic; adapters
 * with batchSize 1 such as Dhwani require this: its graph returns one row for a batch of 2).
 * Not thread-safe by design; callers serialise through DetectorEngine's executor.
 */
class OnnxWindowSession private constructor(
  private val env: OrtEnvironment,
  private val session: OrtSession,
  val inputName: String,
  val settings: Map<String, Any?>,
  val loadMs: Double,
) : AutoCloseable {

  /** Runs one preprocessed [1, n] window; returns the model's output row. */
  fun run(window: FloatArray): FloatArray {
    val buf = ByteBuffer.allocateDirect(window.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    buf.put(window).rewind()
    OnnxTensor.createTensor(env, buf, longArrayOf(1, window.size.toLong())).use { input ->
      session.run(mapOf(inputName to input)).use { result ->
        @Suppress("UNCHECKED_CAST")
        val out = result[0].value as Array<FloatArray>
        check(out.size == 1) { "unexpected output batch ${out.size} (batch size 1 expected)" }
        return out[0].copyOf()
      }
    }
  }

  override fun close() {
    session.close()
  }

  companion object {
    fun open(model: File, intraOpThreads: Int): OnnxWindowSession {
      val env = OrtEnvironment.getEnvironment()
      val t0 = System.nanoTime()
      val opts = OrtSession.SessionOptions().apply {
        // Same optimisation level as Colab's default session (ORT_ENABLE_ALL).
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setIntraOpNumThreads(intraOpThreads)
        setInterOpNumThreads(1)
      }
      val session = env.createSession(model.absolutePath, opts)
      val loadMs = (System.nanoTime() - t0) / 1e6
      val inputName = session.inputNames.first()
      val info = session.inputInfo[inputName]?.info?.toString()
      val outInfo = session.outputInfo.values.firstOrNull()?.info?.toString()
      val settings = mapOf(
        "onnxruntime" to "onnxruntime-android $ORT_VERSION",
        "executionProvider" to "CPU",
        "optimizationLevel" to "ALL_OPT",
        "executionMode" to "SEQUENTIAL",
        "intraOpThreads" to intraOpThreads,
        "interOpThreads" to 1,
        "batchSize" to 1,
        "input" to "$inputName $info",
        "output" to outInfo,
      )
      return OnnxWindowSession(env, session, inputName, settings, loadMs)
    }
  }
}
