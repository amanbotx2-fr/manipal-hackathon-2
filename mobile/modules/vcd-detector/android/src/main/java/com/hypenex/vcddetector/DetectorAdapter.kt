package com.hypenex.vcddetector

/**
 * Model-replaceable detection contract. DetectorEngine (decode, resample, session, threading,
 * timing, reference checks) knows nothing model-specific; everything that differs between
 * models lives behind this interface and its declarative AdapterSpec.
 *
 * Adding a model = one new DetectorAdapter + registering it in Adapters. Only adapters for
 * checkpoints that passed evaluation should be registered (Dhwani is the baseline).
 */

/** A pinned, checksummed ONNX file downloaded once into app-private storage. */
data class ModelArtifact(
  val repo: String,
  val revision: String,
  val file: String,
  val sha256: String,
  val bytes: Long,
  val url: String,
  val licence: String,
)

data class AdapterSpec(
  val id: String,                     // stable id; also names the model directory (id-rev7)
  val displayName: String,
  val version: String,                // adapter (preprocessing/scoring) version, not the model's
  val model: ModelArtifact,
  val sampleRate: Int,                // model input rate (Hz), mono float32
  val windowSamples: Int,             // samples per model call
  val hopSamples: Int,
  val minDurationS: Double,
  val maxDurationS: Double,           // longer audio: first maxDurationS analysed, flagged
  val batchSize: Int,                 // supported batch size (Dhwani: 1)
  val preprocessingModes: List<String>,
  val defaultMode: String,
  val outputLabels: List<String>,     // index -> label of the model's output vector
  val positiveIndex: Int,             // index whose softmax is the "synthetic" score
  val aggregation: String,            // how window scores become a clip score
  val decisionThreshold: Double,
  val thresholdTuned: Boolean,        // false until chosen on a validation split
  val scoreCalibrated: Boolean,       // false: scores are not probabilities
  val attribution: String,
) {
  fun toMap(): Map<String, Any?> = mapOf(
    "id" to id, "displayName" to displayName, "version" to version,
    "model" to mapOf("repo" to model.repo, "revision" to model.revision, "file" to model.file,
      "sha256" to model.sha256, "bytes" to model.bytes, "licence" to model.licence),
    "sampleRate" to sampleRate, "windowSamples" to windowSamples, "hopSamples" to hopSamples,
    "minDurationS" to minDurationS, "maxDurationS" to maxDurationS, "batchSize" to batchSize,
    "preprocessingModes" to preprocessingModes, "defaultMode" to defaultMode,
    "outputLabels" to outputLabels, "positiveIndex" to positiveIndex, "aggregation" to aggregation,
    "decisionThreshold" to decisionThreshold, "thresholdTuned" to thresholdTuned,
    "scoreCalibrated" to scoreCalibrated, "attribution" to attribution,
  )
}

class AudioRejected(message: String) : Exception(message)

/** Audio after the adapter's validation (possibly truncated), plus stats for the result. */
class ValidatedAudio(val x: FloatArray, val info: Map<String, Any>)

interface DetectorAdapter {
  val spec: AdapterSpec

  /** Duration/silence gates on mono audio at spec.sampleRate. Throws AudioRejected. */
  fun validate(x: FloatArray): ValidatedAudio

  /** Model-ready windows (each spec.windowSamples long) and their start sample indices. */
  fun windows(x: FloatArray, mode: String): Pair<List<FloatArray>, List<Int>>

  /** Model output for one window -> uncalibrated positive-class score. */
  fun windowScore(output: FloatArray): Double

  /** Window scores -> clip score. */
  fun aggregate(scores: DoubleArray): Double

  /** Clip score -> label using spec.decisionThreshold. */
  fun label(clipScore: Double): String
}

/** Registered adapters. Only evaluated, verified checkpoints belong here. */
object Adapters {
  val default: DetectorAdapter = DhwaniAdapter
  private val all = listOf<DetectorAdapter>(DhwaniAdapter)
  fun byId(id: String): DetectorAdapter? = all.firstOrNull { it.spec.id == id }
  fun list(): List<AdapterSpec> = all.map { it.spec }
}
