package com.hypenex.vcddetector

/**
 * Baseline adapter: third-party Dhwani checkpoint (ayush2635, MIT; not trained by this project).
 * Preprocessing is the port of src/vcd/dhwani_infer.py in Preprocess.kt. Verified on the
 * emulator against the Colab reference export (runtime |dlogit| <= 2.4e-6 on identical inputs).
 */
object DhwaniAdapter : DetectorAdapter {
  override val spec = AdapterSpec(
    id = "dhwani",
    displayName = "Dhwani baseline (XLS-R 300M + simplified GAT)",
    version = "1.0.0",
    model = ModelArtifact(
      repo = "ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model",
      revision = "ee0187142973e1f02050e47d2a31c55ee0263065",
      file = "best_model.onnx",
      sha256 = "d1c232bf4d7990526804b375c15804c0ee4e9b566478b908c5a57431491b7342",
      bytes = 1262760439L,
      url = "https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model/resolve/" +
        "ee0187142973e1f02050e47d2a31c55ee0263065/best_model.onnx",
      licence = "MIT (per model card)",
    ),
    sampleRate = Preprocess.SR,
    windowSamples = Preprocess.WIN,
    hopSamples = Preprocess.HOP,
    minDurationS = Preprocess.MIN_DUR_S,
    maxDurationS = Preprocess.MAX_DUR_S,
    batchSize = 1, // a batch of 2 returns one row (Colab run 20261024T102407Z)
    preprocessingModes = Preprocess.MODES,
    defaultMode = "train_peak", // provisional, not validated
    outputLabels = listOf("genuine", "synthetic"),
    positiveIndex = Preprocess.SYNTHETIC_INDEX,
    aggregation = "mean of window softmax[positiveIndex]; label uses the 4-dp rounded mean",
    decisionThreshold = Preprocess.DECISION_THRESHOLD,
    thresholdTuned = false,
    scoreCalibrated = false,
    attribution = "Dhwani by ayush2635 (MIT), HF ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model@ee01871; " +
      "not trained by this project",
  )

  override fun validate(x: FloatArray) = Preprocess.validate(x)
  override fun windows(x: FloatArray, mode: String) = Preprocess.makeWindows(x, mode)
  override fun windowScore(output: FloatArray) = Preprocess.softmax(output)[spec.positiveIndex]
  override fun aggregate(scores: DoubleArray) = scores.average()
  override fun label(clipScore: Double) = Preprocess.label(clipScore)
}
