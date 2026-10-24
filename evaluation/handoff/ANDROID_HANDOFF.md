# Android integration hand-off — AntiDeepfake Wav2Vec2-Small (verified candidate, no denoising)

For the development chat. Status 2026-10-24. The evaluation side does not touch `mobile/`; this is the specification the mobile work integrates against.

## What is handed over
A zip produced by the Colab cell `evaluation/colab_cells/14_package_for_android.py` (run in the notebook-04 runtime after `12b`), named
`antideepfake_small_onnx_pkg_<UTC>.zip`, containing:

| File | Purpose |
|---|---|
| `antideepfake_small.onnx` (+ any `*.onnx.data` external-weight file the exporter produced) | fp32 model, 378.8 MB, dynamic time axis |
| `SHA256SUMS` | checksums of every file in the package |
| `MODEL_SPEC.json` | machine-readable spec: files + SHA256, I/O names/shapes, exact preprocessing, threshold, reference vectors, licence, evaluation evidence |
| `reference_inputs/*.npy`, `ref_wav_16k_pcm16_4s.wav` | parity vectors (see below) |
| `LICENSE_ATTRIBUTION.md`, `HANDOFF.md` | attribution + short spec |

Source weights: HF `nii-yamagishilab/wav2vec-small-anti-deepfake` @ `9a13264b…`, `model.safetensors` sha256 `828ee456…fcfe0`. Weights untouched; converted fairseq → transformers (parity with the authors' code verified to 4.6e-7) → ONNX (ORT vs PyTorch 2.1e-7 on 8 clips).

## Interface
- **Input** `wav`: float32 `[1, T]`, normalised mono 16 kHz waveform. Policy range T = 32,000–160,000 (2–10 s); the export's declared range is 16,000–160,000. **Batch size 1 only.**
- **Output**: float32 `[1, 2]` logits in the order **[fake, real]** (name recorded in `MODEL_SPEC.json` → `io.output.name`).
- `synthetic_score = softmax(logits)[0]`, `genuine_score = softmax(logits)[1]`.
- **Decision:** synthetic if `synthetic_score >= 0.5`. 0.5 is the frozen *provisional* threshold (not tuned). A validation-selected threshold may replace it later; keep it a single constant.

## Preprocessing (whole clip; exactly what was evaluated)
1. Decode to float in [−1, 1] (PCM16 ÷ 32768; AAC/Opus through the platform decoder).
2. Mono = mean over channels.
3. Resample to 16,000 Hz with a high-quality (windowed-sinc) resampler; evaluation used libsoxr HQ. Record which resampler the app uses.
4. Gates (reject, never pad or substitute): < 2.0 s → reject; > 10.0 s → analyse the first 10.0 s only (flag "truncated"); peak < −50 dBFS → reject (silent); less than 1.0 s of 25 ms frames that are within 35 dB of the loudest frame *and* above −55 dBFS → reject (mostly silent); any non-finite sample → reject. (Same gates as `src/vcd/dhwani_infer.validate`.)
5. Normalise over the **whole analysed clip**: `x = (x − mean(x)) / sqrt(var(x) + 1e-5)`, biased variance (divide by N), float32. **Epsilon 1e-5.**
6. One forward pass on the whole clip. No 4 s windowing, no cropping, no padding, no peak normalisation, no per-window z-score (those were the Dhwani recipe).
7. Softmax, score, decision as above. The score is an uncalibrated model output: display it as a score, never as a probability.

Do **not** add a denoiser in front of the model: that is an open experiment (notebook 06) and is not part of this hand-off.

## Parity vectors
`reference_inputs/<name>.normalised_input_float32.npy` are ready-to-feed inputs (2 s, 4 s, 10 s deterministic signals plus the decoded WAV); `<name>.raw_float32.npy` are the same signals before normalisation, so the app's normalisation can be checked too; `ref_wav_16k_pcm16_4s.wav` exercises the full decode → normalise → infer path. Expected logits and scores for each are in `MODEL_SPEC.json` → `reference_vectors`. fp32 ORT on Android should agree within 1e-3 on logits (1e-4 on the score). No dataset audio is included (In-the-Wild/MLAAD audio must not be redistributed).

## Size, latency, quantisation
378.8 MB fp32. Colab 2-vCPU ORT: median 568 ms per clip (not a phone number). On a phone CPU expect roughly 0.5–3 s per clip; measure. ONNX Runtime Mobile/Android is the intended runtime; the graph is plain wav2vec2 (conv + transformer + mean-pool + linear), opset recorded in the spec.
**fp16 / int8 are not evaluated.** A quantised build must go back through the notebook-04/05 protocol (fresh selection), not only a parity check.

## What the numbers mean (and don't)
- Held-out In-the-Wild pilot subset, 244 clips: ROC-AUC 0.996, EER 2.5 %; at 0.5: accuracy 0.967, precision 0.975, recall 0.960, FPR 0.025.
- Robustness (paired, 244 clips): 8 kHz band-limit EER 4.1 %, Opus 16 kb/s 4.1 %, MP3 32 kb/s 2.5 %, **white noise 10 dB SNR 16.0 % → fails the ≤ 2.5× ratio requirement**. Loudness changes: no effect. 1 s silence pad +1.6 pp, first-3 s +2.5 pp, telephone band + μ-law +4.9 pp.
- Not evaluated: Hindi/Indic speech, real phone-call or WhatsApp audio, new speakers (In-the-Wild shares speakers across all our subsets), modern unseen generators. English celebrity data only.
- Input gate note for the app: at −20 dB gain, 18 of 244 clips became "mostly silent" and were rejected; quiet voice notes may hit this gate.

## Licence and attribution (blocking for release)
Weights are **CC BY-NC-SA 4.0** (NII): research and educational use. A commercial release needs NII's permission; derived ONNX inherits the licence. Base model `facebook/wav2vec2-base` is Apache-2.0. Show attribution in the app: "Detector: AntiDeepfake (Yamagishi Lab, NII), CC BY-NC-SA 4.0", and cite arXiv:2506.21090.

## Open items on the evaluation side
Noise robustness (denoiser experiment, then fine-tuning if needed), Indic/phone-channel evaluation, validation-selected threshold, quantised-model evaluation.
