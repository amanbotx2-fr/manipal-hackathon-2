# Run 20261024T151905Z — paired robustness, EER-first (AntiDeepfake Wav2Vec2-Small)

Source zip sha256 `17481765…8ea3`, unpacked unchanged. Ran as cells appended to notebook 04 (same T4 runtime; model and cached audio reused). Executed `.ipynb` not yet returned.

> **Scope:** the 244 eligible clips of the held-out In-the-Wild subset (run 20261024T140853Z). Their clean scores were already inspected, so this run
> made **no decision**: threshold frozen at **0.5** (no validation-selected threshold exists yet), nothing tuned. Paired per clip.
> EER is on continuous scores; the EER crossing thresholds below are **descriptive only** and are never used for classification.

## Setup
ffmpeg `/usr/bin/ffmpeg` with `libopus` and `libmp3lame` present → all required conditions ran. Noise seed `sha256("20261027:<clip_id>")`.
Conditions exactly as in `degrade.py` (`run_config.json` → `settings.conditions`). Dhwani not re-run (`RUN_DHWANI = False`).

## Coverage (nothing removed silently)
All 300 clips attempted per condition; 56 are ineligible under the clean 2.0 s minimum in every condition. Degraded-stage rejections, all through the
same validation gate as production input:
- `loud_minus20db`: **18 rejected** as *mostly silent* (< 1 s of signal within 35 dB of the loudest frame and above −55 dBFS); 12 genuine, 6 synthetic. Scored n = 226.
- `duration_first3s`: 1 rejected (mostly silent, genuine). n = 243.
- Every required condition: 244 / 244 scored. **Paired set = 244 clips.**

## Results on the paired set (recomputed locally from `predictions_conditions.jsonl`; identical to `metrics.json`)

| Condition | Group | EER [95 % boot] | ROC-AUC | EER thr (descr.) | @ 0.5: acc / prec / rec / F1 / FPR | Flips at 0.5 vs clean |
|---|---|---|---|---|---|---|
| clean 16 kHz | required | **2.46 %** [0.8, 5.7] | 0.9958 | 0.178 | 0.967 / 0.975 / 0.960 / 0.967 / 0.025 | — |
| 16→8→16 kHz | required | 4.10 % [1.2, 6.2] | 0.9960 | 0.143 | 0.971 / 1.000 / 0.944 / 0.971 / 0.000 | 9 |
| Opus 16 kb/s voip | required | 4.10 % [1.6, 7.4] | 0.9936 | 0.748 | 0.951 / 0.944 / 0.960 / 0.952 / 0.058 | 10 |
| MP3 32 kb/s | required | 2.46 % [0.8, 4.9] | 0.9960 | 0.203 | 0.963 / 0.983 / 0.944 / 0.963 / 0.017 | 9 |
| White noise 10 dB SNR | required | **15.98 %** [11.5, 21.3] | 0.9064 | 0.055 | 0.799 / 0.921 / **0.661** / 0.770 / 0.058 | **47** |
| −20 dB gain (n = 226) | sensitivity | 2.66 % [0.9, 6.2] | 0.9956 | 0.178 | 0.965 / 0.974 / 0.958 / 0.966 / 0.028 | 0 |
| peak −1 dBFS | sensitivity | 2.46 % [0.8, 5.7] | 0.9958 | 0.178 | identical to clean | 0 |
| 1 s silence pad | sensitivity | 4.10 % [0.8, 6.6] | 0.9938 | 0.235 | 0.971 / 0.992 / 0.952 / 0.971 / 0.008 | 5 |
| first 3 s only (n = 243) | sensitivity | 4.94 % [2.1, 8.2] | 0.9872 | 0.137 | 0.955 / 0.975 / 0.935 / 0.955 / 0.025 | 5 |
| telephone 300–3400 Hz + μ-law | sensitivity | 7.38 % [3.3, 10.3] | 0.9854 | 0.385 | 0.947 / 0.974 / 0.919 / 0.946 / 0.025 | 13 |

Bootstrap CIs resample clips only (speaker clustering ignored → optimistic). With 120/124 clips per class one error ≈ 0.8 pp of EER.

## Official verdict (required conditions, paired set)
- Worst degraded condition: **noise**, EER 15.98 %. Ratio worst / clean = 15.98 / 2.46 = **6.50 > 2.5 → FAIL.**
- Without the noise condition, the worst ratio would be 4.10 / 2.46 = 1.67 (narrowband, Opus). Reported for context; it does not change the verdict.
- **EER(8 kHz) − EER(clean) = +1.64 percentage points** (4 extra clip-errors' worth).

## What the noise failure looks like
- Direction is one-sided: under 10 dB white noise, **synthetic clips move toward "genuine"** (92 % of spoofs' scores decrease; median shift −0.08, mean −0.30; 41 spoofs flip from ≈ 1.0 to a median 0.06), while genuine clips barely move (median +0.002; 6 flip).
- It is not a short-clip artefact: noisy EER is 14.5 % for 2–4 s clips and 13.0 % for 4–10 s clips (clean: 4.8 % / 2.2 %).
- Interpretation (hypothesis, not established): broadband noise masks the high-frequency synthesis artefacts the model relies on, so noisy spoofs look like genuine recordings. This is the same failure direction the Dhwani audit predicted for a model whose training augmentation adds noise to the real class; AntiDeepfake used RawBoost augmentation, but its published Deepfake-Eval-2024 result (33 % EER zero-shot) already pointed to weakness on noisy real-world audio.
- Opus is the only condition that pushes **genuine** scores up (p95 shift +0.41; descriptive EER threshold 0.75): a few genuine clips sound "synthetic" after Opus 16 kb/s. Relevant for WhatsApp-style voice notes.

## Consequences (no action taken here)
- The candidate PASSES bandwidth and codec degradation comfortably and FAILS the additive-noise requirement at 10 dB SNR under the ≤ 2.5 ratio rule.
- Options to be decided (not done): (a) noise-robust fine-tuning of the candidate on separate train/validation data with **both-class** noise augmentation (README §6 plan), selecting checkpoints by validation EER; (b) an input-side denoiser before the detector, evaluated with the same paired protocol; (c) report the noise condition as a known limitation. Any of these must be validated on a **fresh** selection (exclude the 200 + 300 inspected clip IDs) and the deployment threshold chosen on validation data only.
- Unseen-generator evaluation: not performed (cannot be established from In-the-Wild/MLAAD; no user-supplied generator clips).
- ONNX export retry result (`onnx_retry.json`) not yet received for run 20261024T140853Z.
