# Implementation audit: our Dhwani pipeline vs upstream code

Date 2026-10-24. Purpose: before anything is changed, find concrete errors in how we run the third-party Dhwani checkpoint, by comparing `src/vcd/dhwani_infer.py` (read only; snapshot sha256 `8a7c9c8f…6077`, the code that produced run 20261024T123319Z) with the upstream repository `ayush2635/Dhwani-Deepfake-Audio-Detection-API` @ `efd512c` (`training/training_notebook.ipynb`, `training/dataset.py`, `src/api/inference.py`, `src/api/main.py`) and the HF model card @ `ee01871`.

**Result: no concrete implementation bug found in label mapping, decoding, normalisation or window aggregation.** The failed baseline stands as a property of the checkpoint on this data under the pre-registered recipe. Details and evidence below; every diagnostic uses the saved predictions only (no re-scoring).

## 1. Label mapping: consistent, and reversal would not help

| Evidence | Upstream | Ours |
|---|---|---|
| Dataset labels | `training_notebook.ipynb` cell 4 and `dataset.py` L41: `labels = [0]*len(real_files) + [1]*len(fake_files)` | `LABELS = {0: "genuine", 1: "synthetic"}` |
| Loss / head | `train.py` (cell 11): `CrossEntropyLoss` over a 2-class `fc`, `torch.max(outputs, 1)` for accuracy, i.e. logits are class scores in label order | softmax over the 2 logits |
| Deployed API | `inference.py` L95: `probs_list.append(probs[0][1])` = fake probability; model card: `fake_probability = probs[0][1]` | `SYNTHETIC_INDEX = 1` |
| Empirical | Smoke test (run 20261024T102407Z): Aman's genuine clip 0.0001, MLAAD Chatterbox clip 0.9998 | consistent with index 1 = synthetic |

Pilot evidence: AUC as run is 0.456; with reversed labels it would be 1 − 0.456 = 0.544. Both are chance. Reversing the mapping would not rescue the result and is not supported by the smoke test, so **the mapping stays as is**.

## 2. Decoding and resampling: no differences that could act on this pilot

- Upstream training decodes with `torchaudio.load` (floats in [−1, 1]), resamples with `torchaudio.transforms.Resample`, mixes to mono by mean (cell 4 L54–63). The API uses `librosa.load(sr=None, mono=False)` then the same.
- Ours: `soundfile.read(dtype="float32")`, with a librosa/audioread fallback for AAC etc.; mono by mean; resample with librosa `soxr_hq`. Same value range and mono rule; the resampler differs (sinc interpolation vs soxr), a negligible difference.
- **For this pilot, decoding differences cannot be the cause:** all 200 In-the-Wild files are `WAV/FLOAT`, mono, 16 kHz (from `predictions.jsonl`: `container_format`, `channels`, `native_sr`). No resampling or channel mixing happened. soundfile reads 32-bit float WAV natively. One genuine clip peaks at +0.58 dBFS (float WAV may exceed 1.0); peak normalisation handles it.
- Zero rows failed to decode. All 45 exclusions are `too short` (< 2.0 s), a documented policy; upstream would have zero-padded these to 4 s instead. Not a bug, but it is a coverage difference versus upstream behaviour. Upstream additionally **replaces** unreadable files with zeros (cell 4 L104); we reject. We consider ours correct.

## 3. Normalisation: `train_peak` is an exact match of the training Dataset

Upstream `training_notebook.ipynb` cell 4 (the `Dataset` used by the training run that exported `best_model.onnx` in cell 14):
```
max_val = torch.abs(waveform).max()
if max_val > 0:
    target_db = -3.0
    target_amp = 10 ** (target_db / 20)
    gain = target_amp / (max_val + 1e-6)
    waveform = waveform * gain
```
Ours (`make_windows`, `train_peak`): `x = x16 * ((10 ** (PEAK_TARGET_DBFS / 20)) / (peak + 1e-6))` with `PEAK_TARGET_DBFS = -3.0`. Identical; the `max_val > 0` guard is covered by our silence rejection.

Two residual, documented differences:
- Training normalised the **whole file** and then took a random 4 s crop; we normalise the analysed clip (first 10 s). For the 25 truncated pilot clips the peak could lie in the discarded tail, changing the gain slightly. Order of magnitude: a few dB at most; not a bug.
- Training applied noise (SNR 10–20 dB) and 8 kHz down/up-sampling to **genuine training clips only**, 30 % each (cell 4 L75–82). Nothing in inference should replicate that; it is a known shortcut risk in the model, not in our code.

The z-score recipe (`(w - mean)/sqrt(var + 1e-5)`) appears only in the notebook's inference helper (cell 17 L30), the API (`inference.py` L58) and the card. Ours implements it as `train_zscore` / `api_replica` for comparison; it was **not** used in this pilot, by design. The ONNX export used `torch.randn(1, 64000)` as the dummy input (cell 14), which says nothing about the expected scale.

## 4. Windowing and aggregation: same windows; aggregation rule differs only in a corner case, which does not change the outcome

| Aspect | Upstream API (`inference.py` L65–113) | Ours |
|---|---|---|
| Window / stride | 64,000 / 32,000 samples | same |
| ≤ 4 s | one window, zero-padded at the end | same |
| Tail | windows `range(0, n - 64000 + 1, 32000)`; the last < 2 s of a longer clip is never scored | extra last window aligned to the clip end (full coverage) |
| Temperature | logits / 1.362 (unpublished calibration) | T = 1 (documented decision) |
| Clip score | mean of window probabilities, **except** if two consecutive windows (or a single window) exceed 0.90, then the max | mean (max also stored) |
| Crop | API `preprocess` first truncates to 48,000 samples (3 s) *before* windowing, so in the deployed API only one 3 s window ever exists | 2–10 s analysed |

Diagnostics on the saved window logits (155 scored clips; **informational only, no rule is adopted from this**):

| Aggregation | ROC-AUC | EER |
|---|---|---|
| mean (as run) | 0.456 | 0.561 |
| max | 0.470 | 0.548 |
| first window only | 0.443 | 0.574 |
| upstream rule with T = 1.362 | 0.456 | 0.561 |
| mean of (logit₁ − logit₀) | 0.454 | 0.574 |

Every variant is at chance. Within-clip window scores are extremely consistent (92 multi-window clips, median max–min range 0.0005; only 6 clips vary by more than 0.5), so the aggregation rule is almost irrelevant: the model gives one verdict per recording.

## 5. Batch size 1: explained

`architecture.py` (cell 7) `AASISTBackEnd.forward` iterates `for i in range(batch_size)` in Python. `torch.onnx.export` traced it with a batch-1 dummy input (cell 14), so the ONNX graph hard-codes a single iteration even though `batch_size` is declared dynamic. That is why a batch of 2 returns `[1, 2]`. Our batch-1 loop is the correct workaround.

## 6. What the pilot predictions do show (observations, not decisions)

- Scores are saturated in both classes: 48/75 genuine and 51/80 synthetic clips score below 0.01; 6 and 9 above 0.99.
- Flagged clips cluster by **speaker/recording**: 23 of 53 scored speakers have a flagged clip, and for several speakers *both* genuine clips are flagged (Boris Johnson, Bob Ross, George Carlin, Christopher Hitchens, John Cleese) while for others *both* synthetic clips are flagged with genuine passing (Adam Driver, Scarlett Johansson, Kamala Harris). This looks like sensitivity to recording identity or channel, not to synthesis.
- A score ≥ 0.5 is mildly associated with louder audio (AUC of peak dBFS predicting "flagged" 0.65; RMS 0.61). Loudness itself does not separate the classes (AUC 0.44 / 0.38). Suggestive of a channel/loudness shortcut, consistent with audit finding 2 in `docs/UPSTREAM_AUDIT.md` (augmentation on genuine only). Weak evidence, n = 155.
- Zero-padded ≤ 4 s clips and multi-window clips behave alike.

## 7. Conclusion

No evidence of an error in our pipeline. The things that differ from upstream (tail window, T = 1, mean vs. consistency-max, 2 s minimum) are documented decisions and the diagnostics show none of them changes the outcome. Run 20261024T123319Z is recorded as the **failed Dhwani baseline**; threshold, label mapping and preprocessing remain untouched, and the 200 clips are development evidence, excluded from any future held-out test.

Open, unexplained item: In-the-Wild `meta.csv` has 54 distinct speaker strings, not 58. No effect on the result.
