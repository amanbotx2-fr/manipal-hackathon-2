# Upstream pins and audit

Pinned 2026-10-24. All third-party; none of these weights were trained by us.

## Pins

| Resource | Revision | Licence | Notes |
|---|---|---|---|
| HF model `ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model` | `ee0187142973e1f02050e47d2a31c55ee0263065` | MIT (card) | Only `best_model.onnx` (1,262,760,439 B, sha256 `d1c232bf4d7990526804b375c15804c0ee4e9b566478b908c5a57431491b7342`). **No PyTorch `.pth` published.** |
| GitHub `ayush2635/Dhwani-Deepfake-Audio-Detection-API` | `efd512c925a4a75271973fcd9e2cfafe3c8e9f44` (single commit, 2026-06-13) | see repo LICENSE | Training notebook, API, scripts |
| GitHub `Yaselley/deepfense-framework` | `a00e76b3fe5bcdb867dfeaa7e139bf9bd4dc69a4` | not yet reviewed | Candidate training framework |
| HF `facebook/wav2vec2-xls-r-300m` | `1a640f32ac3e39899438a2931f9924c02f080a54` | Apache-2.0 | Encoder |
| HF dataset `vdivyasharma/IndicSynth` | `c0a10386b723717aff682f757bd67f72983f269f` | CC-BY-NC-4.0 | 12 language configs, parquet; Bengali alone ~34 GB download. Fields: `Generative Model`, `Source Speaker_ID`, `Target Speaker ID`, `Gender`, `Source/Target Reference Audio`, `TTS Transcript` |
| GitHub `vdivyas/IndicSynth` | `9dc02bdde41c99a9730b3923f548ba75ffb8154b` | — | |
| GitHub `AI4Bharat/IndicSUPERB` | `8860e5c39c726186a0a36f2163cf30397f398e0d` | — | Kathbath genuine speech |
| HF dataset `mueller91/MLAAD` | `30c3dec763fa4803f11c2df1557d3b0828395366` | CC-BY-NC-4.0, **gated (auto)** | Requires accepting terms on HF |

## Dhwani audit findings

1. **Preprocessing conflict.** Evidence (repo @ `efd512c`; notebook line numbers are within the cell source):

   | Location | What it does |
   |---|---|
   | `training/training_notebook.ipynb` cell 4, L74–77 | `target_db = -3.0` … `waveform = waveform * gain` → **peak-normalise to −3 dBFS** in the training `Dataset` |
   | same cell, L90 | `max_len = 64000` → random **4 s** crop, zero-pad at end. No z-score anywhere in the training `Dataset` |
   | same notebook, cell 14 L3–5 | `dummy_input = torch.randn(1, 64000)`; `torch.onnx.export(...)` → `best_model.onnx` (this is the run the HF file is named after) |
   | same notebook, cell 17 L30, L37 | inference helper: **z-score** (`(w - mean)/sqrt(var+1e-5)`), 64,000-sample window |
   | `src/api/inference.py` L50, L58 | deployed API: crop to **48,000** (3 s), then **z-score** |
   | `src/api/inference.py` L66 | `window_size = 64000` → the 3 s clip is zero-padded to 4 s before the model |
   | `src/api/inference.py` L92 | `T = 1.362` temperature applied to logits |
   | `src/api/main.py` L24–25 | comment says "(1, 48000)" but code uses `(1, 64000)` |
   | `training/dataset.py` L63 | standalone script: 48,000 samples, no normalisation (appears stale vs. the notebook) |
   | HF model card | 48,000 samples + z-score |

   Conclusion: the training code we can see used peak normalisation + 4 s; every published inference path uses z-score (+3 s in the API). We cannot tell from code alone which the uploaded checkpoint expects (the published notebook may not be the exact run, and peak vs z-score differ only by a per-clip gain plus DC offset, which the model may or may not be sensitive to).

   **We do not pick silently.** The Colab notebook scores every clip under three controlled modes:
   A `train_peak` (training recipe), B `train_zscore` (same windows, z-score only), C `api_replica` (exact API). A↔B isolates normalisation, B↔C isolates crop/padding. Decide after seeing results on labelled clips; record the decision in `PROGRESS.md`.
2. **Augmentation only on genuine class.** Noise (SNR 10–20 dB) and 8 kHz down/up-sampling were applied only to *real* training clips → the model may have learned "noisy / band-limited ⇒ genuine". Direct shortcut risk for phone audio. Test with telephone-band synthetic clips.
3. **Dataset-origin confound.** Card says genuine = Common Voice v24 (Indic) + garystafford (EN); fake = IndicSynth + garystafford. Indic genuine and Indic fake come from different corpora (IndicSynth is derived from Kathbath, not Common Voice), so corpus/channel identity may separate classes.
4. **Random file-level split** (`random_split`) — no speaker/generator/source-reference grouping. Reported validation numbers (if any) are not evidence of generalisation.
5. **Zero-waveform fallback** on decode failure — a failed file becomes a labelled all-zero clip. We never do this; unreadable input is rejected.
6. **Temperature T = 1.362** hard-coded in the API, described as calibrated; no calibration data/log is published. We report raw logits and T=1 softmax only.
7. **Backend is a simplified GAT** (2 fully connected GAT layers over XLS-R frames + mean pool), not official AASIST.
8. **Label order**: `labels = [0]*real + [1]*fake`, and the API treats `softmax[:,1]` as fake. Documented, not yet empirically confirmed.
9. **Interface** (measured in Colab, run 20261024T102407Z, onnxruntime 1.31.0 CPU): input `[batch_size, time]`, output `[1, 2]`. Lengths 32k/48k/64k/96k samples all run; output is deterministic. **Batch of 2 returns shape `[1, 2]`, so a row is silently dropped: use batch = 1 only.** About 1.8 s per 4 s window on Colab CPU.
10. **Fine-tuning**: only ONNX is published. Re-creating trainable weights from ONNX is possible in principle but brittle; plan on training our own head on a frozen XLS-R instead.
11. **Training-set overlap**: Dhwani trained on IndicSynth, so IndicSynth results are a smoke test, not an unseen-generator benchmark.
