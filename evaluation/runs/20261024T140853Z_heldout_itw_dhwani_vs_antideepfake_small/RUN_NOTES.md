# Run 20261024T140853Z — held-out In-the-Wild: Dhwani (failed baseline) vs AntiDeepfake Wav2Vec2-Small

Source zip sha256 `6b6abc46…247b`, unpacked unchanged. No audio inside. Executed `.ipynb` not yet returned.

> **Scope:** held-out *pilot subset* of In-the-Wild (300 selected, 244 scored). Clip-disjoint from the 200 development clips (verified: 0 overlap);
> **speakers are shared** (all 54; speaker-disjoint is impossible on this dataset). Not a full-dataset benchmark. Generalisation to new speakers is untested.
> Threshold 0.5 was fixed before scoring; nothing was tuned on these clips, and they are now used: any later decision needs another fresh selection.

## Setup
| | |
|---|---|
| Runtime | Colab T4; Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, onnxruntime 1.31.0 |
| Dhwani | `ee01871`, sha256 verified, `train_peak`, mean of 4 s windows, CPU ORT, batch 1 |
| Candidate | `nii-yamagishilab/wav2vec-small-anti-deepfake` @ `9a13264`, sha256 `828ee456…` verified; fairseq→transformers mapping (210 tensors, 8 pre-training-only dropped, architecture fields = fairseq defaults); CUDA |
| **Conversion parity** | **verified**: fairseq 0.12.2 reference (authors' model code, Python 3.8 venv, CPU) vs our transformers path on 12 identical held-out inputs: max \|Δ logit\| 1.1e-5, max \|Δ fake score\| 4.6e-7 |
| Dev-evidence gate | candidate on the 200 pilot clips (155 scored): AUC 0.998, EER 0.026 → passed (stop-and-investigate gate only) |
| Selection | seed 20261026, 150 + 150, speaker round-robin, excluded 200 pilot IDs; clip-ID list sha256 `cd83219f…` (recomputed locally: matches) |

## Coverage (separate from accuracy)
| Label | Selected | Rejected (< 2 s) | Scored |
|---|---|---|---|
| genuine | 150 | 30 | 120 |
| synthetic | 150 | 26 | 124 |

All 56 exclusions are the 2.0 s minimum; none replaced; no fetch or decode failures. All files 16 kHz mono.

## Results on the 244 scored clips (recomputed locally from `predictions_heldout.jsonl`; identical to `metrics.json`)

| Model / score | ROC-AUC [95 % boot] | EER [95 % boot] | @ 0.5: TP / FN / FP / TN | Bal. acc | Synthetic recall | Genuine FPR |
|---|---|---|---|---|---|---|
| Dhwani `train_peak` (baseline) | **0.512** [0.442, 0.588] | 0.488 [0.422, 0.553] | 33 / 91 / 25 / 95 | 0.529 | 0.266 | 0.208 |
| **AntiDeepfake-Small, whole clip (primary)** | **0.996** [0.990, 0.999] | **0.025** [0.008, 0.061] | 119 / 5 / 3 / 117 | **0.967** | 0.960 | 0.025 |
| AntiDeepfake-Small, mean of 4 s windows (secondary) | 0.994 [0.988, 0.999] | 0.033 [0.008, 0.061] | 118 / 6 / 4 / 116 | 0.959 | 0.952 | 0.033 |

Paired bootstrap on the same clips: AUC(candidate) − AUC(Dhwani) = **+0.48** [0.41, 0.56].
CIs resample clips only and ignore speaker clustering (all 54 speakers appear in both classes), so they are optimistic.

Score distribution (candidate, fake score): genuine median 0.0001, 95th pct 0.06; synthetic median 1.0, 5th pct 0.69. Scores are confident but, unlike Dhwani's, on the correct side.

The 8 candidate errors at 0.5: 5 spoofs scored low (Nick Offerman 6.4 s 0.005; Donald Trump 2.1 s 0.03; Robert Kardashian 2.2 s 0.04; Notorious B.I.G. 2.3 s 0.18; Jeff Goldblum 10.3 s→10 s 0.21) and 3 genuine scored high (Notorious B.I.G. 2.7 s 0.68; FDR 2.6 s 0.69; 2Pac 4.5 s 0.99). Three of the five misses are 2.1–2.3 s clips, and the false positives include archival recordings (FDR, 2Pac, B.I.G.): short clips and old/lossy channels are the visible weak spots. n is tiny; indicative only.

Dhwani is at chance again on fresh clips (AUC 0.512 vs 0.456 on the pilot), confirming the failed baseline.

## Runtime
Candidate on T4: median 140 ms per clip including the secondary window pass. Dhwani on 2-vCPU: median 3.6 s per clip.
ONNX export (cell 12) **failed**: `No module named 'onnxscript'` (torch 2.11's default exporter). Retry cell: `evaluation/colab_cells/12b_onnx_export_retry.py` (same runtime). No ONNX size or latency numbers exist yet.

## Caveats carried with these numbers
- Candidate licence **CC BY-NC-SA 4.0** (research use). Not cleared for a commercial app.
- In-the-Wild is not in the candidate's training table, but VoxCeleb2 celebrity speech is: speaker/recording overlap with In-the-Wild public figures is possible and was not checked.
- Pipeline differences from the authors: our 2–10 s policy (truncates > 10 s, rejects < 2 s) vs whole files; parity was verified on our arrays, not on their decoding.
- Dhwani training-data overlap with In-the-Wild remains unknown.
