# Run 20261024T123319Z — pilot, train_peak, In-the-Wild only

Source zip: `20261024T123319Z_pilot_train_peak.zip` (sha256 `05e66490…6175`), unpacked here unchanged. It contains no audio.
The executed notebook has not been returned yet.

> **PILOT SUBSET ONLY: this is not a full-dataset benchmark result.**
> **ASVspoof 2019 LA was NOT TESTED in this run.** Its access failed with `HTTPError 403 Forbidden` on the very first 1-byte Range request from Colab, so no protocol was read and no clip was selected.
> Training-data overlap between Dhwani and In-the-Wild is **unknown**.

## Setup (from run_config.json)

| Item | Value |
|---|---|
| Model | Dhwani `ee01871`, sha256 `d1c232bf…7342`. Downloaded in this runtime and verified |
| Code | `dhwani_infer.py` `8a7c9c8f…`, `eval_lib.py` `92238544…`. Both match the build |
| Runtime | ORT 1.31.0, CPUExecutionProvider, 2 vCPU. Python 3.13.16, librosa 0.11.0, sklearn 1.6.1 |
| Interface probe | batch-1 runs at 32k, 48k, 64k and 96k samples. **batch2 not ok** (returns [1, 2]). Output is deterministic |
| Preprocessing | `train_peak`, batch size 1, clip score = mean of window scores, threshold 0.5 (provisional) |
| Selection | seed 20261025, speaker round-robin, metadata only; 100 genuine + 100 spoof; clip-ID list sha256 `90a285f8…9b20` (recomputed locally from the manifest: matches) |

## In-the-Wild access and metadata
- HF `mueller91/In-The-Wild` @ `eee168f`, via HTTP range. Host us.gcp.cdn.hf.co, size 8,161,489,023 B, ETag equal to the recorded xet hash.
  The whole-archive sha256 was **not** computed (range mode). Each member's CRC-32 was checked on read.
- `meta.csv` has 31,779 rows: 19,963 bona fide and 11,816 spoof. This **matches** the literature counts, and every row has audio.
- It has **54 distinct speaker strings, not the 58 speakers stated by the authors**. The difference is unexplained, possibly merged or renamed names. Noted, not corrected.

## Coverage (reported separately from accuracy)

| Label | Selected | Fetch failed | Rejected | Scored | Coverage | Truncated to 10 s | Single padded window |
|---|---|---|---|---|---|---|---|
| genuine | 100 | 0 | 25 | 75 | 0.75 | 9 | 36 |
| synthetic | 100 | 0 | 20 | 80 | 0.80 | 16 | 27 |

All 45 rejections are `too_short` (container durations 0.64–1.98 s, under the 2.0 s policy). None were replaced; they are listed in `excluded_clips.csv`.

## Metrics on the 155 scored clips (75 genuine, 80 synthetic)

I recomputed these locally from `predictions.jsonl` with `eval_lib` and they match `metrics.json` exactly. The notebook's sklearn cross-check also matches.

**Threshold-free**
- ROC-AUC **0.456** (stratified bootstrap 95 % 0.368–0.544)
- EER **0.561** (bootstrap 0.484–0.645). The EER threshold of 0.0009 is descriptive only.

**At the provisional threshold 0.5**

|  | pred synthetic | pred genuine |
|---|---|---|
| synthetic | 21 | 59 |
| genuine | 18 | 57 |

- Balanced accuracy **0.511** (bootstrap 0.446–0.582)
- Synthetic recall **0.263** (Wilson 0.179–0.368)
- Genuine FPR **0.240** (Wilson 0.158–0.348)

The bootstrap CIs ignore speaker clustering (53 speakers scored), so they are optimistic.

**Score distribution**
- Scores are saturated and bimodal in both classes.
  - Genuine: 48/75 below 0.01, 6/75 above 0.99.
  - Synthetic: 51/80 below 0.01, 9/80 above 0.99.
- Most synthetic clips get a confident "genuine" score.

**Per attack:** not available, because In-the-Wild publishes no generator or attack IDs.

## Runtime
- In-the-Wild, 200 clips: 15.3 min in total.
  - Model time per window: median 2.02 s.
  - Model time per scored clip: median 4.03 s.
  - Range fetch per clip: median 0.76 s.
- Rough projection for all 31,779 clips on the same CPU: ≈ 50 h.

## Interpretation (bounded)
- On this pilot subset of In-the-Wild, the checkpoint under the pre-registered recipe performs **at chance**: the AUC CI includes 0.5 and the point estimate is below it. The two-clip smoke-test success does not carry over.
- This run cannot tell whether the cause is domain shift, the unresolved preprocessing question, or the label-origin confounds listed in `docs/UPSTREAM_AUDIT.md`.
  Per the plan, `train_peak` and 0.5 are **not** changed in response to this result. Any recipe comparison must happen on separate validation data, never on these clips.
- These 200 clips are now *used*: keep them out of training, threshold tuning and any model or preprocessing selection.
