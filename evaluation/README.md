# Evaluation: generalisation of the third-party Dhwani checkpoint

This folder covers **evaluation only**. Nothing here trains, tunes or selects a model, a preprocessing recipe or a threshold.
The model is the **third-party** `ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model` `best_model.onnx`
(MIT, revision `ee0187142973e1f02050e47d2a31c55ee0263065`, sha256 `d1c232bf…7342`). Our team has not trained any weights.

| File | Purpose |
|---|---|
| `03_generalisation_eval_colab.ipynb` | Self-contained Colab notebook: pilot (default) and full evaluation |
| `build_03_notebook.py` | Generates the notebook (stdlib only). It embeds a verbatim snapshot of `src/vcd/dhwani_infer.py` plus `eval_lib.py`, and records their sha256 |
| `eval_lib.py` | Dataset pins, HTTP-range zip reader, protocol/metadata parsers, score-free selection, metrics (stdlib only) |
| `tests/test_eval_lib.py` | Local stdlib-only tests (no ML, no network, no datasets) |
| `PROGRESS.md` | What was done, what was verified, what still needs a Colab run |
| `runs/<run_name>/` | Put each downloaded Colab run zip here, unpacked (create it when the first run comes back) |

`src/vcd/dhwani_infer.py`, the notebooks in `notebooks/`, `mobile/` and `docs/` are read only from here and were not modified.
If `dhwani_infer.py` changes later, rebuilding the notebook picks up the new code and a new hash. The hash is recorded in each run's `run_config.json`.

## 1. Datasets: sources, licences, access, sizes, metadata

Checked on 2026-10-24 from this Mac using only landing pages, APIs, README/licence text and HTTP headers, plus 4-byte range probes.
**No audio and no archive contents were downloaded here.** The counts marked *literature* come from the dataset papers. The notebook recounts them from the protocol and metadata files and warns on any mismatch.

### ASVspoof 2019 Logical Access (LA): evaluation partition
- **Official source:** Edinburgh DataShare, <https://datashare.ed.ac.uk/handle/10283/3336>, DOI `10.7488/ds/2555`. No login needed.
- **Archive:** `LA.zip`, 7,640,952,520 B. DSpace checksum is MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`, which equals the HTTP ETag. Last-Modified 2019-06-04.
  Bitstream URL: `https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content`.
  The server answers byte-range requests (HTTP 206, checked).
- **Licence:** Open Data Commons Attribution (ODC-By 1.0), from the COPYING section of `README.txt`
  (README sha256 `a72c9527…5207`, `LICENSE_text.txt` sha256 `99a4d3e0…555d`). Citation: Todisco et al., Interspeech 2019, plus the DataShare record.
- **Labels and metadata:** CM protocol `LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt` has 5 columns:
  `SPEAKER_ID FILE_ID - SYSTEM_ID KEY`. `KEY` is `bonafide` or `spoof`, and `SYSTEM_ID` is the attack ID (A07–A19) for spoofs.
  Audio is in `LA/ASVspoof2019_LA_eval/flac/<FILE_ID>.flac`, derived from VCTK. The eval set has 48 speakers (21 male, 27 female; README); its speakers are disjoint from train/dev.
- **Size (literature):** 71,237 trials, 7,355 bona fide and 63,882 spoof.
- **Benchmark caveat:** A07–A19 are "unknown" with respect to the *ASVspoof* training partition. A16 and A19 reuse the training attacks A04 and A06 (per the ASVspoof 2019 database paper; not re-verified here).
  None of this establishes that the attacks are unseen *by Dhwani* (§2).

### In-the-Wild (Müller et al., "Does Audio Deepfake Detection Generalize?", Interspeech 2022)
- **Source:** Hugging Face dataset `mueller91/In-The-Wild`, pinned commit `eee168f92c367f8c82ff2cf42b6f61e362fd6211` (not gated).
  The HF dataset is published under the first author's account and the Fraunhofer AISEC project page <https://deepfake-total.com/in_the_wild> links to it.
- **Archive:** `release_in_the_wild.zip`, 8,161,489,023 B, LFS sha256 `46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a`
  (xet hash / CDN ETag `ec530d20…6f`). The CDN answers byte-range requests (HTTP 206, checked).
- **Licence: conflicting.** The HF card says **CC-BY-SA-4.0**. The project page says the dataset and its documentation are under **Apache-2.0**.
  The audio itself is public figures' speech taken from online videos. Use it for evaluation only, do not redistribute any audio, and cite the paper.
  Resolve the conflict with the authors before using it beyond research evaluation.
- **Labels and metadata:** `meta.csv` with columns `file, speaker, label` (`bona-fide` or `spoof`). The notebook parses these defensively and fails loudly on anything else.
  58 speakers. **No attack or generator IDs are published**, so per-attack results are impossible for this dataset.
- **Size:** 37.9 h total (20.7–20.8 h bona fide, 17.2 h spoof; HF card and project page). Literature: 31,779 clips, 19,963 bona fide and 11,816 spoof.

## 2. Overlap with Dhwani's training data: **unknown**

What Dhwani published:
- **Model card:** real audio comes from Mozilla Common Voice v24.0 (Indic) and `garystafford/deepfake-audio-detection` (English). Fake audio comes from `vdivyasharma/IndicSynth` (Indic) and `garystafford/deepfake-audio-detection` (English).
- **Training code** (GitHub `ayush2635/Dhwani-Deepfake-Audio-Detection-API` @ `efd512c`, `training/training_notebook.ipynb` cell 5) reads
  `/kaggle/input/indic-spoof-dataset-sample/`. That is an unpublished Kaggle folder whose contents and composition are not documented. The notebook has no saved outputs.
- `garystafford/deepfake-audio-detection` (CC-BY-4.0; current HF commit `fcf5344`, card v4) has 933 real clips from **14 unnamed public YouTube recordings**.
  Its 933 fakes come from Amazon Polly, ElevenLabs, Kokoro, Hume AI, Luvvoice and Speechify. The card does not say which dataset version Dhwani used.

Assessment:
- **ASVspoof 2019 LA:** none of the named sources is ASVspoof or VCTK. That is *absence of evidence*, not evidence of no overlap, because the actual training folder is unpublished.
  **Overlap: unknown.**
- **In-the-Wild:** none of the named sources is In-the-Wild. However, Dhwani's English real audio is YouTube speech from unnamed sources, and In-the-Wild bona fide audio is public-figure speech from video platforms.
  Speaker or recording overlap **cannot be ruled out**. **Overlap: unknown.**
- **Pretraining:** the XLS-R encoder was pretrained without labels on large public speech corpora, including YouTube-derived VoxLingua107. That is not label leakage, but it limits any claim that the audio is "unseen".
- Consequence: results here measure performance on these benchmarks. They are **not** proof of generalisation to generators or speakers unseen by Dhwani.

## 3. Pilot design (pre-registered in the notebook's settings cell)

- **Selection uses metadata only** (clip ID, speaker, label, attack ID), before any model call. The selection function refuses inputs with score-like fields.
  - Seed `20261025`. Records are sorted by clip ID first.
  - Each stratum has its own RNG, `random.Random("<seed>:<dataset>:<label>[:<attack>]:…")`. String seeds are SHA-512-hashed, so the sample is reproducible across machines and independent of input order.
  - Within each stratum, clips are taken by **speaker round-robin** over a shuffled speaker order, to avoid concentrating on a few speakers.
- **ASVspoof:** 8 spoof clips per attack × 13 attacks (A07–A19) = **104 synthetic**, plus **104 bona fide** (the 100 target was rounded up to equal strata).
- **In-the-Wild:** **100 bona fide + 100 spoof**. There are no attack strata.
- The manifests (`manifest_<dataset>.csv`) and `selection_<dataset>.json` are written before scoring. The JSON records the rule, seed, available and selected counts, shortfalls and the sha256 of the clip-ID list.
  On resume, the selection is recomputed and must match.
- **Model and preprocessing:**
  - `dhwani_infer.analyse(..., mode="train_peak")`, unchanged: decode, mono, 16 kHz (soxr_hq), then validation.
  - Clips under 2.0 s are **rejected and not replaced**. Clips over 10.0 s are cut to the first 10 s and flagged `truncated`.
  - Silent or mostly-silent clips are rejected. Audio is peak-normalised to −3 dBFS.
  - Clips of 4 s or less get one zero-padded window. Longer clips get 4 s windows with a 2 s hop, the last window aligned to the end.
  - Each window runs at **batch size 1**. The clip score is the mean of the window synthetic-class softmax scores.
- **Threshold:** a clip is synthetic if its score is ≥ 0.5 (provisional and untuned). Nothing is optimised on these clips, and they must never be used for training or tuning.
- **Failures:** fetch failures, decode failures and policy rejections stay in `predictions.csv` with their reasons. They are reported as **coverage**, separately from accuracy metrics, which use scored clips only.
  Container duration is recorded even for rejected clips.

### Reported metrics (per dataset; ASVspoof also per attack)
- **Threshold-free:**
  - ROC-AUC (Mann–Whitney, ties = ½).
  - EER using the ASVspoof reference convention (the threshold minimising |FNR − FPR|, reporting their mean). The EER threshold is descriptive only and is not an operating point.
  - Both are cross-checked against scikit-learn in the notebook.
- **At 0.5:** confusion matrix, balanced accuracy, synthetic recall and genuine false-positive rate, with Wilson 95 % intervals for the proportions.
- **Uncertainty:** stratified bootstrap 95 % CIs for AUC, EER and balanced accuracy. These ignore speaker clustering, so they are optimistic.
  With about 100 clips per class, and 8 per ASVspoof attack, intervals are wide and per-attack numbers are indicative only.
- **Per attack (ASVspoof):** each attack's scored spoofs against *all* scored pilot bona fide clips.
- **Also reported:** coverage and rejection reasons by dataset × label (and by attack), duration statistics, runtime (per clip, per window, total, plus a projection for the full partition), and every per-file prediction with window scores and logits.

## 4. Colab steps

1. Open <https://colab.research.google.com>, choose *File → Upload notebook*, and pick `evaluation/03_generalisation_eval_colab.ipynb`. A CPU runtime is fine for the pilot.
2. Optional: set `USE_DRIVE = True` in cell 1 to reuse a Drive-cached checkpoint at `/content/drive/MyDrive/vcd_cache/best_model.onnx` and keep the run folder in Drive.
   Otherwise the checkpoint is downloaded once (1.26 GB) into the runtime's HF cache. Its sha256 is verified in either case.
3. Do **not** change any other setting for the pilot. Choose *Runtime → Run all*. No HF token or login is needed for either dataset.
4. Expected behaviour:
   - Step 7 opens both archives remotely and reads only the central directory and the label files. It prints counts against the literature values and any **BLOCKER**.
   - Step 9 transfers only the selected clips, about 410, to tens of MB.
   - Expected time is roughly 15–30 min, mostly model time (about 1.8 s per 4 s window on Colab CPU in the baseline run).
5. Step 13 downloads `<run_name>.zip`, which contains no audio. Also download the executed notebook (*File → Download → .ipynb*).
6. Put both in `evaluation/runs/<run_name>/` and paste the step 7, 10 and 11 outputs into chat.

**If range access is blocked** (step 7 prints `RangeUnsupported` or HTTP errors), set `ACCESS = "download"` and rerun.
This downloads the whole archives (7.6 GB + 8.2 GB) to local `/content` and verifies the full MD5 / sha256. Report the blocker text either way.

## 5. Path to full evaluation (after the pilot)

Run the same notebook with `RUN_KIND = "full"`, `ACCESS = "download"`, `KEEP_AUDIO = False`, `N_BOOT = 200` and `USE_DRIVE = True`.
- Use `PROVIDER = "cuda"` on a GPU runtime. The first 20 clips are scored on CPU too, and the run aborts if any clip score differs by more than 0.002.
- If a session dies, rerun with `RUN_NAME` set to the same folder. `predictions.jsonl` is appended per clip and the selection is re-verified.
- Full scope is every eval protocol row: 71,237 for ASVspoof and 31,779 for In-the-Wild (literature counts).
  On CPU that is tens of hours. Step 12 prints a projection from the pilot timings.

Report the full run as a separate result and never merge it with the pilot. Any change to the preprocessing recipe, aggregation or threshold must be decided on separate validation data, and recorded in `docs/PROGRESS.md` before running.

## 6. Failed baseline, audit, and the candidate detector (2026-10-24)

- Run `runs/20261024T123319Z_pilot_train_peak/` is the **failed Dhwani baseline** (In-the-Wild pilot: AUC 0.456, EER 0.561, accuracy 0.503 at 0.5). It is frozen.
- `dhwani_implementation_audit.md`: label mapping, decoding, normalisation and aggregation checked line by line against upstream. **No bug found**; nothing was changed.
- `dev_evidence/`: the 200 pilot clip IDs. Burned for tuning; every later held-out subset excludes them.

### Shortlist (research only; sources in `candidate_shortlist.md`)
| Candidate | ITW EER | Weights / licence | Size | Android path | Verdict |
|---|---|---|---|---|---|
| **AntiDeepfake Wav2Vec2-Small** (NII) | 4.24 % self-reported, 5.31 % independent | HF safetensors 380 MB, **CC BY-NC-SA 4.0** | 95 M | wav2vec2-base: ONNX export supported by Optimum, int8 wav2vec2 Android demos exist | **chosen** |
| DF Arena 500M (Idiap/TalTech) | 1.76 % / 1.87 % independent | HF 1.75 GB, custom code, **non-commercial** | 436 M | Conformer on XLS-R, unverified export, multi-second latency | too heavy |
| Nes2Net-X (XLS-R) | 5.15 % self / 7.75 % independent | Google Drive ckpt, Apache-2.0 end to end | 318 M | plausible ONNX, Dhwani-sized | fallback if a permissive licence is required |

Why AntiDeepfake-Small: the only candidate with strong, independently reproduced In-the-Wild evidence at a phone-sized footprint, hosted on HF with a pinned revision and sha256. Costs: non-commercial licence (fine for the hackathon, not for a shipped product without permission), and its published loader needs fairseq, which no longer installs on Colab; `antideepfake_hf.py` maps the weights into `transformers` with a strict all-tensors-accounted check.
Known overlap: In-the-Wild is **not** in its training table; ASVspoof 2019 LA **is** (so 19LA cannot be its test set); VoxCeleb2 celebrity speech is, so speaker overlap with In-the-Wild public figures is possible.

### Notebook 04: `04_candidate_comparison_colab.ipynb` (built by `build_04_notebook.py`)
1. Colab, **GPU (T4)** runtime. Upload the notebook, Run all. No tokens needed.
2. Downloads: Dhwani 1.26 GB (or Drive cache), AntiDeepfake 380 MB; both sha256-verified. In-the-Wild clips via HTTP range (≈ 500 clips).
3. Cell 9 scores the candidate on the 200 pilot clips (development evidence). AUC < 0.80 means **stop and investigate**; it does not by itself prove the mapping is wrong, and a high AUC does not prove parity. Cell 13 is the parity check: a separate Python 3.8 venv with fairseq 0.12.2 scores identical inputs with the authors' code; the result is recorded as `verified` (max |Δ fake score|) or `unverified` (reason). Every candidate number carries that status.
4. Cell 10–11: fresh held-out subset, seed `20261026`, 150 + 150 clips, clip-disjoint from the pilot (speaker-disjoint impossible). Both models on the same clips; threshold-free metrics, fixed threshold 0.5 for both, paired bootstrap. The card's "EER @ 0.8128" is **not** applied: it is the EER point the authors found on In-the-Wild itself (our test data) with a whole-file pipeline we have not proven identical; it is shown as context only.
5. Cell 12: ONNX export + ORT parity + Colab CPU latency for the candidate (deployment evidence only; no mobile files touched).
6. Return the zip and executed notebook into `runs/<run_name>/`.

### If fine-tuning turns out necessary (proposal, not started)
Bounded T4 experiment on the candidate's head (and optionally the top encoder layers), only if the held-out result is insufficient:
- Data: MLAAD (gated, CC-BY-NC) + IndicSynth fakes and Common Voice / Kathbath / LibriTTS genuine, **both classes** receiving the same augmentations (noise, 8 kHz band-limit, codec/MP3, gain). No In-the-Wild data in training.
- Splits grouped by speaker and by generator: train / validation (model + threshold selection) / test, with **≥ 3 generators held out of train and validation entirely**.
- Budget: ≤ 2 h T4, frozen encoder first; log every run; report test only once.
- Final check on a fresh In-the-Wild selection excluding the pilot and the 04 held-out IDs.

## 7. EER-first protocol and robustness evaluation (2026-10-24)

**Metric policy (from now on)**
- Primary metric: **EER** computed on continuous synthetic scores; ROC-AUC alongside. Never on thresholded predictions.
- Checkpoint selection (when we train) uses **validation EER** only. The **deployment threshold** is chosen on validation data only.
- Test EER and its crossing threshold are reported separately, as descriptive numbers; the test threshold is never used for deployed classification.
- Accuracy, precision, recall, F1 and FPR are reported at the frozen validation-selected threshold. **None exists yet → 0.5 is retained.**
- Already-inspected subsets (200 pilot clips, 300 held-out clips of run 20261024T140853Z) are development evidence: no tuning on them.

**Notebook 05: `05_robustness_eer_colab.ipynb`** (built by `build_05_notebook.py`; same Colab runtime as 04 preferred)
- Clips: the 300 held-out clips (244 eligible), decoded once; every condition is applied to the same decoded clip (paired). Degraded audio passes the same validation gate; rejections/failures and coverage per condition are reported, never dropped silently.
- Required conditions: clean 16 kHz; 16→8→16 kHz (soxr_hq); **Opus 16 kb/s VBR voip (libopus)**; **MP3 32 kb/s CBR (libmp3lame)**; white Gaussian noise at 10 dB SNR, seed `sha256("20261027:<clip_id>")`.
- Official ratio: worst degraded EER / clean EER on the paired set; PASS ≤ 2.5; INCOMPLETE if a required condition is missing (e.g. ffmpeg lacks an encoder); clean EER = 0 → UNDEFINED_RATIO with absolute EERs. Also EER(8 kHz) − EER(clean) in percentage points.
- Sensitivity (separate from the ratio): −20 dB gain, peak −1 dBFS, 1 s silence pad, first 3 s only, telephone band-pass + 8-bit μ-law. Reported as paired EER deltas, median |Δscore| and decision flips at 0.5.
- Unseen generators: **not establishable** with In-the-Wild (no generator IDs) or MLAAD (in the candidate's training set). Section 9 scores user-supplied clips of a named generator with a provenance note, reported separately.
- Dhwani re-run under conditions is optional (`RUN_DHWANI`), off by default.

**Colab steps:** after notebook 04 (and `12b` ONNX retry) in the same runtime, upload `05_robustness_eer_colab.ipynb`, Run all (≈ 10–15 min on T4), return the zip and paste cells 3, 7, 8. If `ffmpeg` reports `libopus`/`libmp3lame` missing, the verdict is INCOMPLETE by design; tell me and I will switch to encoders that are present.

## 8. Noise-robustness improvement: denoiser front-end experiment first (2026-10-24)

Run 20261024T151905Z failed the official ratio on the noise condition only (15.98 % vs 2.46 % clean → 6.50). Noise stays a requirement; the
checkpoint and the 0.5 threshold are untouched. The bounded improvement path is two steps; **only step 1 is built**.

**Notebook 06: `06_denoiser_frontend_colab.ipynb`** (`build_06_notebook.py`; same runtime as 04/05 preferred; ≈ 25–35 min)
- Front-ends (`frontend.py`, fixed settings): `nr_stationary` (spectral gating, pure DSP, Android-portable) and `deepfilternet3` (2 M params, optional install).
  The same front-end is applied to *all* inputs: genuine and synthetic, clean and degraded.
- Stage A (development evidence = the 244 inspected held-out clips): all front-ends × 5 required conditions, paired; artefact-removal check on clean audio
  (clean EER and clean synthetic-score shifts); **pre-registered freeze rule** (`eval_lib.frontend_selection`): eligible only if clean EER ≤ baseline + `CLEAN_MARGIN_PP`,
  then lowest worst-degraded EER, ties → simplest, frozen only if it beats the baseline's worst EER. If nothing qualifies, the experiment is declared failed.
- Stage B (fresh): seed `20261028`, 150 + 150 clips excluding all **500** inspected IDs; baseline vs frozen front-end; absolute EERs, coverage, **unrounded** worst/clean
  ratio, PASS/FAIL/INCOMPLETE, plus the clean-margin check so a higher clean EER cannot disguise a better ratio.
- **Margins:** the supplied requirement says Phase-1 EERs must "remain within 2 points". We interpret this **provisionally as 2 percentage points** (`OFFICIAL_MARGIN_PP = 2.0`, pending organiser clarification) for the Stage B clean-margin check. Stage A uses a stricter **development selection margin** of 1 pp (`DEV_SELECTION_MARGIN_PP`) to pick a front-end; that is our own choice and is never presented as the official requirement.
- Limitations reported with every number: speakers shared with inspected sets; VoxCeleb2 overlap possible; In-the-Wild not in the candidate's training table.

**Step 2 (only if 06 fails): notebook 07, noise-augmented fine-tuning — not built, not started.** Start from the verified candidate weights; noise + codec
augmentation on both classes; separate train/validation with grouped splits; checkpoint selection by validation EER (cross-entropy retained); deployment threshold
from validation only; final evaluation on a new fresh selection (excluding all inspected IDs) under the same five conditions and clean-margin rule. Budget ≤ 2 h T4.
