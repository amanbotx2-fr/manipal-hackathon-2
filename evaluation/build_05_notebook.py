"""Build evaluation/05_robustness_eer_colab.ipynb.

Paired robustness evaluation of the candidate (AntiDeepfake Wav2Vec2-Small) with EER as the
primary metric, on the eligible held-out In-the-Wild clips of run 20261024T140853Z. Designed to
run in the SAME Colab runtime as notebook 04 (reuses the loaded model and cached audio when
present; otherwise re-downloads). Embeds dhwani_infer.py (read only), eval_lib.py,
antideepfake_hf.py, degrade.py and the held-out manifest. Standard library only.
Run: python3 evaluation/build_05_notebook.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN04 = HERE / "runs/20261024T140853Z_heldout_itw_dhwani_vs_antideepfake_small"
SRC = {
    "dhwani_infer.py": (ROOT / "src/vcd/dhwani_infer.py").read_text(),
    "eval_lib.py": (HERE / "eval_lib.py").read_text(),
    "antideepfake_hf.py": (HERE / "antideepfake_hf.py").read_text(),
    "degrade.py": (HERE / "degrade.py").read_text(),
}
MANIFEST = (RUN04 / "manifest_heldout_in_the_wild.csv").read_text()
SHA = {k: hashlib.sha256((v.rstrip() + "\n").encode()).hexdigest() for k, v in SRC.items()}
OUT = HERE / "05_robustness_eer_colab.ipynb"
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip().splitlines(keepends=True)})


md("""
# 05 — Robustness evaluation, EER-first (candidate AntiDeepfake Wav2Vec2-Small)

**Clips:** the 300 held-out In-the-Wild clips of run 20261024T140853Z (manifest embedded; 244 were eligible under the 2–10 s policy).
They are clip-disjoint from the 200 development clips, but their clean scores have already been inspected, so this notebook makes
**no decision** on them: the threshold stays at the frozen **0.5** (no validation-selected threshold exists yet), and nothing is tuned.

**Protocol**
- Primary metric: **EER** on continuous synthetic scores (never on thresholded predictions). ROC-AUC alongside.
- The EER crossing threshold is reported as **descriptive only** and is never used for classification.
- Accuracy, precision, recall, F1 and FPR are reported at the frozen threshold 0.5.
- Conditions are **paired**: the same decoded clip is degraded and re-scored. Every condition records its own rejections/failures and coverage;
  nothing is removed silently. Metrics are given (a) per condition on everything it scored and (b) on the **paired set** of clips scored in all
  required conditions, which is what the official ratio uses.
- **Official ratio** = worst degraded EER / clean EER over the required conditions {narrowband 16→8→16 kHz, Opus 16 kb/s, MP3 32 kb/s, white noise 10 dB SNR};
  **PASS if ≤ 2.5**; INCOMPLETE if any required condition is missing; clean EER = 0 → ratio UNDEFINED (absolute EERs reported). Also: EER(8 kHz) − EER(clean) in percentage points.
- **Sensitivity checks** (separate from the official ratio): −20 dB gain, peak −1 dBFS, 1 s silence padding, first 3 s only, telephone band-pass + 8-bit μ-law.
- **Unseen generators:** In-the-Wild publishes no generator IDs, and MLAAD (which has them) is in the candidate's training set, so a generator-disjoint
  evaluation **cannot be established** with the data we have. An optional section scores user-supplied clips from a *named* generator with a documented
  release date after the candidate's training cut-off; it is reported separately and labelled by the evidence given.
- Dhwani (failed baseline) is not re-run here by default (`RUN_DHWANI = False`; ≈ 75 min CPU if enabled).

Runs in the **same runtime as notebook 04** (model + cached audio reused) or standalone on a T4 (re-downloads). Estimated 10–15 min.
""")

code("""
# 1. Settings (fixed before any degraded score is seen)
FROZEN_THRESHOLD = 0.5          # no validation-selected threshold exists yet; 0.5 retained (never the test EER threshold)
RUN_DHWANI = False              # True: also score the failed baseline under every condition (CPU, slow)
REQUIRED = ["clean", "narrowband_8k", "codec_1", "codec_2", "noise"]
SENSITIVITY = ["loud_minus20db", "loud_peak_minus1db", "silence_pad_1s", "duration_first3s", "telephone_quality"]
N_BOOT = 500
UNSEEN_DIR = "/content/unseen_generators"   # optional: <generator_name>/{genuine,synthetic}/*.wav + PROVENANCE.md (see section 9)
USE_DRIVE = False
DRIVE_DIR = "/content/drive/MyDrive/vcd_cache"
WORK = "/content/vcd_eval"
""")

code("""
# 2. Dependencies (no-op in the notebook-04 runtime)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxruntime", "librosa", "soundfile", "huggingface_hub",
                "transformers", "safetensors", "scipy"], check=True)
""")

for name, text in SRC.items():
    code(f"%%writefile {name}\n" + text)
code("%%writefile manifest_heldout_in_the_wild.csv\n" + MANIFEST)

code(f"""
# 3. Imports, provenance, environment, ffmpeg capabilities
import os, json, time, hashlib, platform, datetime, importlib, traceback, shutil, csv
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import dhwani_infer as dh, eval_lib as el, antideepfake_hf as ad, degrade as dg
for m in (dh, el, ad, dg): importlib.reload(m)
EXPECTED = {json.dumps(SHA, indent=4)}
CODE_SHA256 = {{f: hashlib.sha256((open(f, encoding="utf-8").read().rstrip() + "\\n").encode()).hexdigest() for f in EXPECTED}}
assert CODE_SHA256 == EXPECTED, CODE_SHA256
assert FROZEN_THRESHOLD == 0.5 and dh.DECISION_THRESHOLD == 0.5
MAN = list(csv.DictReader(open("manifest_heldout_in_the_wild.csv")))
assert len(MAN) == 300
FFMPEG = dg.ffmpeg_available(); print("ffmpeg:", FFMPEG)
import transformers, librosa, soundfile, scipy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
           librosa=librosa.__version__, soundfile=soundfile.__version__, scipy=scipy.__version__, device=DEVICE,
           gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, ffmpeg=FFMPEG, code_sha256=CODE_SHA256)
print(json.dumps(ENV, indent=1))
""")

code("""
# 4. Candidate model (reuse the runtime's object from notebook 04 if present; else load from cache/download, sha256-verified)
from huggingface_hub import hf_hub_download
if "cand" in globals() and getattr(cand, "report", None):
    print("reusing loaded candidate from this runtime")
else:
    if USE_DRIVE:
        from google.colab import drive; drive.mount("/content/drive")
    try: AD_PATH = hf_hub_download(ad.MODEL_REPO, ad.MODEL_FILE, revision=ad.MODEL_REVISION, local_files_only=True)
    except Exception: AD_PATH = hf_hub_download(ad.MODEL_REPO, ad.MODEL_FILE, revision=ad.MODEL_REVISION)
    assert os.path.getsize(AD_PATH) == ad.MODEL_BYTES and dh.sha256_file(AD_PATH) == ad.MODEL_SHA256
    AD_CFG = hf_hub_download(ad.MODEL_REPO, "config.json", revision=ad.MODEL_REVISION)
    cand = ad.AntiDeepfakeHF.load(AD_PATH, os.path.dirname(AD_CFG), device=DEVICE)
print({k: cand.report[k] for k in ("n_encoder_mapped", "n_head", "n_params_encoder")})
sess = None
if RUN_DHWANI:
    try: DH_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_files_only=True)
    except Exception: DH_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION)
    sess = dh.load_session(DH_PATH, verify_sha=True)
""")

code("""
# 5. Audio: reuse /content/vcd_eval/audio/heldout from notebook 04; fetch any missing clip via HTTP range
AUD = os.path.join(WORK, "audio", "heldout"); os.makedirs(AUD, exist_ok=True)
missing = [m for m in MAN if not os.path.exists(os.path.join(AUD, os.path.basename(m["archive_member"])))]
print(len(MAN) - len(missing), "clips cached,", len(missing), "to fetch")
if missing:
    src = el.SOURCES["in_the_wild"]; zf, raw = el.open_remote_zip(src["url"], src["bytes"])
    for m in missing:
        with open(os.path.join(AUD, os.path.basename(m["archive_member"])), "wb") as f: f.write(zf.read(m["archive_member"]))
BASE = os.path.join(DRIVE_DIR if USE_DRIVE else WORK, "runs")
utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_NAME = f"{utc}_robustness_eer_itw_heldout"; RUN_DIR = os.path.join(BASE, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
el.jdump({"settings": dict(FROZEN_THRESHOLD=FROZEN_THRESHOLD, RUN_DHWANI=RUN_DHWANI, REQUIRED=REQUIRED, SENSITIVITY=SENSITIVITY,
          N_BOOT=N_BOOT, noise_seed=dg.NOISE_SEED, conditions=dg.CONDITIONS, source_run="20261024T140853Z_heldout_itw_dhwani_vs_antideepfake_small",
          candidate=dict(repo=ad.MODEL_REPO, revision=ad.MODEL_REVISION, sha256=ad.MODEL_SHA256)), "env": ENV, "started_utc": utc},
         os.path.join(RUN_DIR, "run_config.json"))
print(RUN_DIR)
""")

code("""
# 6. Decode + validate each clip once (clean eligibility), then score every condition on the SAME decoded clip (paired)
CONDS = REQUIRED + SENSITIVITY
PRED_PATH = os.path.join(RUN_DIR, "predictions_conditions.jsonl")
done = set()
if os.path.exists(PRED_PATH):
    done = {(r["clip_id"], r["condition"]) for r in map(json.loads, open(PRED_PATH)) if r}
t_start = time.time()
with open(PRED_PATH, "a", encoding="utf-8") as out:
    for i, m in enumerate(MAN):
        base = {"clip_id": m["clip_id"], "label": m["label"], "speaker_id": m["speaker_id"]}
        local = os.path.join(AUD, os.path.basename(m["archive_member"]))
        try:
            y, sr = dh.load_audio(local); x16 = dh.to_mono_16k(y, sr); x16, info = dh.validate(x16); clean_ok = True
        except dh.AudioRejected as e:
            clean_ok, info = False, {"reject_reason": str(e)}
        for cond in CONDS:
            if (m["clip_id"], cond) in done: continue
            row = dict(base, condition=cond, clean_eligible=clean_ok, **{k: info.get(k) for k in ("duration_s", "truncated")})
            if not clean_ok:
                row.update(status="rejected_clean", reject_reason=info["reject_reason"]); out.write(json.dumps(row) + "\\n"); continue
            t0 = time.perf_counter()
            try:
                xd = dg.apply(cond, x16, clip_id=m["clip_id"])
                row["degrade_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                if cond == "duration_first3s": row["unchanged_shorter_than_3s"] = bool(len(x16) <= 3 * dh.SR)
                # degraded audio goes through the SAME validation gate as production input (silence/too-short rules)
                xv, vinfo = dh.validate(xd)
                row.update(deg_peak_dbfs=vinfo["peak_dbfs"], deg_rms_dbfs=vinfo["rms_dbfs"], deg_duration_s=vinfo["duration_s"])
                c = ad.score_clip(cand, xv)
                row.update(status="ok", cand_fake_score=c["fake_score_whole"], cand_logits=json.dumps(c["logits_fake_real"]), cand_ms=c["latency_ms_model"])
                if sess is not None:
                    r = dh.analyse_waveform(sess, xv, dh.SR, mode="train_peak", label=m["clip_id"])
                    row.update(dhwani_score=r.clip_score_mean, dhwani_status=r.status)
            except dh.AudioRejected as e:
                row.update(status="rejected_degraded", reject_reason=str(e), reject_category=el.reject_category(str(e)))
            except Exception as e:
                row.update(status="failed", reject_reason=f"{type(e).__name__}: {str(e)[:300]}")
            out.write(json.dumps(row, default=str) + "\\n"); out.flush()
        if (i + 1) % 25 == 0 or i + 1 == len(MAN):
            e = time.time() - t_start; print(f"{i+1}/{len(MAN)} {e/60:.1f} min, ETA {e/(i+1)*(len(MAN)-i-1)/60:.1f} min")
P = pd.DataFrame([json.loads(l) for l in open(PRED_PATH) if l.strip()]).drop_duplicates(["clip_id", "condition"], keep="last")
""")

code("""
# 7. Coverage and failures per condition (nothing hidden), then EER-first metrics: per condition and on the paired set
cov = P.groupby("condition").status.value_counts().unstack(fill_value=0).reindex(CONDS)
cov["selected"] = cov.sum(axis=1); cov["coverage"] = (cov.get("ok", 0) / cov["selected"]).round(4)
print("Coverage by condition:\\n", cov.to_string())
fails = P[~P.status.isin(["ok", "rejected_clean"])]
print("\\nDegraded-stage rejections/failures:", "none" if fails.empty else "")
if not fails.empty: print(fails[["condition", "clip_id", "label", "status", "reject_reason"]].to_string(index=False))

OK = P[P.status == "ok"]
paired_ids = set.intersection(*[set(OK[OK.condition == c].clip_id) for c in REQUIRED])
print(f"\\nPaired set (scored in ALL required conditions): {len(paired_ids)} clips")

def metrics(df, thr=FROZEN_THRESHOLD, n_boot=N_BOOT):
    y = (df.label == "synthetic").astype(int).tolist(); s = df.cand_fake_score.astype(float).tolist()
    if len(set(y)) < 2: return None
    m = el.evaluate(y, s, thr, n_boot=n_boot, seed=20261027)
    a = m["at_threshold"]; m["at_frozen_threshold"] = a; m.pop("at_threshold")
    return m

RES = {}
for scope_name, ids in (("all_scored", None), ("paired", paired_ids)):
    RES[scope_name] = {}
    for c in CONDS:
        d = OK[OK.condition == c]
        if ids is not None: d = d[d.clip_id.isin(ids)]
        RES[scope_name][c] = metrics(d)

def row(c, m):
    if m is None: return f"{c:20s} n/a"
    tf, a, bs = m["threshold_free"], m["at_frozen_threshold"], m["bootstrap95"]
    return (f"{c:20s} n={m['n_scored']:3d}  EER {tf['eer']*100:5.2f}% [{bs['eer'][0]*100:4.1f},{bs['eer'][1]*100:4.1f}]  AUC {tf['roc_auc']:.4f}  "
            f"EER-thr(descriptive) {tf['eer_threshold_descriptive_only']:.4f} | @0.5 acc {a['accuracy']:.3f} prec {a['synthetic_precision'] if a['synthetic_precision'] is not None else float('nan'):.3f} "
            f"rec {a['synthetic_recall']:.3f} F1 {a['f1_synthetic'] if a['f1_synthetic'] is not None else float('nan'):.3f} FPR {a['genuine_fpr']:.3f}")
for scope_name in ("paired", "all_scored"):
    print(f"\\n===== {scope_name.upper()}  (threshold 0.5 is the frozen placeholder; EER thresholds are descriptive only) =====")
    for c in CONDS: print(("  [required]   " if c in REQUIRED else "  [sensitivity]") + row(c, RES[scope_name][c]))
""")

code("""
# 8. Official ratio (paired set, required conditions) + 8 kHz delta; sensitivity deltas kept separate
eers = {c: (RES["paired"][c]["threshold_free"]["eer"] if RES["paired"][c] else None) for c in REQUIRED}
VERDICT = el.robustness_verdict(eers, required=tuple(REQUIRED), n_pairs=len(paired_ids))
print("Official robustness verdict (paired set):", json.dumps(VERDICT, indent=1))
if VERDICT["verdict"] == "PASS": print(f"PASS: worst degraded EER / clean EER = {VERDICT['ratio']:.2f} <= 2.5 ({VERDICT['worst_condition']})")
elif VERDICT["verdict"] == "FAIL": print(f"FAIL: ratio {VERDICT['ratio']:.2f} > 2.5 ({VERDICT['worst_condition']})")
else: print("Verdict:", VERDICT["verdict"], "-", VERDICT.get("note", "a required condition is missing"))
if "eer_8k_minus_clean_pp" in VERDICT: print(f"EER(16->8->16 kHz) - EER(clean) = {VERDICT['eer_8k_minus_clean_pp']:+.2f} percentage points")

# paired per-clip score shifts (same clip, clean vs condition): median |delta| and sign-flips at 0.5
clean = OK[OK.condition == "clean"].set_index("clip_id").cand_fake_score.astype(float)
SHIFTS = {}
for c in CONDS:
    if c == "clean": continue
    d = OK[OK.condition == c].set_index("clip_id").cand_fake_score.astype(float)
    j = clean.index.intersection(d.index)
    delta = (d[j] - clean[j])
    flips = int(((d[j] >= 0.5) != (clean[j] >= 0.5)).sum())
    e_c = RES["paired"][c]["threshold_free"]["eer"] if RES["paired"][c] else None
    SHIFTS[c] = {"group": dg.CONDITIONS[c]["group"], "n_pairs": int(len(j)), "median_abs_delta": float(delta.abs().median()),
                 "p95_abs_delta": float(delta.abs().quantile(0.95)), "decision_flips_at_0.5": flips,
                 "eer_minus_clean_pp": (e_c - eers["clean"]) * 100 if (e_c is not None and eers["clean"] is not None) else None}
print("\\nPaired per-clip shifts vs clean (sensitivity rows are NOT part of the official ratio):")
print(pd.DataFrame(SHIFTS).T.to_string())
""")

md("""
## 9. Unseen generators (optional, separate)

In-the-Wild has no generator labels, and MLAAD (which does) is in the candidate's training table, so **generator-disjoint evaluation cannot be
established from these datasets**. To test a genuinely unseen generator, supply clips yourself:

```
/content/unseen_generators/<GeneratorName>/synthetic/*.wav     # made with that generator
/content/unseen_generators/<GeneratorName>/genuine/*.wav       # optional: matched genuine recordings (same speakers/channel if possible)
/content/unseen_generators/<GeneratorName>/PROVENANCE.md       # generator version, release date, source of the voices, how clips were made
```
The cell labels each generator's result with the evidence you give (release date vs the candidate's training cut-off, mid-2025). Without matched genuine
clips only synthetic recall at the frozen 0.5 and the score distribution are reported (an EER against In-the-Wild genuine clips would be a cross-corpus
confound and is **not** computed). Nothing here feeds back into any threshold.
""")

code("""
# 9. Unseen generators (runs only if UNSEEN_DIR exists)
UNSEEN = {}
if os.path.isdir(UNSEEN_DIR):
    for gen in sorted(os.listdir(UNSEEN_DIR)):
        gdir = os.path.join(UNSEEN_DIR, gen)
        if not os.path.isdir(gdir): continue
        prov = open(os.path.join(gdir, "PROVENANCE.md")).read() if os.path.exists(os.path.join(gdir, "PROVENANCE.md")) else "MISSING"
        rows = []
        for lab in ("synthetic", "genuine"):
            for fn in sorted(os.listdir(os.path.join(gdir, lab))) if os.path.isdir(os.path.join(gdir, lab)) else []:
                r = {"generator": gen, "label": lab, "file": fn}
                try:
                    y, sr = dh.load_audio(os.path.join(gdir, lab, fn)); x16, info = dh.validate(dh.to_mono_16k(y, sr))
                    r.update(status="ok", cand_fake_score=ad.score_clip(cand, x16)["fake_score_whole"], **info)
                except dh.AudioRejected as e:
                    r.update(status="rejected", reject_reason=str(e))
                rows.append(r)
        d = pd.DataFrame(rows); ok = d[d.status == "ok"] if len(d) else d
        res = {"provenance": prov[:2000], "n_files": len(d), "n_scored": int(len(ok)), "unseen_claim": "as stated in PROVENANCE.md; not independently verified"}
        if len(ok) and set(ok.label) == {"synthetic", "genuine"}: res["metrics_within_generator_set"] = metrics(ok, n_boot=200)
        elif len(ok):
            syn = ok[ok.label == "synthetic"].cand_fake_score.astype(float)
            res.update(synthetic_recall_at_0_5=float((syn >= 0.5).mean()) if len(syn) else None, score_quantiles=syn.describe().to_dict(),
                       note="no matched genuine clips: EER not computed (cross-corpus EER would be confounded)")
        UNSEEN[gen] = res
        d.to_csv(os.path.join(RUN_DIR, f"unseen_{gen}.csv"), index=False)
    print(json.dumps(UNSEEN, indent=1, default=str))
else:
    print("UNSEEN_DIR not present: unseen-generator evaluation NOT performed (cannot be established from In-the-Wild/MLAAD).")
""")

code("""
# 10. Save (no audio) + download
P.to_csv(os.path.join(RUN_DIR, "predictions_conditions.csv"), index=False)
cov.to_csv(os.path.join(RUN_DIR, "coverage_by_condition.csv"))
el.jdump({"run_name": RUN_NAME, "scope": "paired robustness on held-out ITW pilot subset (clips already inspected for clean scores; no decisions taken here)",
          "frozen_threshold": FROZEN_THRESHOLD, "threshold_policy": "0.5 retained until a validation-selected threshold exists; EER thresholds descriptive only",
          "paired_n": len(paired_ids), "coverage": cov.reset_index().to_dict("records"), "metrics": RES, "official_verdict": VERDICT,
          "paired_shifts": SHIFTS, "unseen_generators": UNSEEN or "not performed", "env": ENV,
          "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}, os.path.join(RUN_DIR, "metrics.json"))
zip_path = shutil.make_archive(os.path.join(WORK, RUN_NAME), "zip", RUN_DIR)
print(sorted(os.listdir(RUN_DIR)), zip_path)
try:
    from google.colab import files; files.download(zip_path)
except ImportError: pass
""")

md("""
## Report back
Zip + executed notebook into `evaluation/runs/<run_name>/`. Paste the outputs of cells 3 (ffmpeg line), 7 and 8.

## EER protocol going forward (recorded in `evaluation/README.md` §7)
Checkpoint selection and the deployment threshold come from **validation data only** (speaker/generator-grouped, disjoint from every test set).
Test EER and its crossing threshold are reported separately and never deployed. Until a validation set exists, 0.5 stays.
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, json.dumps(SHA))
