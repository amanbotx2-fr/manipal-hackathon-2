"""Build evaluation/06_denoiser_frontend_colab.ipynb.

Bounded noise-robustness experiment, stage 1 of 2: a lightweight denoising FRONT-END in front
of the frozen candidate (AntiDeepfake Wav2Vec2-Small, weights untouched, threshold 0.5).
  A. Development stage on the 244 already-inspected held-out clips (development evidence only):
     every front-end x every required condition, paired, same front-end on clean audio too;
     artefact-removal check on clean synthetic clips. A pre-registered rule freezes ONE front-end.
  B. Fresh evaluation: new selection excluding all 500 inspected clip IDs; baseline (no front-end)
     and the frozen front-end on clean, narrowband, two codecs, noise. Absolute EERs, coverage,
     unrounded worst/clean ratio, clean-margin check.
If stage B fails, the fine-tuning notebook (07) is the next step; it is NOT started here.
Embeds dhwani_infer.py (read only), eval_lib.py, antideepfake_hf.py, degrade.py, frontend.py and
the inspected-ID lists. Standard library only. Run: python3 evaluation/build_06_notebook.py
"""

import csv
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
    "frontend.py": (HERE / "frontend.py").read_text(),
}
DEV_MANIFEST = (RUN04 / "manifest_heldout_in_the_wild.csv").read_text()
PILOT_IDS = (HERE / "dev_evidence/itw_pilot_clip_ids.txt").read_text().split()
HELD_IDS = [r["clip_id"] for r in csv.DictReader((RUN04 / "manifest_heldout_in_the_wild.csv").open())]
INSPECTED = sorted(set(PILOT_IDS) | set(HELD_IDS))
assert len(INSPECTED) == 500, len(INSPECTED)
SHA = {k: hashlib.sha256((v.rstrip() + "\n").encode()).hexdigest() for k, v in SRC.items()}
OUT = HERE / "06_denoiser_frontend_colab.ipynb"
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip().splitlines(keepends=True)})


md("""
# 06 — Denoising front-end experiment (bounded; candidate weights and threshold untouched)

Run 20261024T151905Z: the candidate's EER goes from 2.46 % (clean) to 15.98 % under white noise at 10 dB SNR → ratio 6.50 > 2.5 (**FAIL**).
This notebook tests whether a **lightweight denoiser in front of the detector** fixes that *without* destroying the synthesis artefacts the
detector relies on. It is an experiment, not an assumed fix. Noise stays a requirement; the threshold stays **0.5**; the checkpoint is unchanged.

**Stage A — development (already-inspected 244 clips; development evidence only)**
- Front-ends: `none`, `nr_stationary` (spectral gating, pure DSP), `deepfilternet3` (2 M-param neural denoiser, if it installs). Fixed settings, see `frontend.py`.
- The same front-end is applied to **every** input: genuine and synthetic, **clean and degraded** (clean → front-end → detector is exactly the deployed path).
- Paired per clip across required conditions {clean, 16→8→16 kHz, Opus 16 kb/s, MP3 32 kb/s, white noise 10 dB}. Per-condition rejections/coverage reported.
- **Artefact-removal check:** clean EER with front-end vs without, and how clean *synthetic* scores move (a front-end that lowers spoof scores on clean audio is eating artefacts).
- **Pre-registered freeze rule** (`eval_lib.frontend_selection`): eligible only if dev clean EER ≤ baseline clean EER + `DEV_SELECTION_MARGIN_PP`; among eligible, lowest worst-degraded EER; ties → simplest; frozen only if it beats the baseline's worst EER. Otherwise **no front-end is frozen** and the experiment is declared failed → fine-tuning (notebook 07, not run here).

**Stage B — fresh evaluation (new clips, no decisions)**
- Selection seed `20261028`, 150 + 150, speaker round-robin, **excluding all 500 inspected clip IDs** (200 pilot + 300 held-out). In-the-Wild shares all 54 speakers
  with the inspected sets (speaker-disjoint impossible) and the candidate's training data includes VoxCeleb2 celebrity speech: both limitations apply to every number.
- Baseline (`none`) and the frozen front-end on the five required conditions. Reported: absolute EERs with CIs, coverage, **unrounded** worst/clean ratio,
  PASS/FAIL/INCOMPLETE, **and** a clean-margin check (fresh clean EER with front-end must be within `OFFICIAL_MARGIN_PP` of the fresh baseline clean EER; a higher
  clean EER is never allowed to make the ratio look better).

**Margins.** The requirement we were given says Phase-1 EERs must "remain within 2 points"; we interpret that **provisionally as 2 percentage points**
(`OFFICIAL_MARGIN_PP = 2.0`, pending organiser clarification) and apply it to the Stage B clean-margin check. Stage A additionally uses a **stricter
development selection margin** of 1 pp (`DEV_SELECTION_MARGIN_PP = 1.0`) to choose the front-end; that is our own choice, **not** the official requirement.

Same T4 runtime as notebooks 04/05 preferred (model + cached audio reused); standalone also works. ≈ 25–35 min.
""")

code("""
# 1. Settings (fixed before any score is seen)
FRONTENDS = ["none", "nr_stationary", "deepfilternet3"]   # 'none' is the baseline; order = simplicity order for tie-breaks
TRY_DEEPFILTERNET = True        # pip install deepfilternet; marked unavailable if it fails (never silently skipped)
OFFICIAL_MARGIN_PP = 2.0        # PROVISIONAL interpretation of "Phase-1 EERs must remain within 2 points" = 2 percentage points (pending organiser clarification)
DEV_SELECTION_MARGIN_PP = 1.0   # stricter development-only margin for choosing a front-end at Stage A; NOT the official requirement
FROZEN_THRESHOLD = 0.5          # retained: no validation-selected threshold exists
REQUIRED = ["clean", "narrowband_8k", "codec_1", "codec_2", "noise"]
FRESH_SEED = 20261028
FRESH_N = 150                   # per class
N_BOOT = 500
WORK = "/content/vcd_eval"
USE_DRIVE = False
DRIVE_DIR = "/content/drive/MyDrive/vcd_cache"
""")

code("""
# 2. Dependencies
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxruntime", "librosa", "soundfile", "huggingface_hub",
                "transformers", "safetensors", "scipy", "noisereduce"], check=True)
DFN_INSTALL = {"attempted": TRY_DEEPFILTERNET}
if TRY_DEEPFILTERNET:
    r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "deepfilternet"], capture_output=True, text=True)
    DFN_INSTALL.update(returncode=r.returncode, tail=(r.stderr or r.stdout)[-400:])
print(DFN_INSTALL)
""")

for name, text in SRC.items():
    code(f"%%writefile {name}\n" + text)
code("%%writefile manifest_dev_heldout_in_the_wild.csv\n" + DEV_MANIFEST)
code("%%writefile inspected_clip_ids.txt\n" + "\n".join(INSPECTED) + "\n")

code(f"""
# 3. Imports, provenance, environment, front-end availability
import os, json, time, hashlib, platform, datetime, importlib, traceback, shutil, csv
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import dhwani_infer as dh, eval_lib as el, antideepfake_hf as ad, degrade as dg, frontend as fe
for m in (dh, el, ad, dg, fe): importlib.reload(m)
EXPECTED = {json.dumps(SHA, indent=4)}
CODE_SHA256 = {{f: hashlib.sha256((open(f, encoding="utf-8").read().rstrip() + "\\n").encode()).hexdigest() for f in EXPECTED}}
assert CODE_SHA256 == EXPECTED, CODE_SHA256
assert FROZEN_THRESHOLD == 0.5
INSPECTED = set(l.strip() for l in open("inspected_clip_ids.txt") if l.strip()); assert len(INSPECTED) == 500
DEV_MAN = list(csv.DictReader(open("manifest_dev_heldout_in_the_wild.csv"))); assert len(DEV_MAN) == 300
AVAIL = fe.available(); print("front-ends:", AVAIL)
FRONTENDS_RUN = [f for f in FRONTENDS if AVAIL.get(f) is True]
print("running front-ends:", FRONTENDS_RUN, "| unavailable:", {{f: AVAIL.get(f) for f in FRONTENDS if AVAIL.get(f) is not True}})
FFMPEG = dg.ffmpeg_available(); print("ffmpeg:", FFMPEG)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import transformers, librosa, soundfile, scipy, noisereduce
ENV = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
           librosa=librosa.__version__, soundfile=soundfile.__version__, scipy=scipy.__version__, noisereduce=noisereduce.__version__,
           device=DEVICE, gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, ffmpeg=FFMPEG, frontends_available=AVAIL,
           dfn_install=DFN_INSTALL, code_sha256=CODE_SHA256)
print(json.dumps(ENV, indent=1, default=str))
""")

code("""
# 4. Candidate (frozen weights; reuse runtime object if present)
from huggingface_hub import hf_hub_download
if "cand" in globals() and getattr(cand, "report", None):
    print("reusing loaded candidate")
else:
    if USE_DRIVE:
        from google.colab import drive; drive.mount("/content/drive")
    try: AD_PATH = hf_hub_download(ad.MODEL_REPO, ad.MODEL_FILE, revision=ad.MODEL_REVISION, local_files_only=True)
    except Exception: AD_PATH = hf_hub_download(ad.MODEL_REPO, ad.MODEL_FILE, revision=ad.MODEL_REVISION)
    assert os.path.getsize(AD_PATH) == ad.MODEL_BYTES and dh.sha256_file(AD_PATH) == ad.MODEL_SHA256
    AD_CFG = hf_hub_download(ad.MODEL_REPO, "config.json", revision=ad.MODEL_REVISION)
    cand = ad.AntiDeepfakeHF.load(AD_PATH, os.path.dirname(AD_CFG), device=DEVICE)
print({k: cand.report[k] for k in ("n_encoder_mapped", "n_params_encoder")})
BASE = os.path.join(DRIVE_DIR if USE_DRIVE else WORK, "runs")
utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_NAME = f"{utc}_denoiser_frontend_itw"; RUN_DIR = os.path.join(BASE, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
el.jdump({"settings": dict(FRONTENDS=FRONTENDS, FRONTENDS_RUN=FRONTENDS_RUN, OFFICIAL_MARGIN_PP=OFFICIAL_MARGIN_PP, DEV_SELECTION_MARGIN_PP=DEV_SELECTION_MARGIN_PP,
          margin_note="official 2 pp is a provisional interpretation of 'within 2 points'; 1 pp is a development selection margin only", FROZEN_THRESHOLD=FROZEN_THRESHOLD,
          REQUIRED=REQUIRED, FRESH_SEED=FRESH_SEED, FRESH_N=FRESH_N, N_BOOT=N_BOOT, frontend_specs=fe.FRONTENDS, conditions=dg.CONDITIONS,
          noise_seed=dg.NOISE_SEED, candidate=dict(repo=ad.MODEL_REPO, revision=ad.MODEL_REVISION, sha256=ad.MODEL_SHA256)),
          "env": ENV, "started_utc": utc}, os.path.join(RUN_DIR, "run_config.json"))
print(RUN_DIR)
""")

code("""
# 5. Shared scoring loop: decode once -> degrade -> front-end -> validate -> score (paired; every failure recorded)
AUD = os.path.join(WORK, "audio"); os.makedirs(os.path.join(AUD, "heldout"), exist_ok=True); os.makedirs(os.path.join(AUD, "fresh"), exist_ok=True)
_zf = None
def ensure_audio(m, sub):
    global _zf
    p = os.path.join(AUD, sub, os.path.basename(m["archive_member"]))
    if not os.path.exists(p):
        if _zf is None:
            src = el.SOURCES["in_the_wild"]; _zf, _ = el.open_remote_zip(src["url"], src["bytes"])
        with open(p, "wb") as f: f.write(_zf.read(m["archive_member"]))
    return p

def run_stage(manifest, sub, frontends, path, stage):
    done = set()
    if os.path.exists(path):
        done = {(r["clip_id"], r["condition"], r["frontend"]) for r in map(json.loads, open(path)) if r}
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as out:
        for i, m in enumerate(manifest):
            base = {"stage": stage, "clip_id": m["clip_id"], "label": m["label"], "speaker_id": m["speaker_id"]}
            try:
                y, sr = dh.load_audio(ensure_audio(m, sub)); x16 = dh.to_mono_16k(y, sr); x16, info = dh.validate(x16); ok = True
            except Exception as e:
                ok, info = False, {"reject_reason": f"{type(e).__name__}: {e}"}
            for cond in REQUIRED:
                xd = None
                for f in frontends:
                    if (m["clip_id"], cond, f) in done: continue
                    row = dict(base, condition=cond, frontend=f, clean_eligible=ok, duration_s=info.get("duration_s"))
                    if not ok:
                        row.update(status="rejected_clean", reject_reason=info["reject_reason"]); out.write(json.dumps(row) + "\\n"); continue
                    try:
                        if xd is None: xd = dg.apply(cond, x16, clip_id=m["clip_id"])
                        t1 = time.perf_counter(); xf = fe.apply(f, xd); row["frontend_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                        xv, vinfo = dh.validate(xf)
                        c = ad.score_clip(cand, xv)
                        row.update(status="ok", cand_fake_score=c["fake_score_whole"], cand_logits=json.dumps(c["logits_fake_real"]),
                                   out_peak_dbfs=vinfo["peak_dbfs"], out_rms_dbfs=vinfo["rms_dbfs"])
                    except dh.AudioRejected as e:
                        row.update(status="rejected_after_frontend", reject_reason=str(e), reject_category=el.reject_category(str(e)))
                    except Exception as e:
                        row.update(status="failed", reject_reason=f"{type(e).__name__}: {str(e)[:300]}")
                    out.write(json.dumps(row, default=str) + "\\n"); out.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(manifest):
                e = time.time() - t0; print(f"[{stage}] {i+1}/{len(manifest)} {e/60:.1f} min, ETA {e/(i+1)*(len(manifest)-i-1)/60:.1f} min")
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]).drop_duplicates(["clip_id", "condition", "frontend"], keep="last")

def eer_table(P, frontends):
    OK = P[P.status == "ok"]
    paired = set.intersection(*[set(OK[(OK.condition == c) & (OK.frontend == f)].clip_id) for c in REQUIRED for f in frontends])
    res, eers = {}, {}
    for f in frontends:
        res[f], eers[f] = {}, {}
        for c in REQUIRED:
            d = OK[(OK.condition == c) & (OK.frontend == f) & OK.clip_id.isin(paired)]
            y = (d.label == "synthetic").astype(int).tolist(); s = d.cand_fake_score.astype(float).tolist()
            m = el.evaluate(y, s, FROZEN_THRESHOLD, n_boot=N_BOOT, seed=FRESH_SEED) if len(set(y)) == 2 else None
            res[f][c] = m; eers[f][c] = m["threshold_free"]["eer"] if m else None
    return paired, res, eers

def print_table(res, eers, frontends, title):
    print(f"\\n===== {title} =====")
    for f in frontends:
        print(f"-- front-end: {f}")
        for c in REQUIRED:
            m = res[f][c]
            if m is None: print(f"   {c:15s} n/a"); continue
            tf, bs, a = m["threshold_free"], m["bootstrap95"], m["at_threshold"]
            print(f"   {c:15s} n={m['n_scored']:3d} EER {tf['eer']*100:5.2f}% [{bs['eer'][0]*100:4.1f},{bs['eer'][1]*100:4.1f}] AUC {tf['roc_auc']:.4f} "
                  f"| @0.5 acc {a['accuracy']:.3f} rec {a['synthetic_recall']:.3f} FPR {a['genuine_fpr']:.3f}")
        d = {c: eers[f][c] for c in REQUIRED if c != "clean"}
        if all(v is not None for v in d.values()) and eers[f]["clean"]:
            w = max(d, key=d.get); print(f"   worst {w} / clean = {d[w]/eers[f]['clean']!r} (unrounded)")
""")

code("""
# 6. STAGE A — development evidence: all front-ends x required conditions on the 244 inspected clips
DEV_PATH = os.path.join(RUN_DIR, "stageA_dev_predictions.jsonl")
PA = run_stage(DEV_MAN, "heldout", FRONTENDS_RUN, DEV_PATH, "A_dev")
covA = PA.groupby(["frontend", "condition"]).status.value_counts().unstack(fill_value=0)
print("Stage A coverage:\\n", covA.to_string())
pairedA, resA, eersA = eer_table(PA, FRONTENDS_RUN)
print(f"paired dev clips: {len(pairedA)}")
print_table(resA, eersA, FRONTENDS_RUN, "STAGE A (development evidence; no decisions on these numbers except the pre-registered freeze rule)")
""")

code("""
# 7. STAGE A — artefact-removal check on CLEAN audio + pre-registered freeze
OKA = PA[(PA.status == "ok") & (PA.condition == "clean") & PA.clip_id.isin(pairedA)]
base = OKA[OKA.frontend == "none"].set_index("clip_id").cand_fake_score.astype(float)
ARTEFACT = {}
for f in FRONTENDS_RUN:
    if f == "none": continue
    d = OKA[OKA.frontend == f].set_index("clip_id").cand_fake_score.astype(float)
    lab = OKA[OKA.frontend == f].set_index("clip_id").label
    j = base.index.intersection(d.index); delta = d[j] - base[j]
    syn, gen = delta[lab[j] == "synthetic"], delta[lab[j] == "genuine"]
    ARTEFACT[f] = {"n_pairs": int(len(j)),
                   "clean_eer_none": eersA["none"]["clean"], "clean_eer_frontend": eersA[f]["clean"],
                   "clean_eer_delta_pp": (eersA[f]["clean"] - eersA["none"]["clean"]) * 100,
                   "synthetic_median_delta": float(syn.median()), "synthetic_frac_dropped_over_0.25": float((syn < -0.25).mean()),
                   "synthetic_flips_to_genuine_at_0.5": int(((d[j] < 0.5) & (base[j] >= 0.5) & (lab[j] == "synthetic")).sum()),
                   "genuine_median_delta": float(gen.median()), "genuine_flips_to_synthetic_at_0.5": int(((d[j] >= 0.5) & (base[j] < 0.5) & (lab[j] == "genuine")).sum())}
print("Artefact-removal check on clean audio (front-end vs none, same clips):")
print(pd.DataFrame(ARTEFACT).T.to_string())
FREEZE = el.frontend_selection(eersA, baseline="none", margin_pp=DEV_SELECTION_MARGIN_PP, required=tuple(REQUIRED), order=[f for f in FRONTENDS_RUN if f != "none"])
print("\\nPre-registered freeze rule (development selection margin", DEV_SELECTION_MARGIN_PP, "pp; stricter than the provisional official 2 pp):", FREEZE["rule"])
print(pd.DataFrame(FREEZE["table"]).T.to_string())
FROZEN = FREEZE["frozen_frontend"]
print("\\nFROZEN FRONT-END:", FROZEN if FROZEN else "NONE (no front-end is eligible AND better than baseline -> denoiser experiment FAILED at stage A; proceed to fine-tuning plan)")
el.jdump({"artefact_check": ARTEFACT, "freeze": FREEZE, "frozen_frontend": FROZEN, "paired_dev_n": len(pairedA), "dev_eers": eersA,
          "dev_metrics": resA}, os.path.join(RUN_DIR, "stageA_summary.json"))
""")

code("""
# 8. STAGE B — fresh selection (excludes all 500 inspected IDs), then baseline + frozen front-end on required conditions
src = el.SOURCES["in_the_wild"]
if _zf is None: _zf, _ = el.open_remote_zip(src["url"], src["bytes"])
names = _zf.namelist(); meta_m = el.find_member(names, src["meta_suffix"]); prefix = meta_m[:-len("meta.csv")]
recs = el.parse_itw_meta(_zf.read(meta_m).decode("utf-8")); nameset = set(names)
for r in recs: r["archive_member"] = prefix + r["clip_id"] if prefix + r["clip_id"] in nameset else ""
FRESH, fmeta = el.select_subset(recs, "in_the_wild", FRESH_SEED, FRESH_N, n_synthetic=FRESH_N, exclude_clip_ids=INSPECTED)
assert not ({r["clip_id"] for r in FRESH} & INSPECTED)
fmeta["limitations"] = ["speakers shared with inspected sets (all 54; speaker-disjoint impossible on In-the-Wild)",
                        "candidate trained on VoxCeleb2 celebrity speech: speaker/recording overlap with In-the-Wild possible, unverified",
                        "In-the-Wild not in the candidate's training table (card)"]
pd.DataFrame(FRESH).to_csv(os.path.join(RUN_DIR, "manifest_fresh_in_the_wild.csv"), index=False)
el.jdump(fmeta, os.path.join(RUN_DIR, "selection_fresh_in_the_wild.json"))
print("fresh:", fmeta["n_selected"], fmeta["selected"]["by_label"], "excluded", fmeta["exclusion"], "| clip-ID sha256", fmeta["clip_id_list_sha256"][:16])
FRONTENDS_B = ["none"] + ([FROZEN] if FROZEN else [])
FRESH_PATH = os.path.join(RUN_DIR, "stageB_fresh_predictions.jsonl")
PB = run_stage(FRESH, "fresh", FRONTENDS_B, FRESH_PATH, "B_fresh")
covB = PB.groupby(["frontend", "condition"]).status.value_counts().unstack(fill_value=0)
covB["selected"] = covB.sum(axis=1); covB["coverage"] = (covB.get("ok", 0) / covB["selected"]).round(4)
print("Stage B coverage (every rejection shown):\\n", covB.to_string())
pairedB, resB, eersB = eer_table(PB, FRONTENDS_B)
print(f"paired fresh clips: {len(pairedB)}")
print_table(resB, eersB, FRONTENDS_B, "STAGE B (fresh clips; threshold 0.5 frozen; EER thresholds descriptive only)")
""")

code("""
# 9. STAGE B — verdicts: official ratio (unrounded) per pipeline + clean-margin check
VERDICTS = {}
for f in FRONTENDS_B:
    v = el.robustness_verdict({c: eersB[f][c] for c in REQUIRED}, required=tuple(REQUIRED), n_pairs=len(pairedB))
    v["ratio_unrounded"] = v.get("ratio"); VERDICTS[f] = v
    print(f"\\n[{f}] verdict {v['verdict']} | worst {v.get('worst_condition')} EER {v.get('worst_eer')} | clean EER {eersB[f]['clean']} | "
          f"ratio (unrounded) {v.get('ratio')!r} | EER(8k)-EER(clean) {v.get('eer_8k_minus_clean_pp')} pp")
if FROZEN:
    MARGIN = el.clean_margin_check(eersB[FROZEN]["clean"], eersB["none"]["clean"], OFFICIAL_MARGIN_PP)
    MARGIN["basis"] = "provisional interpretation of the supplied requirement 'Phase-1 EERs must remain within 2 points' as 2 percentage points; pending organiser clarification"
    MARGIN_DEV = el.clean_margin_check(eersB[FROZEN]["clean"], eersB["none"]["clean"], DEV_SELECTION_MARGIN_PP)
    print("\\nClean-margin check (fresh, official provisional 2 pp):", MARGIN)
    print("Clean-margin check (fresh, stricter development 1 pp, informational):", MARGIN_DEV)
    final = ("PASS" if VERDICTS[FROZEN]["verdict"] == "PASS" and MARGIN["within_margin"] else
             ("FAIL (ratio ok but clean EER worsened beyond margin: not accepted)" if VERDICTS[FROZEN]["verdict"] == "PASS" else VERDICTS[FROZEN]["verdict"]))
    print(f"\\nFINAL (frozen front-end '{FROZEN}'): {final}")
    abs_base = {c: round(eersB["none"][c] * 100, 2) for c in REQUIRED}; abs_fe = {c: round(eersB[FROZEN][c] * 100, 2) for c in REQUIRED}
    print("   absolute EERs (%) fresh: baseline", abs_base, " with front-end", abs_fe)
else:
    MARGIN, MARGIN_DEV, final = None, None, "NOT APPLICABLE: no front-end frozen at stage A; fresh baseline numbers reported only"
    print("\\nFINAL:", final)
""")

code("""
# 10. Save (no audio) + download
PA.to_csv(os.path.join(RUN_DIR, "stageA_dev_predictions.csv"), index=False); PB.to_csv(os.path.join(RUN_DIR, "stageB_fresh_predictions.csv"), index=False)
covA.to_csv(os.path.join(RUN_DIR, "stageA_coverage.csv")); covB.to_csv(os.path.join(RUN_DIR, "stageB_coverage.csv"))
el.jdump({"run_name": RUN_NAME, "frozen_frontend": FROZEN, "official_margin_pp_provisional": OFFICIAL_MARGIN_PP, "dev_selection_margin_pp": DEV_SELECTION_MARGIN_PP,
          "frozen_threshold": FROZEN_THRESHOLD,
          "stageA": {"paired_n": len(pairedA), "eers": eersA, "artefact_check": ARTEFACT, "freeze": FREEZE},
          "stageB": {"paired_n": len(pairedB), "eers": eersB, "metrics": resB, "verdicts": VERDICTS, "clean_margin_official": MARGIN, "clean_margin_dev": MARGIN_DEV, "final": final,
                     "selection": fmeta, "coverage": covB.reset_index().to_dict("records")},
          "env": ENV, "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}, os.path.join(RUN_DIR, "metrics.json"))
zip_path = shutil.make_archive(os.path.join(WORK, RUN_NAME), "zip", RUN_DIR); print(sorted(os.listdir(RUN_DIR)), zip_path)
try:
    from google.colab import files; files.download(zip_path)
except ImportError: pass
""")

md("""
## Report back
Zip + executed notebook → `evaluation/runs/<run_name>/`. Paste cells 3 (front-end availability + ffmpeg), 6, 7, 8, 9.

## If the final verdict is not PASS
The denoiser experiment is recorded as failed and the next step is **notebook 07: noise-augmented fine-tuning** (not built or started yet):
start from the verified candidate weights; noise + codec augmentation on **both** classes; separate training and validation data with grouped splits
(speaker / generator where metadata permits); checkpoint selection by **validation EER**; cross-entropy retained; deployment threshold chosen on validation only;
final evaluation on a new fresh selection excluding every inspected ID (now 800 after this run) with the same five conditions and the same clean-margin rule.
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, json.dumps(SHA))
