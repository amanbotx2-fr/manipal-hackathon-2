"""Build evaluation/04_candidate_comparison_colab.ipynb.

Compares the failed Dhwani baseline with the candidate AntiDeepfake Wav2Vec2-Small on a
FRESH held-out In-the-Wild subset (clip-disjoint from the 200 pilot clips). Embeds
src/vcd/dhwani_infer.py (read only), eval_lib.py, antideepfake_hf.py and the pilot
clip-ID list via %%writefile. Standard library only. Run: python3 evaluation/build_04_notebook.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = {
    "dhwani_infer.py": (ROOT / "src/vcd/dhwani_infer.py").read_text(),
    "eval_lib.py": (HERE / "eval_lib.py").read_text(),
    "antideepfake_hf.py": (HERE / "antideepfake_hf.py").read_text(),
    "fairseq_parity_ref.py": (HERE / "fairseq_parity_ref.py").read_text(),
}
PILOT_IDS = (HERE / "dev_evidence/itw_pilot_clip_ids.txt").read_text()
SHA = {k: hashlib.sha256((v.rstrip() + "\n").encode()).hexdigest() for k, v in SRC.items()}
OUT = HERE / "04_candidate_comparison_colab.ipynb"
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip().splitlines(keepends=True)})


md("""
# 04 — Candidate comparison on a fresh held-out In-the-Wild subset

**Baseline (failed):** third-party Dhwani ONNX (`ee01871`), `train_peak`, as in run 20261024T123319Z (In-the-Wild pilot: AUC 0.456, EER 0.561).
**Candidate:** [AntiDeepfake Wav2Vec2-Small](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) (NII Yamagishi Lab, 95 M params,
revision `9a13264`, **CC BY-NC-SA 4.0: research use; a commercial app would need permission**). Published In-the-Wild: EER 4.24 %, AUC 0.993
(self-reported; 5.31 % in the independent lab260 arena). In-the-Wild is **not** in its training table; VoxCeleb2 (celebrity YouTube speech) is,
so speaker overlap with In-the-Wild celebrities cannot be excluded.

**Held-out design (fixed before any score is seen)**
- In-the-Wild only (ASVspoof 2019 LA access from Colab is blocked with HTTP 403, and it is in the candidate's training set anyway).
- New seed, `HELDOUT_N` genuine + `HELDOUT_N` spoof, speaker round-robin, metadata only, **excluding the 200 pilot clip IDs** (development evidence).
  Speaker-disjointness is impossible: the pilot touched all 54 In-the-Wild speakers. Stated with every result.
- Both models score the same clips. Same 2–10 s policy (reject < 2 s, analyse first 10 s), same decoding (`dhwani_infer`), batch size 1.
- Dhwani: `train_peak`, mean of 4 s windows (unchanged). Candidate: published recipe (layer-norm, whole clip, fake = softmax index 0);
  a 4 s-window mean is recorded as a secondary number.
- Thresholds are **not** tuned: 0.5 for both (symmetric default). The card's "EER @ 0.8128" is **not** used as an operating point:
  it is the EER threshold the authors found *on In-the-Wild itself* with their whole-file pipeline, so it is tuned on our test data and
  depends on a pipeline we have not proven identical. It is shown as context only. Primary comparison is threshold-free (ROC-AUC, EER)
  with a paired bootstrap on the shared clips.
- **Conversion:** the candidate is loaded without fairseq (strict key mapping into `transformers`; architecture fields checked against
  fairseq defaults; see the docstring of `antideepfake_hf.py` for the side-by-side read of activations, norms, padding, masks, pooling and
  score direction). **Parity with the authors' code is "unverified" until cell 13 reports it.** Cell 13 builds a separate Python 3.8 venv
  with fairseq 0.12.2 and scores identical inputs with the authors' reference code.
- **Development-evidence gate (cell 9):** the candidate is first scored on the 200 pilot clips. AUC < 0.80 means **stop and investigate**;
  it does not by itself prove the mapping is wrong, and a high AUC does not prove parity. Nothing is tuned on it.
- Failures are kept; coverage is reported separately. Results are a **held-out pilot subset**, not a full-dataset benchmark.

Runtime: **GPU (T4)** recommended; the candidate runs on CUDA, Dhwani on CPU ORT as before (≈ 2 s per window).
""")

code("""
# 1. Settings (fixed before any result is seen)
HELDOUT_SEED = 20261026        # new seed (pilot used 20261025)
HELDOUT_N = 150                # per class; 300 clips total
MIN_PILOT_DEV_AUC = 0.80       # dev-evidence gate: below -> stop and investigate (not proof of a mapping error; not tuning)
CAND_WINDOW, CAND_HOP = 64000, 32000   # secondary 4 s-window variant for the candidate
EXPORT_ONNX = True             # deployment feasibility check: export candidate to ONNX + ORT parity on a few clips
TRY_FAIRSEQ_PARITY = True      # build a Python 3.8 venv (uv) with fairseq 0.12.2 and compare logits on identical inputs
PARITY_N_CLIPS = 12            # clips compared in the parity check (whole-clip inputs)
USE_DRIVE = False
DRIVE_DIR = "/content/drive/MyDrive/vcd_cache"
WORK = "/content/vcd_eval"
RUN_NAME = None
""")

code("""
# 2. Dependencies
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxruntime", "librosa", "soundfile", "huggingface_hub",
                "transformers", "safetensors", "onnx", "uv"], check=True)
""")

for name, text in SRC.items():
    code(f"%%writefile {name}\n" + text)
code("%%writefile itw_pilot_clip_ids.txt\n" + PILOT_IDS)

code(f"""
# 3. Imports, code provenance, environment
import os, json, time, hashlib, platform, datetime, importlib, traceback, shutil
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import dhwani_infer as dh, eval_lib as el, antideepfake_hf as ad
for m in (dh, el, ad): importlib.reload(m)
EXPECTED = {json.dumps(SHA, indent=4)}
CODE_SHA256 = {{f: hashlib.sha256((open(f, encoding="utf-8").read().rstrip() + "\\n").encode()).hexdigest() for f in EXPECTED}}
PARITY = {{"status": "unverified", "attempted": False}}   # overwritten by cell 13 only if the reference run succeeds
assert CODE_SHA256 == EXPECTED, CODE_SHA256
PILOT_IDS = [l.strip() for l in open("itw_pilot_clip_ids.txt") if l.strip()]
assert len(PILOT_IDS) == 200 and len(set(PILOT_IDS)) == 200
assert dh.DECISION_THRESHOLD == 0.5 and dh.LABELS == {{0: "genuine", 1: "synthetic"}}
import onnxruntime as ort, transformers, librosa, soundfile, sklearn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
           onnxruntime=ort.__version__, numpy=np.__version__, pandas=pd.__version__, librosa=librosa.__version__,
           soundfile=soundfile.__version__, sklearn=sklearn.__version__, device=DEVICE,
           gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, code_sha256=CODE_SHA256)
print(json.dumps(ENV, indent=1))
if DEVICE != "cuda": print("WARNING: no GPU; the candidate will run on CPU (slower, still fine for 500 clips)")
""")

code("""
# 4. Checkpoints: Dhwani (pinned, sha256) and candidate (pinned revision, sha256). Cached copies reused.
from huggingface_hub import hf_hub_download
if USE_DRIVE:
    from google.colab import drive; drive.mount("/content/drive"); os.makedirs(DRIVE_DIR, exist_ok=True)
def get(repo, fname, rev, sha, nbytes):
    cands = [os.path.join(DRIVE_DIR, fname)] if USE_DRIVE else []
    try: cands.append(hf_hub_download(repo, fname, revision=rev, local_files_only=True))
    except Exception: pass
    p = next((c for c in cands if os.path.exists(c) and os.path.getsize(c) == nbytes), None)
    src = "cache"
    if p is None:
        src = "downloaded"; print("downloading", repo, fname, f"({nbytes/1e9:.2f} GB)")
        p = hf_hub_download(repo, fname, revision=rev, **({"local_dir": DRIVE_DIR} if USE_DRIVE else {}))
    assert os.path.getsize(p) == nbytes
    got = dh.sha256_file(p); assert got == sha, f"{fname} sha256 mismatch: {got}"
    print(f"{src}: {p} sha256 ok"); return p, src
DH_PATH, DH_SRC = get(dh.MODEL_REPO, dh.MODEL_FILE, dh.MODEL_REVISION, dh.MODEL_SHA256, dh.MODEL_BYTES)
AD_PATH, AD_SRC = get(ad.MODEL_REPO, ad.MODEL_FILE, ad.MODEL_REVISION, ad.MODEL_SHA256, ad.MODEL_BYTES)
AD_CFG = hf_hub_download(ad.MODEL_REPO, "config.json", revision=ad.MODEL_REVISION)
""")

code("""
# 5. Load both models. Candidate: strict fairseq->transformers mapping (fails loudly on any leftover tensor)
sess = dh.load_session(DH_PATH, verify_sha=False)
PROBE = dh.interface_self_test(sess)
assert PROBE["len_64000_batch1"]["ok"]
cand = ad.AntiDeepfakeHF.load(AD_PATH, os.path.dirname(AD_CFG), device=DEVICE)
print(json.dumps({k: v for k, v in cand.report.items() if k not in ("dropped_pretrain_only", "architecture")}, indent=1))
print("architecture fields == fairseq defaults:", cand.report["architecture"])
print("dropped (pre-training only):", cand.report["dropped_pretrain_only"])
assert cand.report["n_params_encoder"] + cand.report["n_params_head"] > 94_000_000
print(ad.verify_notes())
""")

code("""
# 6. Run folder
BASE = os.path.join(DRIVE_DIR if USE_DRIVE else WORK, "runs")
utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_NAME = RUN_NAME or f"{utc}_heldout_itw_dhwani_vs_antideepfake_small"
RUN_DIR = os.path.join(BASE, RUN_NAME); os.makedirs(RUN_DIR, exist_ok=True)
for d in ("audio/dev", "audio/heldout"): os.makedirs(os.path.join(WORK, d), exist_ok=True)
SETTINGS = dict(HELDOUT_SEED=HELDOUT_SEED, HELDOUT_N=HELDOUT_N, MIN_PILOT_DEV_AUC=MIN_PILOT_DEV_AUC, CAND_WINDOW=CAND_WINDOW,
                CAND_HOP=CAND_HOP, EXPORT_ONNX=EXPORT_ONNX, dhwani=dict(repo=dh.MODEL_REPO, revision=dh.MODEL_REVISION,
                sha256=dh.MODEL_SHA256, preprocessing="train_peak", clip_score="mean of 4 s windows", threshold=0.5),
                candidate=dict(repo=ad.MODEL_REPO, revision=ad.MODEL_REVISION, sha256=ad.MODEL_SHA256, license=ad.LICENSE,
                recipe="layer_norm whole clip, mean-pool, fake=softmax[0]", thresholds=[0.5],
                card_eer_threshold_context_only=ad.CARD_EER_THRESHOLD_ITW_CONTEXT_ONLY),
                policy=dict(min_s=dh.MIN_DUR_S, max_s=dh.MAX_DUR_S), batch_size=1)
el.jdump({"settings": SETTINGS, "env": ENV, "probe": PROBE, "candidate_load_report": cand.report, "started_utc": utc},
         os.path.join(RUN_DIR, "run_config.json"))
print(RUN_DIR)
""")

code("""
# 7. In-the-Wild: open archive (HTTP range), parse meta.csv, build DEV (pilot 200) and HELD-OUT (fresh, clip-disjoint) manifests
src = el.SOURCES["in_the_wild"]
zf, raw = el.open_remote_zip(src["url"], src["bytes"])
names = zf.namelist(); meta_m = el.find_member(names, src["meta_suffix"]); prefix = meta_m[:-len("meta.csv")]
recs = el.parse_itw_meta(zf.read(meta_m).decode("utf-8"))
nameset = set(names)
for r in recs: r["archive_member"] = prefix + r["clip_id"] if prefix + r["clip_id"] in nameset else ""
desc = el.describe_records(recs); print("meta.csv:", desc)
assert desc["by_label"] == src["expected_counts"], desc["by_label"]

pilot_set = set(PILOT_IDS)
DEV = [dict(r, dataset="in_the_wild", split="dev_pilot") for r in recs if r["clip_id"] in pilot_set]
assert len(DEV) == 200
HELD, hmeta = el.select_subset(recs, "in_the_wild", HELDOUT_SEED, HELDOUT_N, n_synthetic=HELDOUT_N, exclude_clip_ids=pilot_set)
for r in HELD: r["split"] = "heldout"
assert not ({r["clip_id"] for r in HELD} & pilot_set)
hmeta["speaker_overlap_with_pilot"] = len({r["speaker_id"] for r in HELD} & {r["speaker_id"] for r in DEV})
hmeta["note"] = "clip-disjoint from pilot; speaker-disjoint impossible (pilot covered all 54 speakers)"
el.jdump(hmeta, os.path.join(RUN_DIR, "selection_heldout_in_the_wild.json"))
pd.DataFrame(HELD).to_csv(os.path.join(RUN_DIR, "manifest_heldout_in_the_wild.csv"), index=False)
pd.DataFrame(DEV).to_csv(os.path.join(RUN_DIR, "manifest_dev_pilot_in_the_wild.csv"), index=False)
print("held-out:", hmeta["n_selected"], hmeta["selected"]["by_label"], "| speakers:", hmeta["selected"]["n_speakers_by_label"],
      "| clip-ID sha256:", hmeta["clip_id_list_sha256"], "| shortfalls:", hmeta["shortfalls"] or "none")
print("speakers shared with pilot:", hmeta["speaker_overlap_with_pilot"], "(expected: most; see note)")
""")

code("""
# 8. Scoring helpers (both models on the same decoded, validated audio; batch size 1)
def fetch(m, split):
    local = os.path.join(WORK, "audio", split, os.path.basename(m["archive_member"]))
    if not os.path.exists(local):
        data = zf.read(m["archive_member"])          # CRC-32 checked by zipfile
        with open(local, "wb") as f: f.write(data)
    return local

def score_both(m, split, run_dhwani=True):
    row = {k: m.get(k, "") for k in ("dataset", "split", "clip_id", "label", "speaker_id", "selection_rank")}
    t0 = time.perf_counter()
    try:
        local = fetch(m, split)
        row["file_sha256"] = dh.sha256_file(local)
        y, sr = dh.load_audio(local); row.update(native_sr=sr, channels=int(y.shape[0]))
        x16 = dh.to_mono_16k(y, sr); x16, info = dh.validate(x16); row.update(info); row["status"] = "ok"
    except dh.AudioRejected as e:
        row.update(status="rejected", reject_reason=str(e), reject_category=el.reject_category(str(e)))
        row["latency_ms_total"] = round((time.perf_counter() - t0) * 1000, 1); return row
    except Exception as e:
        row.update(status="fetch_failed", reject_reason=f"{type(e).__name__}: {e}", reject_category="fetch_failed")
        row["latency_ms_total"] = round((time.perf_counter() - t0) * 1000, 1); return row
    # candidate (published recipe + secondary window variant)
    c = ad.score_clip(cand, x16, window=CAND_WINDOW, hop=CAND_HOP)
    row.update(cand_fake_score=c["fake_score_whole"], cand_logits_fake_real=json.dumps(c["logits_fake_real"]),
               cand_fake_score_window_mean=c["fake_score_window_mean"], cand_window_scores=json.dumps(c["window_fake_scores"]),
               cand_latency_ms=c["latency_ms_model"])
    if run_dhwani:
        r = dh.analyse(sess, local, mode="train_peak")   # re-decodes; identical validation path
        assert r.status == "ok", r.reject_reason
        row.update(dhwani_score=r.clip_score_mean, dhwani_score_max=r.clip_score_max, dhwani_window_scores=json.dumps(r.window_scores),
                   dhwani_window_logits=json.dumps(r.window_logits), dhwani_latency_ms=r.latency_ms_model)
    row["latency_ms_total"] = round((time.perf_counter() - t0) * 1000, 1)
    return row

def run_split(manifest, split, run_dhwani, path):
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["clip_id"] for l in open(path) if l.strip()}
    todo = [m for m in manifest if m["clip_id"] not in done]
    print(f"[{split}] {len(manifest)} clips, {len(done)} done, {len(todo)} to go")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as out:
        for i, m in enumerate(todo):
            out.write(json.dumps(score_both(m, split, run_dhwani), default=str) + "\\n"); out.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                e = time.time() - t0; print(f"  {i+1}/{len(todo)} {e/60:.1f} min, ETA {e/(i+1)*(len(todo)-i-1)/60:.1f} min")
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]).drop_duplicates("clip_id", keep="last")
""")

code("""
# 9. DEV-EVIDENCE GATE: candidate on the 200 pilot clips (Dhwani already scored there; not repeated).
#    AUC < MIN_PILOT_DEV_AUC -> stop and investigate. This does NOT establish that the mapping is wrong, and a high
#    AUC does NOT establish parity (cell 13 does that). Nothing is tuned or changed based on this value.
DEV_PATH = os.path.join(RUN_DIR, "predictions_dev_pilot.jsonl")
D = run_split(DEV, "dev", run_dhwani=False, path=DEV_PATH)
ok = D[D.status == "ok"]
y = (ok.label == "synthetic").astype(int).tolist()
dev_auc = el.roc_auc(y, ok.cand_fake_score.tolist()); dev_eer = el.eer(y, ok.cand_fake_score.tolist())[0]
print(f"candidate on dev pilot: n={len(ok)}  AUC {dev_auc:.4f}  EER {dev_eer:.4f}  (published full-ITW: AUC 0.993, EER 0.042)")
print("Dhwani on the same clips (run 20261024T123319Z, 155 scored): AUC 0.456, EER 0.561")
GATE = {"dev_n_scored": int(len(ok)), "candidate_dev_auc": dev_auc, "candidate_dev_eer": dev_eer, "min_auc": MIN_PILOT_DEV_AUC,
        "passed": dev_auc is not None and dev_auc >= MIN_PILOT_DEV_AUC,
        "meaning": "stop-and-investigate gate on development evidence; neither a proof of a mapping error nor of parity"}
el.jdump(GATE, os.path.join(RUN_DIR, "dev_evidence_gate.json"))
assert GATE["passed"], ("STOP: candidate AUC on development clips is below the gate. Investigate (run cell 13 parity first; check "
                        "decoding/normalisation) before any held-out scoring. Report the value as is.")
""")

code("""
# 10. HELD-OUT scoring: both models, same clips (resumable)
HO_PATH = os.path.join(RUN_DIR, "predictions_heldout.jsonl")
H = run_split(HELD, "heldout", run_dhwani=True, path=HO_PATH)
assert len(H) == len(HELD)
""")

code("""
# 11. Coverage (separate from accuracy) and metrics: threshold-free first, then fixed thresholds; paired bootstrap
cov = H.groupby("label").agg(selected=("clip_id", "size"), scored=("status", lambda s: int((s == "ok").sum())),
                             rejected=("status", lambda s: int((s == "rejected").sum())),
                             fetch_failed=("status", lambda s: int((s == "fetch_failed").sum())))
cov["coverage"] = (cov.scored / cov.selected).round(4)
print("HELD-OUT coverage:\\n", cov.to_string()); print("\\nrejections:", H[H.status != "ok"].reject_category.value_counts().to_dict() or "none")
OK = H[H.status == "ok"].copy(); yy = (OK.label == "synthetic").astype(int).tolist()
SC = {"dhwani_train_peak_mean4s": OK.dhwani_score.astype(float).tolist(),
      "candidate_whole_clip": OK.cand_fake_score.astype(float).tolist(),
      "candidate_window_mean4s": OK.cand_fake_score_window_mean.astype(float).tolist()}
THR = {"dhwani_train_peak_mean4s": [0.5], "candidate_whole_clip": [0.5], "candidate_window_mean4s": [0.5]}
METRICS = {}
for k, s in SC.items():
    m = el.evaluate(yy, s, THR[k][0], n_boot=1000, seed=HELDOUT_SEED)
    m["at_thresholds"] = {str(t): el.at_threshold(yy, s, t) for t in THR[k]}
    METRICS[k] = m
    tf, bs = m["threshold_free"], m["bootstrap95"]
    print(f"\\n== {k}  (n={len(yy)}; HELD-OUT PILOT SUBSET, clip-disjoint from dev, speakers shared)")
    print(f"   ROC-AUC {tf['roc_auc']:.4f} [{bs['roc_auc'][0]:.3f}, {bs['roc_auc'][1]:.3f}]   EER {tf['eer']:.4f} [{bs['eer'][0]:.3f}, {bs['eer'][1]:.3f}]")
    for t, a in m["at_thresholds"].items():
        c = a["confusion"]
        print(f"   @ {t}: TP {c['tp_synthetic_as_synthetic']} FN {c['fn_synthetic_as_genuine']} FP {c['fp_genuine_as_synthetic']} TN {c['tn_genuine_as_genuine']} | "
              f"bal.acc {a['balanced_accuracy']:.3f}  recall {a['synthetic_recall']:.3f}  FPR {a['genuine_fpr']:.3f}")
# paired bootstrap: AUC(candidate) - AUC(dhwani) on the same clips
import random
rng = random.Random(f"paired:{HELDOUT_SEED}"); pos = [i for i, v in enumerate(yy) if v == 1]; neg = [i for i, v in enumerate(yy) if v == 0]
diffs = []
for _ in range(1000):
    idx = [pos[rng.randrange(len(pos))] for _ in pos] + [neg[rng.randrange(len(neg))] for _ in neg]
    l = [yy[i] for i in idx]
    diffs.append(el.roc_auc(l, [SC["candidate_whole_clip"][i] for i in idx]) - el.roc_auc(l, [SC["dhwani_train_peak_mean4s"][i] for i in idx]))
diffs.sort(); PAIRED = {"auc_diff_point": METRICS["candidate_whole_clip"]["threshold_free"]["roc_auc"] - METRICS["dhwani_train_peak_mean4s"]["threshold_free"]["roc_auc"],
                        "auc_diff_95": [diffs[25], diffs[974]]}
print("\\nPaired AUC difference (candidate - Dhwani):", PAIRED)
print(f"Threshold 0.5 fixed in advance for both models; nothing was tuned here. The card's EER threshold {ad.CARD_EER_THRESHOLD_ITW_CONTEXT_ONLY} "
      "was found on In-the-Wild itself and is NOT applied (context only). EER thresholds printed by eval_lib are descriptive.")
""")

code("""
# 12. Deployment feasibility: ONNX export of the candidate + ORT parity (NOT a mobile build; evidence for the Android path)
ONNX_REPORT = {"attempted": EXPORT_ONNX}
if EXPORT_ONNX:
    class Wrapped(torch.nn.Module):
        def __init__(s, enc, head): super().__init__(); s.enc, s.head = enc, head
        def forward(s, wav): return s.head(s.enc(wav).last_hidden_state.mean(dim=1))
    w = Wrapped(cand.encoder, cand.head).eval().cpu()
    onnx_path = os.path.join(WORK, "antideepfake_small.onnx")
    try:
        dummy = torch.randn(1, 64000)
        torch.onnx.export(w, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"input": {1: "time"}})
        s2 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        diffs, lat = [], []
        for m in OK.head(8).itertuples():
            x16 = dh.to_mono_16k(*dh.load_audio(os.path.join(WORK, "audio", "heldout", os.path.basename(m.clip_id))))
            x16, _ = dh.validate(x16); x = ad.AntiDeepfakeHF.preprocess(x16)
            t0 = time.perf_counter(); zo = s2.run(None, {"input": x[None]})[0][0]; lat.append((time.perf_counter() - t0) * 1000)
            zt = w(torch.from_numpy(x)[None]).detach().numpy()[0]
            diffs.append(float(np.abs(ad.softmax(zo)[0] - ad.softmax(zt)[0])))
        ONNX_REPORT.update(ok=True, file_mb=round(os.path.getsize(onnx_path) / 1e6, 1), opset=17,
                           max_abs_fake_score_diff=max(diffs), ort_cpu_ms_median=float(np.median(lat)), n_clips=len(diffs),
                           note="fp32 ONNX, dynamic time axis, Colab CPU latency; int8/fp16 and on-device latency not measured")
    except Exception as e:
        ONNX_REPORT.update(ok=False, error=f"{type(e).__name__}: {str(e)[:500]}")
    cand.encoder.to(DEVICE); cand.head.to(DEVICE)
print(json.dumps(ONNX_REPORT, indent=1))
""")

code("""
# 13. PARITY with the authors' reference implementation: separate Python 3.8 venv + fairseq 0.12.2, identical inputs.
#     Outcome is recorded as verified (with max |diff|) or unverified (with the reason). Never silently skipped.
PARITY = {"status": "unverified", "attempted": TRY_FAIRSEQ_PARITY}
if TRY_FAIRSEQ_PARITY:
    VENV = "/content/fsenv"; PY = f"{VENV}/bin/python"
    t0 = time.time()
    try:
        steps = [["uv", "python", "install", "3.8"], ["uv", "venv", "--python", "3.8", VENV],
                 ["uv", "pip", "install", "--python", PY, "torch==2.4.1", "torchaudio==2.4.1", "--index-url", "https://download.pytorch.org/whl/cpu"],
                 ["uv", "pip", "install", "--python", PY, "fairseq==0.12.2", "safetensors==0.5.3", "numpy<1.25"]]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd[:3])} failed: {(r.stderr or r.stdout)[-600:]}")
        PARITY["env_setup_s"] = round(time.time() - t0, 1)
        # identical inputs: the exact arrays the transformers path scores (layer-normed, 2-10 s policy applied)
        sample = OK.head(PARITY_N_CLIPS)
        arrays, hf_logits = {}, {}
        for n, m in enumerate(sample.itertuples()):
            x16 = dh.to_mono_16k(*dh.load_audio(os.path.join(WORK, "audio", "heldout", os.path.basename(m.clip_id))))
            x16, _ = dh.validate(x16); x = ad.AntiDeepfakeHF.preprocess(x16)
            arrays[f"w{n:03d}"] = x; hf_logits[f"w{n:03d}"] = cand.logits(x)
        np.savez(os.path.join(WORK, "parity_inputs.npz"), **arrays)
        r = subprocess.run([PY, "fairseq_parity_ref.py", AD_PATH, os.path.join(WORK, "parity_inputs.npz"), os.path.join(WORK, "parity_ref.npz")],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError("reference run failed: " + (r.stderr or r.stdout)[-800:])
        ref = np.load(os.path.join(WORK, "parity_ref.npz"))
        dl = [float(np.max(np.abs(ref[k] - hf_logits[k]))) for k in arrays]
        dp = [float(abs(ad.softmax(ref[k])[0] - ad.softmax(hf_logits[k])[0])) for k in arrays]
        PARITY.update(status="verified" if max(dp) < 1e-3 else "MISMATCH", n_clips=len(dl), max_abs_logit_diff=max(dl),
                      max_abs_fake_score_diff=max(dp), reference_stdout=r.stdout.strip()[-300:], total_s=round(time.time() - t0, 1),
                      note="fairseq 0.12.2 (authors' dependency) on CPU vs transformers mapping on " + DEVICE)
    except Exception as e:
        PARITY.update(status="unverified", reason=f"{type(e).__name__}: {str(e)[:800]}", total_s=round(time.time() - t0, 1))
print(json.dumps(PARITY, indent=1))
if PARITY["status"] != "verified":
    print("\\nCONVERSION PARITY UNVERIFIED (or mismatched): report the candidate numbers with this caveat.")
""")

code("""
# 14. Save everything (no audio) and download
H.to_csv(os.path.join(RUN_DIR, "predictions_heldout.csv"), index=False)
D.to_csv(os.path.join(RUN_DIR, "predictions_dev_pilot_candidate.csv"), index=False)
cov.reset_index().to_csv(os.path.join(RUN_DIR, "coverage_heldout.csv"), index=False)
RUNTIME = {"candidate_ms_median": float(OK.cand_latency_ms.median()), "dhwani_ms_median": float(OK.dhwani_latency_ms.median()),
           "device_candidate": DEVICE, "provider_dhwani": sess.get_providers()[0]}
el.jdump({"run_name": RUN_NAME, "scope": "HELD-OUT PILOT SUBSET of In-the-Wild (clip-disjoint from dev pilot; speakers shared). Not a full-dataset benchmark.",
          "settings": SETTINGS, "env": ENV, "conversion_gate": GATE, "heldout_selection": hmeta, "coverage": cov.reset_index().to_dict("records"),
          "metrics": METRICS, "paired_auc_difference": PAIRED, "onnx": ONNX_REPORT, "conversion_parity": PARITY, "runtime": RUNTIME,
          "candidate_license": ad.LICENSE, "overlap_notes": "In-the-Wild not in candidate training table; VoxCeleb2 (celebrity speech) is -> speaker overlap possible. Dhwani overlap unknown.",
          "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}, os.path.join(RUN_DIR, "metrics.json"))
zip_path = shutil.make_archive(os.path.join(WORK, RUN_NAME), "zip", RUN_DIR)
print(sorted(os.listdir(RUN_DIR)), "\\n", zip_path, f"{os.path.getsize(zip_path)/1e6:.2f} MB")
try:
    from google.colab import files; files.download(zip_path)
except ImportError: pass
""")

md("""
## Report back
The run zip (no audio) and the executed notebook, into `evaluation/runs/<run_name>/`. Paste the outputs of cells 5, 7, 9, 11, 12 and 13.

## How to read the result
- Cell 9 is a stop-and-investigate gate on development evidence, not a result and not proof about the conversion either way.
- Cell 11 is the result: a held-out pilot subset of In-the-Wild (300 clips planned, minus < 2 s rejections). Threshold-free numbers first;
  0.5 fixed before scoring. The clips share speakers with the pilot (unavoidable), so generalisation to new speakers is untested.
- Cell 13 decides whether "conversion parity" is **verified** (max fake-score difference vs the authors' fairseq code) or **unverified**.
  Every candidate number must carry that status.
- Any further model, recipe or threshold decision taken after seeing this result again needs a fresh selection (new seed, exclude both the pilot and these clip IDs).
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, json.dumps(SHA))
