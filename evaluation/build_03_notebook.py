"""Build evaluation/03_generalisation_eval_colab.ipynb (evaluation only).

Embeds, via %%writefile, a verbatim snapshot of src/vcd/dhwani_infer.py (read
only, owned by the inference work) and evaluation/eval_lib.py, and records
their sha256 so the notebook can prove which code produced a run.
Standard library only. Run: python3 evaluation/build_03_notebook.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DH_SRC = (ROOT / "src/vcd/dhwani_infer.py").read_text()
EL_SRC = (HERE / "eval_lib.py").read_text()
# Hash of the text with trailing whitespace normalised (cell magics may drop the final newline).
DH_SHA = hashlib.sha256((DH_SRC.rstrip() + "\n").encode()).hexdigest()
EL_SHA = hashlib.sha256((EL_SRC.rstrip() + "\n").encode()).hexdigest()
OUT = HERE / "03_generalisation_eval_colab.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip().splitlines(keepends=True)})


md("""
# 03 — Generalisation evaluation of the third-party Dhwani checkpoint (pilot)

Scores the **third-party** ONNX model
[`ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model`](https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model)
(MIT, revision `ee01871`, sha256 `d1c232bf…`) on two public benchmarks. We did not train this model and nothing here trains,
tunes or selects anything.

| Dataset | Source (pinned) | Licence | Labels used |
|---|---|---|---|
| ASVspoof 2019 LA **evaluation** partition | Edinburgh DataShare `LA.zip`, DOI 10.7488/ds/2555, 7,640,952,520 B, MD5 `30c98f11…` | ODC-By 1.0 | CM protocol `ASVspoof2019.LA.cm.eval.trl.txt`: bonafide → genuine, spoof → synthetic; attack IDs A07–A19 |
| In-the-Wild | HF `mueller91/In-The-Wild` @ `eee168f`, `release_in_the_wild.zip`, 8,161,489,023 B, sha256 `46665f30…` | **Conflicting**: HF card CC-BY-SA-4.0, project page Apache-2.0 | `meta.csv`: bona-fide → genuine, spoof → synthetic; speaker = public figure |

**What this run is and is not**
- A **pilot subset** (≈ 100 genuine + ≈ 100 synthetic per dataset), selected from metadata only, with a fixed seed, *before* the model runs.
  Pilot numbers are **not** full-dataset benchmark results and must never be reported as such.
- Training-data overlap with Dhwani is **unknown** (see `evaluation/README.md`). Its training code reads an unpublished Kaggle folder.
  ASVspoof's "unseen attacks" (A07–A19 vs. its own train set) are **not** thereby unseen by Dhwani; In-the-Wild's YouTube-sourced bona fide audio
  could overlap with Dhwani's English real audio (14 unnamed YouTube recordings). Neither overlap can be ruled out.
- Fixed analysis plan: preprocessing `train_peak` (provisional primary recipe), our existing windowing (`dhwani_infer.py`, embedded verbatim),
  clip score = mean of window scores, threshold **0.5 (provisional, untuned; not optimised here)**, batch size **1**.
- Scores are uncalibrated softmax outputs, **not probabilities**. Failures are kept and reported; no clip is replaced or dropped silently.
- These clips must never be used for training, threshold tuning, model or preprocessing selection.

**Order:** run cells top to bottom. Runtime: CPU is enough for the pilot (≈ 15–30 min, mostly model time).
""")

code("""
# 1. Settings (fixed BEFORE any result is seen; changing them is a new, separately documented run)
RUN_KIND = "pilot"              # "pilot" | "full" (every clip of both evaluation partitions; see the end of the notebook)
DATASETS = ["asvspoof2019_la_eval", "in_the_wild"]
SELECTION_SEED = 20261025
ASV_N_PER_ATTACK = 8            # 8 x 13 attack IDs (A07-A19) = 104 synthetic (100 rounded up to equal strata)
ASV_N_GENUINE = 104             # matched to the synthetic count
ITW_N_GENUINE = 100
ITW_N_SYNTHETIC = 100
PREPROCESSING = "train_peak"    # provisional primary recipe; do not switch to whichever looks best
ACCESS = "range"                # "range": HTTP range reads of only the needed zip members | "download": whole archives
PROVIDER = "cpu"                # "cpu" (pilot; same as the baseline run) | "cuda" (full eval; CPU-parity-checked first)
PARITY_N = 20                   # with PROVIDER="cuda": first N clips also scored on CPU; abort if they disagree
N_BOOT = 1000                   # bootstrap resamples for AUC/EER/balanced-accuracy CIs (use ~200 for the full run)
USE_DRIVE = False               # True: model cache + run folder in Drive (survives runtime resets, enables resume)
DRIVE_DIR = "/content/drive/MyDrive/vcd_cache"
WORK = "/content/vcd_eval"      # local scratch: archives (download mode), extracted audio
KEEP_AUDIO = True               # pilot: keep extracted clips for listening; full run: False (delete after scoring)
RUN_NAME = None                 # set to an existing run folder name to RESUME it (selection is re-verified)
""")

code("""
# 2. Dependencies (Colab ships most; onnxruntime-gpu only if PROVIDER == "cuda")
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxruntime", "librosa", "soundfile", "huggingface_hub"], check=True)
if PROVIDER == "cuda":
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "onnxruntime"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxruntime-gpu"], check=True)
""")

code("%%writefile dhwani_infer.py\n" + DH_SRC)
code("%%writefile eval_lib.py\n" + EL_SRC)

code(f"""
# 3. Imports, code provenance, runtime report
import os, sys, json, time, hashlib, platform, datetime, importlib, traceback, zipfile, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import dhwani_infer as dh, eval_lib as el
importlib.reload(dh); importlib.reload(el)

EXPECTED_CODE_SHA256 = {{
    "dhwani_infer.py": "{DH_SHA}",   # snapshot of src/vcd/dhwani_infer.py at build time
    "eval_lib.py": "{EL_SHA}",
}}
CODE_SHA256 = {{f: hashlib.sha256((open(f, encoding="utf-8").read().rstrip() + "\\n").encode()).hexdigest()
               for f in EXPECTED_CODE_SHA256}}
assert CODE_SHA256 == EXPECTED_CODE_SHA256, f"embedded code was edited: {{CODE_SHA256}}"

assert PREPROCESSING == "train_peak", "train_peak is the pre-registered primary recipe for this evaluation"
assert dh.DECISION_THRESHOLD == 0.5, "threshold must stay at the provisional 0.5"
assert dh.LABELS == {{0: "genuine", 1: "synthetic"}} and dh.SYNTHETIC_INDEX == 1
THRESHOLD = dh.DECISION_THRESHOLD

import onnxruntime as ort, librosa, soundfile, sklearn
ENV = {{
    "python": platform.python_version(), "platform": platform.platform(),
    "onnxruntime": ort.__version__, "ort_available_providers": ort.get_available_providers(),
    "numpy": np.__version__, "pandas": pd.__version__, "librosa": librosa.__version__,
    "soundfile": soundfile.__version__, "libsndfile": soundfile.__libsndfile_version__,
    "sklearn": sklearn.__version__, "cpu_count": os.cpu_count(),
    "code_sha256": CODE_SHA256,
}}
try:
    import torch
    ENV["torch_cuda"] = torch.cuda.is_available()
    ENV["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except ImportError:
    pass
print(json.dumps(ENV, indent=2))
print("policy:", dict(SR=dh.SR, WIN=dh.WIN, HOP=dh.HOP, MIN_DUR_S=dh.MIN_DUR_S, MAX_DUR_S=dh.MAX_DUR_S,
                      PEAK_TARGET_DBFS=dh.PEAK_TARGET_DBFS, threshold=THRESHOLD, preprocessing=PREPROCESSING))
""")

code("""
# 4. Pinned checkpoint: reuse a cached copy if present, else download once; always verify sha256
from huggingface_hub import hf_hub_download
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)

cands = []
if USE_DRIVE:
    cands.append(os.path.join(DRIVE_DIR, dh.MODEL_FILE))
try:
    cands.append(hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_files_only=True))
except Exception:
    pass
MODEL_PATH = next((p for p in cands if os.path.exists(p) and os.path.getsize(p) == dh.MODEL_BYTES), None)
MODEL_SOURCE = "cache"
if MODEL_PATH is None:
    MODEL_SOURCE = "downloaded"
    print("No cached copy; downloading", dh.MODEL_REPO, "@", dh.MODEL_REVISION[:7], "(1.26 GB) once ...")
    MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION,
                                 **({"local_dir": DRIVE_DIR} if USE_DRIVE else {}))
assert os.path.getsize(MODEL_PATH) == dh.MODEL_BYTES, "unexpected model size"
t0 = time.time()
MODEL_SHA = dh.sha256_file(MODEL_PATH)
assert MODEL_SHA == dh.MODEL_SHA256, f"sha256 mismatch: {MODEL_SHA}"
print(f"{MODEL_SOURCE}: {MODEL_PATH}\\nsha256 verified ({time.time()-t0:.1f} s): {MODEL_SHA}")
""")

code("""
# 5. ONNX session(s) + interface probe (batch size 1 only: batch 2 silently drops a row)
if PROVIDER == "cpu":
    sess = dh.load_session(MODEL_PATH, verify_sha=False)        # sha checked above
    cpu_sess = None
elif PROVIDER == "cuda":
    sess = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", f"CUDA not active: {sess.get_providers()}"
    cpu_sess = dh.load_session(MODEL_PATH, verify_sha=False)    # parity reference for the first PARITY_N clips
else:
    raise ValueError(PROVIDER)
SESSION = dh.describe_session(sess)
PROBE = dh.interface_self_test(sess)
print(json.dumps({"session": SESSION, "probe": PROBE}, indent=2, default=str))
assert PROBE["len_64000_batch1"]["ok"]
print("batch2 ok?", PROBE["batch2"].get("ok"), "-> regardless, every window is run with batch size 1 (dhwani_infer._score)")
""")

code("""
# 6. Run folder (resumable)
BASE = os.path.join(DRIVE_DIR if USE_DRIVE else WORK, "runs")
utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_NAME = RUN_NAME or f"{utc}_{RUN_KIND}_{PREPROCESSING}"
RUN_DIR = os.path.join(BASE, RUN_NAME)
RESUMING = os.path.exists(os.path.join(RUN_DIR, "predictions.jsonl"))
os.makedirs(RUN_DIR, exist_ok=True)
for d in ("archives", "audio"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
SETTINGS = dict(RUN_KIND=RUN_KIND, DATASETS=DATASETS, SELECTION_SEED=SELECTION_SEED,
                ASV_N_PER_ATTACK=ASV_N_PER_ATTACK, ASV_N_GENUINE=ASV_N_GENUINE,
                ITW_N_GENUINE=ITW_N_GENUINE, ITW_N_SYNTHETIC=ITW_N_SYNTHETIC,
                PREPROCESSING=PREPROCESSING, THRESHOLD=THRESHOLD, ACCESS=ACCESS, PROVIDER=PROVIDER,
                PARITY_N=PARITY_N, N_BOOT=N_BOOT, KEEP_AUDIO=KEEP_AUDIO, clip_score="mean of window scores",
                batch_size=1)
if not RESUMING:
    el.jdump({"settings": SETTINGS, "env": ENV, "model": {"repo": dh.MODEL_REPO, "revision": dh.MODEL_REVISION,
              "sha256": MODEL_SHA, "path": MODEL_PATH, "source": MODEL_SOURCE},
              "session": SESSION, "probe": PROBE, "started_utc": utc}, os.path.join(RUN_DIR, "run_config.json"))
else:
    prev = json.load(open(os.path.join(RUN_DIR, "run_config.json")))["settings"]
    assert prev == SETTINGS, f"resume with different settings: {prev} vs {SETTINGS}"
print("run dir:", RUN_DIR, "| resuming" if RESUMING else "| new run")
""")

code("""
# 7. Dataset access: open archives, read labels/metadata, check counts. Blockers are recorded, not hidden.
import urllib.parse, urllib.request

def download_archive(ds, src):
    dest = os.path.join(WORK, "archives", src["archive"])
    free = shutil.disk_usage(WORK).free
    if not os.path.exists(dest):
        assert free > src["bytes"] * 1.3, f"not enough disk: {free/1e9:.1f} GB free"
        if ds == "in_the_wild":
            p = hf_hub_download(src["hf_repo"], src["archive"], repo_type="dataset", revision=src["revision"])
            os.symlink(os.path.realpath(p), dest)
        else:
            with urllib.request.urlopen(src["url"], timeout=120) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            os.rename(dest + ".part", dest)
    assert os.path.getsize(dest) == src["bytes"], "archive size mismatch"
    algo = "sha256" if "sha256" in src else "md5"
    t = time.time(); got = el.file_digest(dest, algo)
    assert got == src[algo], f"{algo} mismatch: {got}"
    return dest, {"access": "full download", "archive_path": dest, f"{algo}_verified": got,
                  "verify_s": round(time.time() - t, 1)}

ARCH, RECORDS, BLOCKERS, ACCESS_INFO, DESC = {}, {}, [], {}, {}
for ds in DATASETS:
    src = el.SOURCES[ds]
    t0 = time.time()
    try:
        if ACCESS == "range":
            zf, raw = el.open_remote_zip(src["url"], src["bytes"])
            etag = (raw.headers.get("etag") or "").strip('"')
            info = {"access": "http range", "host": urllib.parse.urlparse(raw._final_url).netloc,
                    "content_length_total": raw.size, "etag": etag,
                    "last_modified": raw.headers.get("last-modified"),
                    "etag_matches_recorded": etag in (src.get("md5"), src.get("xet_hash")),
                    "integrity_note": "range mode cannot hash the whole archive; size + server ETag are checked "
                                      "and zipfile verifies each extracted member's CRC-32"}
        else:
            path, info = download_archive(ds, src)
            zf, raw = zipfile.ZipFile(path), None
        names = zf.namelist()
        info["n_members"] = len(names)
        if ds == "asvspoof2019_la_eval":
            proto = el.find_member(names, src["protocol_suffix"])
            recs = el.parse_asvspoof_cm_protocol(zf.read(proto).decode("utf-8"))
            audio = {n.rsplit("/", 1)[1][: -len(src["audio_ext"])]: n for n in names
                     if src["audio_dir_suffix"] in n and n.endswith(src["audio_ext"])
                     and "__MACOSX" not in n and not n.rsplit("/", 1)[1].startswith("._")}
            info["metadata_member"] = proto
        else:
            meta = el.find_member(names, src["meta_suffix"])
            recs = el.parse_itw_meta(zf.read(meta).decode("utf-8"))
            prefix = meta[: -len("meta.csv")]
            nameset = set(names)
            audio = {r["clip_id"]: prefix + r["clip_id"] for r in recs if prefix + r["clip_id"] in nameset}
            info["metadata_member"] = meta
        missing = [r["clip_id"] for r in recs if r["clip_id"] not in audio]
        for r in recs:
            r["archive_member"] = audio.get(r["clip_id"], "")
        d = el.describe_records(recs)
        d["n_metadata_rows_without_audio"] = len(missing)
        d["expected_counts"] = src["expected_counts"]
        d["counts_match_expected"] = d["by_label"] == src["expected_counts"]
        if "expected_attacks" in src:
            d["attacks_match_expected"] = sorted(d["by_attack"]) == src["expected_attacks"]
        info["open_s"] = round(time.time() - t0, 1)
        if raw is not None:
            info["requests_so_far"], info["bytes_fetched_so_far"] = raw.n_requests, raw.bytes_fetched
        ARCH[ds], RECORDS[ds], ACCESS_INFO[ds], DESC[ds] = zf, recs, info, d
        print(f"== {ds}\\n", json.dumps({"access": info, "metadata": d}, indent=1))
        if not d["counts_match_expected"]:
            print(f"WARNING {ds}: label counts differ from literature values {src['expected_counts']}")
        if missing:
            print(f"WARNING {ds}: {len(missing)} metadata rows have no audio member (kept as fetch failures if selected)")
    except Exception as e:
        BLOCKERS.append({"dataset": ds, "stage": "access", "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-2000:]})
        print(f"BLOCKER {ds}: {type(e).__name__}: {e}")
el.jdump({"sources": {k: el.SOURCES[k] for k in DATASETS}, "access": ACCESS_INFO, "metadata": DESC,
          "blockers": BLOCKERS}, os.path.join(RUN_DIR, "sources_and_access.json"))
print("blockers:", BLOCKERS or "none")
""")

code("""
# 8. Score-free selection -> manifests written BEFORE any audio is scored
MANIFEST = []
for ds, recs in RECORDS.items():
    full = RUN_KIND == "full"
    if ds == "asvspoof2019_la_eval":
        sel, smeta = el.select_subset(recs, ds, SELECTION_SEED, ASV_N_GENUINE, n_per_attack=ASV_N_PER_ATTACK, full=full)
    else:
        sel, smeta = el.select_subset(recs, ds, SELECTION_SEED, ITW_N_GENUINE, n_synthetic=ITW_N_SYNTHETIC, full=full)
    smeta.update(source=el.SOURCES[ds], python=platform.python_version())
    sp = os.path.join(RUN_DIR, f"selection_{ds}.json")
    if os.path.exists(sp):   # resume: selection must be byte-for-byte the same list
        prev = json.load(open(sp))
        assert prev["clip_id_list_sha256"] == smeta["clip_id_list_sha256"], "selection changed on resume"
    else:
        el.jdump(smeta, sp)
        pd.DataFrame(sel).to_csv(os.path.join(RUN_DIR, f"manifest_{ds}.csv"), index=False)
    MANIFEST += sel
    print(f"== {ds}: {smeta['n_selected']} selected | {smeta['rule']}")
    print("   by label:", smeta["selected"]["by_label"], "| by attack:", smeta["selected"]["by_attack"] or "-",
          "| speakers:", smeta["selected"]["n_speakers_by_label"])
    print("   clip-ID list sha256:", smeta["clip_id_list_sha256"], "| shortfalls:", smeta["shortfalls"] or "none")
assert MANIFEST, f"nothing selected; blockers: {BLOCKERS}"
pd.DataFrame(MANIFEST)[["dataset", "selection_rank", "stratum", "clip_id", "label", "speaker_id", "attack_id"]].head(12)
""")

code("""
# 9. Fetch + score every selected clip (batch size 1, train_peak, mean of windows). Appends to predictions.jsonl (resumable).
PRED_PATH = os.path.join(RUN_DIR, "predictions.jsonl")
done = set()
if os.path.exists(PRED_PATH):
    for line in open(PRED_PATH, encoding="utf-8"):
        r = json.loads(line); done.add((r["dataset"], r["clip_id"]))
todo = [m for m in MANIFEST if (m["dataset"], m["clip_id"]) not in done]
print(f"{len(MANIFEST)} selected, {len(done)} already scored, {len(todo)} to go")

parity = []
t_start = time.time()
with open(PRED_PATH, "a", encoding="utf-8") as out:
    for i, m in enumerate(todo):
        ds = m["dataset"]
        row = {k: m[k] for k in ("dataset", "selection_rank", "stratum", "clip_id", "label", "original_label",
                                 "speaker_id", "attack_id", "archive_member")}
        local = os.path.join(WORK, "audio", ds, os.path.basename(m["archive_member"] or m["clip_id"]))
        tf = time.perf_counter()
        try:
            if not m["archive_member"]:
                raise KeyError("no audio member for this metadata row")
            data = ARCH[ds].read(m["archive_member"])          # zipfile verifies CRC-32
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "wb") as f:
                f.write(data)                                   # byte copy, no re-encoding
            row.update(fetch_status="ok", fetch_error="", archive_bytes=len(data))
            try:   # container duration, recorded even for clips the policy later rejects
                inf = soundfile.info(local)
                row.update(container_duration_s=round(inf.duration, 3), container_sr=inf.samplerate,
                           container_format=f"{inf.format}/{inf.subtype}")
            except Exception as e:
                row.update(container_duration_s=None, container_sr=None, container_format=f"unreadable: {type(e).__name__}")
        except Exception as e:
            row.update(fetch_status="failed", fetch_error=f"{type(e).__name__}: {e}", archive_bytes=None)
        row["fetch_ms"] = round((time.perf_counter() - tf) * 1000, 1)
        if row["fetch_status"] == "ok":
            r = dh.analyse(sess, local, mode=PREPROCESSING)
            row.update(dh.result_row(r, provider=sess.get_providers()[0]))
            if cpu_sess is not None and len(parity) < PARITY_N and r.status == "ok":
                rc = dh.analyse(cpu_sess, local, mode=PREPROCESSING)
                parity.append(abs(rc.clip_score_mean - r.clip_score_mean))
                if len(parity) == PARITY_N:
                    print("CUDA vs CPU max |clip score diff| over", PARITY_N, "clips:", max(parity))
                    assert max(parity) <= 2e-3, "CUDA scores disagree with CPU; use PROVIDER='cpu'"
            if not KEEP_AUDIO:
                os.remove(local)
        else:
            row.update(status="fetch_failed", reject_reason=row["fetch_error"])
        row["reject_category"] = (el.reject_category(row.get("reject_reason", "")) if row.get("status") == "rejected"
                                  else ("fetch_failed" if row.get("status") == "fetch_failed" else ""))
        out.write(json.dumps(row, default=str) + "\\n"); out.flush()
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el_s = time.time() - t_start
            print(f"{i+1}/{len(todo)}  {el_s/60:.1f} min elapsed, ETA {el_s/(i+1)*(len(todo)-i-1)/60:.1f} min")
if parity:
    el.jdump({"n": len(parity), "max_abs_diff": max(parity)}, os.path.join(RUN_DIR, "cuda_cpu_parity.json"))
""")

# Cell 10 lives in colab_cells/ so the same text can be pasted into a running Colab session.
code((HERE / "colab_cells/10_coverage_report.py").read_text())

code("""
# 11. Metrics by dataset (threshold-free vs. at the provisional 0.5) + per attack; sklearn cross-check
from sklearn.metrics import roc_auc_score, roc_curve
OK = P[P.status == "ok"].copy()
OK["y"] = (OK.label == "synthetic").astype(int)
OK["score"] = OK.clip_score_mean.astype(float)
METRICS, PER_ATTACK, XCHECK = {}, [], {}
for ds, g in OK.groupby("dataset"):
    y, s = g.y.tolist(), g.score.tolist()
    m = el.evaluate(y, s, THRESHOLD, n_boot=N_BOOT, seed=SELECTION_SEED)
    m["score_quantiles"] = g.groupby("label").score.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_dict("index")
    METRICS[ds] = m
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); fnr = 1 - tpr
        k = int(np.nanargmin(np.abs(fnr - fpr)))
        XCHECK[ds] = {"auc_ours": m["threshold_free"]["roc_auc"], "auc_sklearn": float(roc_auc_score(y, s)),
                      "eer_ours": m["threshold_free"]["eer"], "eer_sklearn_convention": float((fnr[k] + fpr[k]) / 2)}
        assert abs(XCHECK[ds]["auc_ours"] - XCHECK[ds]["auc_sklearn"]) < 1e-9, XCHECK[ds]
        if abs(XCHECK[ds]["eer_ours"] - XCHECK[ds]["eer_sklearn_convention"]) > 1e-6:
            print("NOTE: EER tie-break differs from sklearn sweep:", XCHECK[ds])
    gen = g[g.y == 0]
    for atk, ga in g[(g.y == 1) & (g.attack_id != "")].groupby("attack_id"):
        yy, ss = [1] * len(ga) + [0] * len(gen), ga.score.tolist() + gen.score.tolist()
        mt = el.evaluate(yy, ss, THRESHOLD, n_boot=0)
        PER_ATTACK.append({"dataset": ds, "attack_id": atk, "n_synthetic_scored": len(ga), "n_genuine_scored_pooled": len(gen),
                           "roc_auc_vs_all_genuine": mt["threshold_free"]["roc_auc"], "eer_vs_all_genuine": mt["threshold_free"]["eer"],
                           "synthetic_recall@0.5": mt["at_threshold"]["synthetic_recall"],
                           "recall_wilson95": mt["at_threshold"]["synthetic_recall_wilson95"],
                           "median_score": float(ga.score.median())})
PER_ATTACK = pd.DataFrame(PER_ATTACK)

def fmt(x, nd=4):
    if isinstance(x, (tuple, list)) and None in x:
        return "n/a"
    return "n/a" if x is None else (f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]" if isinstance(x, (tuple, list)) else f"{x:.{nd}f}")
for ds, m in METRICS.items():
    tf, at, bs = m["threshold_free"], m["at_threshold"], m["bootstrap95"]
    c = at["confusion"]
    print(f"===== {ds}  [{RUN_KIND.upper()} SUBSET, n_scored={m['n_scored']}; not a full-dataset benchmark]" if RUN_KIND == "pilot"
          else f"===== {ds}  [FULL evaluation partition, n_scored={m['n_scored']}]")
    print(f"Threshold-free: ROC-AUC {fmt(tf['roc_auc'])} (95% boot {fmt(bs.get('roc_auc'))}) | "
          f"EER {fmt(tf['eer'])} (95% boot {fmt(bs.get('eer'))}); EER threshold {fmt(tf['eer_threshold_descriptive_only'])} is descriptive only")
    print(f"At threshold {THRESHOLD} (provisional, untuned):")
    print(f"   confusion  pred_synthetic  pred_genuine\\n   synthetic  {c['tp_synthetic_as_synthetic']:>14}  {c['fn_synthetic_as_genuine']:>12}"
          f"\\n   genuine    {c['fp_genuine_as_synthetic']:>14}  {c['tn_genuine_as_genuine']:>12}")
    print(f"   balanced accuracy {fmt(at['balanced_accuracy'])} (95% boot {fmt(bs.get('balanced_accuracy@thr'))})")
    print(f"   synthetic recall  {fmt(at['synthetic_recall'])} (Wilson {fmt(at['synthetic_recall_wilson95'])})")
    print(f"   genuine FPR       {fmt(at['genuine_fpr'])} (Wilson {fmt(at['genuine_fpr_wilson95'])})")
    print("   CIs: clip-level resampling only; speaker clustering ignored -> optimistic.\\n")
if len(PER_ATTACK):
    print("Per attack (ASVspoof; each attack vs ALL scored genuine pilot clips; n per attack is tiny):")
    print(PER_ATTACK.round(4).to_string(index=False))
print("\\nsklearn cross-check:", json.dumps(XCHECK, indent=1))
""")

code("""
# 12. Runtime
RUNTIME = {}
for ds, g in P.groupby("dataset"):
    ok = g[g.status == "ok"]
    win = ok.latency_ms_model / ok.n_windows.clip(lower=1)
    RUNTIME[ds] = {
        "clips_scored": int(len(ok)), "windows_scored": int(ok.n_windows.sum()),
        "model_ms_per_clip_median": float(ok.latency_ms_model.median()) if len(ok) else None,
        "model_ms_per_window_median": float(win.median()) if len(ok) else None,
        "total_ms_per_clip_median": float(ok.latency_ms_total.median()) if len(ok) else None,
        "fetch_ms_per_clip_median": float(g.fetch_ms.median()),
        "sum_total_min": round(float(g.latency_ms_total.fillna(0).sum() + g.fetch_ms.sum()) / 60000, 2),
        "provider": sess.get_providers()[0],
    }
    exp_n = sum(el.SOURCES[ds]["expected_counts"].values())
    if len(ok):
        RUNTIME[ds]["rough_full_partition_hours_same_hardware"] = round(
            exp_n * float((ok.latency_ms_total.mean() + g.fetch_ms.mean())) / 3.6e6, 1)
print(json.dumps(RUNTIME, indent=1))
""")

code("""
# 13. Save everything (+ zip for download). Per-file predictions include every selected clip, failures too.
P.to_csv(os.path.join(RUN_DIR, "predictions.csv"), index=False)
COV.to_csv(os.path.join(RUN_DIR, "coverage_by_dataset_label.csv"), index=False)
COV_ATTACK.to_csv(os.path.join(RUN_DIR, "coverage_by_attack.csv"), index=False)
EXCLUDED.to_csv(os.path.join(RUN_DIR, "excluded_clips.csv"), index=False)
PER_ATTACK.to_csv(os.path.join(RUN_DIR, "metrics_by_attack.csv"), index=False)
SUMMARY = {
    "run_name": RUN_NAME, "run_kind": RUN_KIND,
    "scope_warning": ("PILOT SUBSET ONLY - not a full-dataset benchmark result" if RUN_KIND == "pilot"
                      else "full evaluation partitions"),
    "overlap_with_dhwani_training": "unknown (training data not published; see evaluation/README.md)",
    "datasets_tested": TESTED, "datasets_NOT_tested": NOT_TESTED, "dataset_status": DATASET_STATUS,
    "settings": SETTINGS, "env": ENV, "model": {"repo": dh.MODEL_REPO, "revision": dh.MODEL_REVISION, "sha256": MODEL_SHA},
    "access": ACCESS_INFO, "blockers": BLOCKERS, "metadata": DESC,
    "coverage": COV.to_dict("records"), "rejections": REJ.to_dict("records"),
    "metrics_by_dataset": METRICS, "sklearn_crosscheck": XCHECK, "runtime": RUNTIME,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
}
el.jdump(SUMMARY, os.path.join(RUN_DIR, "metrics.json"))
zip_path = shutil.make_archive(os.path.join(WORK, RUN_NAME), "zip", RUN_DIR)
print("saved:", sorted(os.listdir(RUN_DIR)), "\\nzip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.2f} MB, no audio inside)")
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    pass
""")

md("""
## Report back
1. The downloaded `<run>.zip` (predictions, manifests, selection JSONs, coverage, metrics, access/provenance). It contains **no audio**.
2. The executed notebook (*File → Download → .ipynb*).
3. Any blocker printed by step 7, verbatim.

## Path to the full evaluation (after the pilot)
The pilot answers "does the pipeline work and roughly where does the model sit"; it is **not** a benchmark. For the full run:
1. Keep every setting except: `RUN_KIND = "full"`, `ACCESS = "download"` (verifies the whole-archive MD5 / sha256),
   `KEEP_AUDIO = False`, `N_BOOT = 200`, `USE_DRIVE = True` (the run folder then survives resets; set `RUN_NAME` to resume).
2. Full scope: ASVspoof 2019 LA eval = every protocol row (literature: 7,355 bona fide + 63,882 spoof); In-the-Wild = every `meta.csv` row
   (literature: 19,963 + 11,816). Step 12 prints a CPU time projection from the pilot; at ≈ 2 s per clip it is **tens of hours**, so
   use `PROVIDER = "cuda"` on a GPU runtime (the first `PARITY_N` clips are also scored on CPU and the run aborts if they disagree by > 0.002).
   Resume across sessions with the same `RUN_NAME`.
3. Disk: ≈ 16 GB of archives on local `/content` (not Drive) plus one extracted clip at a time.
4. Report the full run as a new, separate result; never merge pilot and full numbers. Keep `train_peak` and 0.5 unless a decision
   made on **separate validation data** (not these clips) is recorded in `docs/PROGRESS.md` first.
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, "| dhwani_infer.py sha256", DH_SHA, "| eval_lib.py sha256", EL_SHA)
