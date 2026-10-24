"""Build notebooks/build/01_dhwani_baseline_inference.ipynb (a fresh, unexecuted copy).

The inference logic lives in src/vcd/dhwani_infer.py and is embedded into the
notebook via %%writefile, so Colab needs only the single .ipynb upload.
Run: python3 scripts/build_inference_notebook.py
"""

from pathlib import Path

import json

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "src/vcd/dhwani_infer.py").read_text()
# Never overwrite the executed notebook in notebooks/; generated copies go to notebooks/build/.
OUT = ROOT / "notebooks/build/01_dhwani_baseline_inference.ipynb"
MLAAD_CELL = (ROOT / "src/notebook_cells/mlaad_fetch_5b.py").read_text()

cells = []


def _src(text):
    lines = text.splitlines(keepends=True)
    return lines


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text.strip())})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(text.strip())})

md("""
# 01 — Dhwani baseline inference (third-party checkpoint)

Smoke-tests the **third-party** ONNX model
[`ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model`](https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model)
(MIT, pinned revision `ee01871`) on recordings you upload. We did not train this model.

**Read before interpreting output**
- Scores are softmax outputs of an *uncalibrated* classifier — **model scores, not probabilities**.
- The 0.5 threshold is a placeholder, not tuned on any validation set.
- Upstream code disagrees about preprocessing (evidence: `docs/UPSTREAM_AUDIT.md`). We do **not** pick one silently;
  every clip is scored under three controlled modes (all 16 kHz mono, same model, same clip):

  | Mode | Normalisation | Crop / windows | Source |
  |---|---|---|---|
  | A `train_peak` | peak → −3 dBFS | 4 s windows, 2 s hop | training notebook cell 4 (the run that exported this ONNX) |
  | B `train_zscore` | z-score | identical to A | normalisation used by the API / notebook inference cell |
  | C `api_replica` | z-score | first 3 s only, zero-padded to 4 s | deployed API `src/api/inference.py` |

  A vs B isolates normalisation; B vs C isolates crop/padding. The mode whose scores separate known genuine
  from known synthetic clips is *evidence* for the matching recipe — with 2 clips it is a hint, not proof.
- Label order (0 = genuine, 1 = synthetic) comes from upstream source and must be **confirmed on clips with known labels**.
- A handful of clips is a sanity check, not an accuracy estimate.

**Windowing policy (2–10 s clips)**
- < 2.0 s → rejected. > 10.0 s → only the first 10 s analysed (flagged `truncated`).
- ≤ 4 s → one window, zero-padded at the end (as in training).
- > 4 s → 4 s windows with 2 s hop; final window aligned to the clip end so every sample is covered.
- Clip score = mean of window scores (max also reported).
- Empty, unreadable, NaN, silent (peak < −50 dBFS) or mostly silent (< 1 s active signal) inputs are rejected — never replaced with zeros.

Run cells top to bottom. Runtime type: CPU is enough (ONNX Runtime uses `CPUExecutionProvider` here even on a GPU runtime).
""")

code("""
# 1. Dependencies (Colab ships most of these; this is a no-op if already present)
!pip -q install onnxruntime librosa soundfile huggingface_hub
""")

code("""
# 2. Settings
USE_DRIVE_CACHE = False   # True: keep the 1.26 GB model in Google Drive so new runtimes skip the download
DRIVE_CACHE_DIR = "/content/drive/MyDrive/vcd_cache"
MODES = ["train_peak", "train_zscore", "api_replica"]   # controlled comparison; see table above
UPLOAD_DIR = "/content/vcd_uploads"
RESULTS_DIR = "/content/vcd_results"
""")

code("%%writefile dhwani_infer.py\n" + MODULE)

code("""
# 3. Import helper module + runtime report
import importlib, sys, platform, os, json
import numpy as np
sys.path.insert(0, os.getcwd())
import dhwani_infer as dh
importlib.reload(dh)

import onnxruntime as ort, librosa, soundfile
print("python", platform.python_version(), "| onnxruntime", ort.__version__,
      "| librosa", librosa.__version__, "| soundfile", soundfile.__version__)
try:
    import torch
    print("PyTorch CUDA available:", torch.cuda.is_available(),
          "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:
    print("PyTorch not installed")
print("ONNX Runtime providers installed:", ort.get_available_providers())
print("NOTE: PyTorch CUDA availability does not mean ONNX Runtime uses the GPU. This notebook uses CPUExecutionProvider.")
""")

code("""
# 4. Get the REAL pinned Dhwani checkpoint (downloads only if not already cached), verify sha256
from huggingface_hub import hf_hub_download

if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(DRIVE_CACHE_DIR, dh.MODEL_FILE)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) == dh.MODEL_BYTES:
        print("Using Drive-cached model:", MODEL_PATH)
    else:
        print("Downloading", dh.MODEL_REPO, "@", dh.MODEL_REVISION[:7], "(1.26 GB) to Drive ...")
        MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_dir=DRIVE_CACHE_DIR)
else:
    # Default HF cache: reuses a copy already downloaded earlier in this runtime
    # (e.g. by "AI Voice Detection.ipynb") when the blob matches the pinned revision.
    MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION)
    print("Model path:", MODEL_PATH)
assert os.path.getsize(MODEL_PATH) == dh.MODEL_BYTES, "unexpected model size"
marker = os.path.join(os.path.dirname(os.path.realpath(MODEL_PATH)), ".dhwani_sha256_ok")

stamp = f"{os.path.getsize(MODEL_PATH)}:{os.path.getmtime(MODEL_PATH)}"
if os.path.exists(marker) and open(marker).read() == stamp:
    print("sha256 previously verified for this file")
else:
    got = dh.sha256_file(MODEL_PATH)
    assert got == dh.MODEL_SHA256, f"sha256 mismatch: {got}"
    open(marker, "w").write(stamp)
    print("sha256 verified:", got)
""")

code("""
# 5. Load session, print interface, probe what it really accepts
sess = dh.load_session(MODEL_PATH, verify_sha=False)  # sha checked in the previous cell
desc = dh.describe_session(sess)
for k, v in desc.items():
    print(f"{k}: {v}")
probe = dh.interface_self_test(sess)
print(json.dumps(probe, indent=2))
""")

md("""
## Optional: fetch one labelled English synthetic clip from MLAAD (gated)

MLAAD ([mueller91/MLAAD](https://huggingface.co/datasets/mueller91/MLAAD), CC-BY-NC-4.0, non-commercial research only)
requires a Hugging Face account that has **accepted the dataset terms** on its page. Then add your token as a
Colab secret named `HF_TOKEN` (key icon in the left sidebar → enable notebook access). Never paste the token in chat or code.

Selection is **pre-registered** and never uses detector output:
- generator `Chatterbox`: zero-shot voice cloning, and **not** one of the generators in Dhwani's English training data
  (garystafford/deepfake-audio-detection: ElevenLabs, Kokoro, Hume, Luvvoice, Amazon Polly, Speechify).
- the candidate list is the generator's `meta.csv` rows, sorted by path and shuffled with seed `20261024`; the first clip whose published duration is 5–10 s is taken
  (run 20261024T102407Z selected `fake/en/Chatterbox/emerald_city_of_oz_20_f000055.wav`, 7.44 s).
- Only `meta.csv` and that one clip are downloaded (if `meta.csv` lacked durations, up to 10 trial clips). The metadata row is saved alongside.
""")

# Verbatim copy of the 5b cell that ran successfully in Colab (run 20261024T102407Z).
code(MLAAD_CELL)

md("""
## Upload recordings

Name files so the ground truth is recorded automatically (anything after the prefix is free text):

- `genuine__<lang>__<note>.wav` — e.g. `genuine__hi__aman_phone_voicenote.m4a`
- `synthetic__<lang>__<generator>__<note>.wav` — e.g. `synthetic__en__elevenlabs__clone_of_harsh.mp3`

Files without a prefix are recorded as `unknown`. You can also fix labels in the `OVERRIDES` dict below.
Only upload audio you have the right to use; don't upload other people's voices without consent.
""")

code("""
# 6. Upload (Colab file picker). Re-running adds more files.
import shutil, glob
os.makedirs(UPLOAD_DIR, exist_ok=True)
try:
    from google.colab import files
    up = files.upload()
    for name in up:
        shutil.move(name, os.path.join(UPLOAD_DIR, os.path.basename(name)))
except ImportError:
    print("Not in Colab: put files in", UPLOAD_DIR)
paths = sorted(p for p in glob.glob(os.path.join(UPLOAD_DIR, "*")) if os.path.isfile(p))
print(len(paths), "file(s):", [os.path.basename(p) for p in paths])
""")

code("""
# 7. Metadata from filenames (+ optional manual overrides)
import pandas as pd

OVERRIDES = {
    # "some_file.wav": {"label": "genuine", "language": "hi", "generator": "", "note": ""},
}

def parse_name(fn):
    parts = os.path.splitext(fn)[0].split("__")
    meta = {"label": "unknown", "language": "", "generator": "", "note": ""}
    if parts[0].lower() in ("genuine", "synthetic"):
        meta["label"] = parts[0].lower()
        if len(parts) > 1: meta["language"] = parts[1]
        if meta["label"] == "synthetic" and len(parts) > 2: meta["generator"] = parts[2]
        meta["note"] = "__".join(parts[3 if meta["label"] == "synthetic" else 2:])
    side = os.path.join("/content/vcd_meta", fn + ".json")
    if os.path.exists(side):  # provenance saved by step 5b
        meta["source_meta"] = open(side, encoding="utf-8").read()
    meta.update(OVERRIDES.get(fn, {}))
    return meta

meta = {os.path.basename(p): parse_name(os.path.basename(p)) for p in paths}
pd.DataFrame.from_dict(meta, orient="index").drop(columns="source_meta", errors="ignore")
""")

code("""
# 8. Inspect: validation report + playback (before any model scoring)
from IPython.display import Audio, display

for p in paths:
    fn = os.path.basename(p)
    print(f"--- {fn}  [{meta[fn]['label']}]")
    try:
        y, sr = dh.load_audio(p)
        x16 = dh.to_mono_16k(y, sr)
        print(f"native sr={sr} Hz, channels={y.shape[0]}, duration={y.shape[1]/sr:.2f} s", dh.signal_stats(x16))
        try:
            dh.validate(x16); print("validation: OK")
        except dh.AudioRejected as e:
            print("validation: REJECTED ->", e)
        display(Audio(x16, rate=dh.SR))  # playback is loudness-normalised by the browser widget
    except dh.AudioRejected as e:
        print("REJECTED ->", e)
""")

code("""
# 9. Score every file with every preprocessing mode
rows = []
for p in paths:
    fn = os.path.basename(p)
    for mode in MODES:
        r = dh.analyse(sess, p, mode=mode)
        rows.append(dh.result_row(r, file_name=fn, ground_truth=meta[fn]["label"],
                                  language=meta[fn]["language"], generator=meta[fn]["generator"],
                                  note=meta[fn]["note"], source_meta=meta[fn].get("source_meta", ""),
                                  provider=sess.get_providers()[0]))
df = pd.DataFrame(rows)
cols = ["file_name", "ground_truth", "preprocessing", "status", "reject_reason", "duration_s",
        "n_win", "window_scores", "clip_score_mean", "clip_score_max", "predicted_label",
        "latency_ms_model", "latency_ms_total"]
df["n_win"] = df["window_scores"].map(lambda s: len(json.loads(s)))
pd.set_option("display.max_colwidth", 80)
df[[c for c in cols if c in df]]
""")

code("""
# 10. Controlled preprocessing comparison + sanity summary (NOT an accuracy estimate)
ok = df[df.status == "ok"]
print("synthetic-class score = softmax(logits)[1]; uncalibrated; threshold 0.5 is a placeholder\\n")
for mode, g in ok.groupby("preprocessing"):
    print(f"[{mode}]")
    by = g.groupby("ground_truth")["clip_score_mean"].agg(["count", "mean", "min", "max"])
    print(by.round(4).to_string())
    if {"genuine", "synthetic"} <= set(by.index):
        gm, sm = by.loc["genuine", "mean"], by.loc["synthetic", "mean"]
        if sm > gm:
            print("-> synthetic clips score higher than genuine on average: consistent with mapping 1 = synthetic")
        else:
            print("-> WARNING: genuine clips score >= synthetic. Mapping may be inverted, preprocessing wrong, or the model fails on these clips.")
    labelled = g[g.ground_truth.isin(["genuine", "synthetic"])]
    if len(labelled):
        agree = (labelled.predicted_label == labelled.ground_truth).sum()
        print(f"-> {agree}/{len(labelled)} labelled clips match at the placeholder threshold (tiny sample)")
    print()
print("Per-file clip_score_mean by mode (A vs B: normalisation; B vs C: crop):")
print(ok.pivot_table(index=["file_name", "ground_truth"], columns="preprocessing", values="clip_score_mean").round(4).to_string())
print()
print("rejected:", df[df.status != "ok"][["file_name", "preprocessing", "reject_reason"]].drop_duplicates("file_name").to_string(index=False) or "none")
""")

code("""
# 11. Save + download CSV (also copied to Drive if enabled)
import datetime
os.makedirs(RESULTS_DIR, exist_ok=True)
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
csv_path = os.path.join(RESULTS_DIR, f"dhwani_predictions_{ts}.csv")
df.assign(run_utc=ts, onnxruntime=ort.__version__).to_csv(csv_path, index=False)
with open(os.path.join(RESULTS_DIR, f"dhwani_interface_{ts}.json"), "w") as f:
    json.dump({"session": desc, "probe": probe}, f, indent=2, default=str)
print("saved", csv_path)
if USE_DRIVE_CACHE:
    shutil.copy(csv_path, DRIVE_CACHE_DIR)
try:
    from google.colab import files
    files.download(csv_path)
except ImportError:
    pass
""")

md("""
## Report back
Paste into the chat: output of cells 3, 5 and 10, and attach the CSV. Mention how each clip was made
(device, app, generator, whether it went through WhatsApp/phone).
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT)
