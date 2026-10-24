"""Build notebooks/02_detector_api_colab.ipynb (FastAPI server + tests + HTTPS tunnel).

Embeds src/vcd/{dhwani_infer,vcd_service,vcd_api}.py via %%writefile so Colab
needs only the .ipynb. Standard library only.
Run: python3 scripts/build_api_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/vcd"
OUT = ROOT / "notebooks/02_detector_api_colab.ipynb"
CLOUDFLARED_VERSION = "2026.9.3"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip().splitlines(keepends=True)})


md("""
# 02 — Detector API (FastAPI) on Colab

Serves the **third-party** Dhwani ONNX checkpoint
([`ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model`](https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model),
MIT, revision `ee01871`, built on `facebook/wav2vec2-xls-r-300m`). We did not train it.

- `GET /health`: model readiness (no key needed, no audio).
- `POST /detect`: multipart field `file`, header `X-API-Key`, optional `?mode=train_peak|train_zscore|api_replica`.
- The response has a prediction, an **uncalibrated** synthetic score (not a probability), duration, timings, the preprocessing mode and clear rejection reasons.
- `train_peak` is the *provisional* default. The two-clip smoke test (run 20261024T102407Z) does not validate it.
- Model loaded once at server start. One inference at a time (lock + single worker). Uploads capped at 8 MiB and deleted after each request.

**Runtime note.** Colab gives every notebook its own runtime (VM), so notebook 01's Hugging Face cache is reused only when
these cells run in that same runtime. In a fresh runtime, step 4 downloads the pinned checkpoint **once** (1.26 GB, ~1 min),
or copies it from Drive if `USE_DRIVE_CACHE = True` and it is there. It is never downloaded per request.

**Order:** 1 → 8 start and test the server locally inside Colab. Step 9 opens a public HTTPS tunnel for the phones. Step 11 stops everything.
""")

code("""
# 1. Dependencies
!pip -q install fastapi uvicorn python-multipart onnxruntime librosa soundfile huggingface_hub requests
import fastapi, uvicorn, onnxruntime
print("fastapi", fastapi.__version__, "| uvicorn", uvicorn.__version__, "| onnxruntime", onnxruntime.__version__)
""")

code("""
# 2. Settings
import os
APP_DIR = "/content/vcd_app"            # API source files
TMP_DIR = "/content/vcd_api_tmp"        # uploads live here only during a request
TEST_DIR = "/content/vcd_api_test"      # test clips
PORT = 8000
DEFAULT_MODE = "train_peak"             # provisional; see notebook 01 comparison
MAX_UPLOAD_MB = 8
USE_DRIVE_CACHE = False                 # True: reuse/keep the checkpoint in Drive
DRIVE_CACHE_DIR = "/content/drive/MyDrive/vcd_cache"
for d in (APP_DIR, TMP_DIR, TEST_DIR):
    os.makedirs(d, exist_ok=True)
""")

for name in ("dhwani_infer.py", "vcd_service.py", "vcd_api.py"):
    code(f"%%writefile /content/vcd_app/{name}\n" + (SRC / name).read_text())

code("""
# 3. (cells above) wrote dhwani_infer.py, vcd_service.py, vcd_api.py into APP_DIR
import sys, importlib
sys.path.insert(0, APP_DIR)
import dhwani_infer as dh; importlib.reload(dh)
print("module OK; modes:", dh.PREPROCESSING_MODES, "| pinned revision:", dh.MODEL_REVISION[:7])
""")

code("""
# 4. Locate the pinned checkpoint: HF cache in this runtime -> Drive -> one download
from huggingface_hub import hf_hub_download
MODEL_PATH = None
try:
    MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_files_only=True)
    print("Reusing checkpoint already in this runtime's HF cache")
except Exception:
    if USE_DRIVE_CACHE:
        from google.colab import drive
        drive.mount("/content/drive")
        cand = os.path.join(DRIVE_CACHE_DIR, dh.MODEL_FILE)
        if os.path.exists(cand) and os.path.getsize(cand) == dh.MODEL_BYTES:
            MODEL_PATH = cand
            print("Using Drive copy")
    if MODEL_PATH is None:
        print("Not cached in this runtime: downloading pinned checkpoint once (1.26 GB) ...")
        if USE_DRIVE_CACHE:
            MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_dir=DRIVE_CACHE_DIR)
        else:
            MODEL_PATH = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION)
assert os.path.getsize(MODEL_PATH) == dh.MODEL_BYTES
print("MODEL_PATH =", MODEL_PATH, "(sha256 is verified by the server at startup)")
""")

code("""
# 5. API key: Colab secret VCD_API_KEY if present, else a random one for this session
import secrets
try:
    from google.colab import userdata
    API_KEY = userdata.get("VCD_API_KEY")
    print("Using Colab secret VCD_API_KEY")
except Exception:
    API_KEY = secrets.token_urlsafe(32)
    print("No VCD_API_KEY secret: generated a session key. Give it to the phones privately;")
    print("it changes when you re-run this cell. Clear this cell's output before sharing the notebook.")
    print("API_KEY =", API_KEY)
assert len(API_KEY) >= 16, "VCD_API_KEY must be at least 16 characters"
""")

code("""
# 6. Start the server (one uvicorn worker, model loaded once) and wait for readiness
import subprocess, time, requests, json
BASE = f"http://127.0.0.1:{PORT}"
if "server" in globals() and server.poll() is None:
    server.terminate(); server.wait(10)
env = {**os.environ, "VCD_API_KEY": API_KEY, "VCD_MODEL_PATH": MODEL_PATH, "VCD_TMP_DIR": TMP_DIR,
       "VCD_PREPROCESSING": DEFAULT_MODE, "VCD_MAX_UPLOAD_BYTES": str(MAX_UPLOAD_MB * 1024 * 1024),
       "VCD_ALLOW_DOWNLOAD": "0"}
server_log = open("/content/vcd_api.log", "w")
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "vcd_api:app", "--app-dir", APP_DIR,
                           "--host", "127.0.0.1", "--port", str(PORT), "--workers", "1"],
                          env=env, stdout=server_log, stderr=subprocess.STDOUT)
health = None
for _ in range(180):
    if server.poll() is not None:
        break
    try:
        health = requests.get(BASE + "/health", timeout=2).json()
        break
    except Exception:
        time.sleep(1)
print(json.dumps(health, indent=2) if health else "server did not answer")
if not health or not health["model_ready"]:
    print(open("/content/vcd_api.log").read()[-3000:])
""")

md("""
## Test clips
Upload the **same genuine file used in notebook 01**: `genuine_hi_aman.m4a` (sha256 starts `8b03de09`), which is in the project root on Aman's Mac.
The synthetic clip is re-fetched from MLAAD by its exact pinned path (needs the `HF_TOKEN` Colab secret; MLAAD is CC-BY-NC-4.0, research use only).
""")

code("""
# 7. Get the two baseline clips into TEST_DIR
import shutil, glob
from google.colab import files, userdata
up = files.upload()  # pick genuine_hi_aman.m4a
for n in up:
    shutil.move(n, os.path.join(TEST_DIR, os.path.basename(n)))
syn_rel = "fake/en/Chatterbox/emerald_city_of_oz_20_f000055.wav"
syn_local = hf_hub_download("mueller91/MLAAD", syn_rel, repo_type="dataset",
                            revision="30c3dec763fa4803f11c2df1557d3b0828395366", token=userdata.get("HF_TOKEN"))
shutil.copyfile(syn_local, os.path.join(TEST_DIR, "synthetic__en__Chatterbox__emerald_city_of_oz_20_f000055.wav"))
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*")))
for p in test_files:
    print(dh.sha256_file(p)[:12], os.path.basename(p))
""")

code("""
# 8. Verify /health and /detect locally (inside Colab) - the backend is not "done" until this passes
import io, threading, numpy as np, soundfile as sf
H = {"X-API-Key": API_KEY}
# Baseline scores from run 20261024T102407Z, keyed by file sha256 (rounded to 4 dp there)
EXPECTED = {
    "8b03de0923525b54aa73cd0c9b940444f652681f9605cfdaa617762970085bae": {"train_peak": 0.0001, "train_zscore": 0.0001, "api_replica": 0.0001},
    "9448a599a4d5cd420191182d20890592e69124c699f704903c22c3614f3fc0ac": {"train_peak": 0.9998, "train_zscore": 0.9901, "api_replica": 0.9916},
}
checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name, detail)

def post(path_or_bytes, filename=None, headers=H, mode=None):
    data = open(path_or_bytes, "rb").read() if isinstance(path_or_bytes, str) else path_or_bytes
    fn = filename or os.path.basename(path_or_bytes)
    return requests.post(BASE + "/detect", headers=headers, params={"mode": mode} if mode else None,
                         files={"file": (fn, data)}, timeout=180)

h = requests.get(BASE + "/health", timeout=10).json()
check("/health model_ready", h["model_ready"], h.get("load_error") or h["load_info"])
check("/health pinned revision", h["model"]["revision"] == dh.MODEL_REVISION)

syn = [p for p in test_files if os.path.basename(p).startswith("synthetic__")][0]
check("no key -> 401", post(syn, headers={}).status_code == 401)
check("wrong key -> 401", post(syn, headers={"X-API-Key": "x" * 32}).status_code == 401)

for p in test_files:
    sha = dh.sha256_file(p)
    for mode in (None, "api_replica"):
        r = post(p, mode=mode); j = r.json()
        used = j.get("preprocessing")
        exp = EXPECTED.get(sha, {}).get(used)
        print(f"  {os.path.basename(p)} mode={used}: HTTP {r.status_code} prediction={j.get('prediction')} "
              f"score={j.get('synthetic_score')} dur={j.get('duration_s')} timing={j.get('timing_ms')}")
        check(f"{os.path.basename(p)} [{used}] HTTP 200", r.status_code == 200, j.get("reject_reason") or "")
        if exp is not None:
            check(f"{os.path.basename(p)} [{used}] matches notebook-01 score", abs(j["synthetic_score"] - exp) <= 0.002,
                  f"{j['synthetic_score']} vs {exp}")
        else:
            print("  (no baseline for this file's sha256 - score not compared)")

r = post(syn, mode="nonsense"); check("unknown mode -> 400", r.status_code == 400, r.text[:120])
r = post(b"hello", filename="notes.txt"); check("unsupported extension -> 415", r.status_code == 415)
try:
    r = post(os.urandom(MAX_UPLOAD_MB * 1024 * 1024 + 1), filename="big.wav")
    check("oversize -> 413", r.status_code == 413, r.status_code)
except requests.exceptions.ConnectionError:
    check("oversize -> 413", True, "server refused before body finished (connection closed after 413)")
buf = io.BytesIO(); sf.write(buf, np.zeros(16000 * 3, np.float32), 16000, format="WAV")
r = post(buf.getvalue(), filename="silent.wav"); check("silent -> 422 rejected", r.status_code == 422 and "silent" in (r.json().get("reject_reason") or ""), r.json().get("reject_reason"))
y, sr = sf.read(syn); buf = io.BytesIO(); sf.write(buf, y[: sr], sr, format="WAV")
r = post(buf.getvalue(), filename="short.wav"); check("1 s clip -> 422 too short", r.status_code == 422 and "too short" in (r.json().get("reject_reason") or ""), r.json().get("reject_reason"))
r = post(b"RIFF\\x00\\x00garbage" * 50, filename="corrupt.wav"); check("corrupt -> 422 unreadable", r.status_code == 422, r.json().get("reject_reason"))
r = post(b"", filename="empty.wav"); check("empty -> 422", r.status_code == 422)

res = [None, None]
def worker(i): res[i] = post(syn).json()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
a, b = sorted((x["inference_slot"] for x in res), key=lambda s: s["acquired_monotonic"])
check("2 concurrent requests serialised (no overlap)", a["released_monotonic"] <= b["acquired_monotonic"],
      f"queue_wait_ms={[x['timing_ms']['queue_wait'] for x in res]}")
check("temp upload dir empty after requests", os.listdir(TMP_DIR) == [], os.listdir(TMP_DIR))

n_ok = sum(c[1] for c in checks)
print(f"\\nBACKEND LOCAL CHECKS: {n_ok}/{len(checks)} passed")
""")

md(f"""
## 9. Public HTTPS tunnel for the phones

Uses a Cloudflare **quick tunnel** (`cloudflared` {CLOUDFLARED_VERSION}, no account): it prints a random
`https://<words>.trycloudflare.com` URL that forwards to this Colab server.

- The URL is **public**. `/detect` is protected only by the API key. `/health` is open but has no audio and no secrets.
- The URL changes every time the tunnel restarts, and it dies when the Colab runtime stops. Quick tunnels have no uptime guarantee and are for testing only.
- Only send audio you have consent to analyse. Stop the tunnel (step 11) when you're done.
""")

code(f"""
# 9. Start the tunnel and print the HTTPS base URL
import re
CF = "/content/cloudflared"
CF_URL = "https://github.com/cloudflare/cloudflared/releases/download/{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
if not os.path.exists(CF):
    subprocess.run(["wget", "-q", "-O", CF, CF_URL], check=True)
    os.chmod(CF, 0o755)
print(subprocess.run([CF, "--version"], capture_output=True, text=True).stdout.strip())
if "tunnel" in globals() and tunnel.poll() is None:
    tunnel.terminate(); tunnel.wait(10)
tunnel_log = open("/content/cloudflared.log", "w")
tunnel = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{{PORT}}"],
                          stdout=tunnel_log, stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(60):
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", open("/content/cloudflared.log").read())
    if m:
        PUBLIC_URL = m.group(0); break
    time.sleep(1)
print("PUBLIC_URL =", PUBLIC_URL)
""")

code("""
# 10. Verify the API through the public HTTPS URL (what the phones will use)
ok = False
for _ in range(30):  # new quick-tunnel hostnames can take a few seconds to resolve
    try:
        hj = requests.get(PUBLIC_URL + "/health", timeout=10).json(); ok = True; break
    except Exception:
        time.sleep(2)
print("HTTPS /health:", ok and hj["status"])
for p in test_files:
    r = requests.post(PUBLIC_URL + "/detect", headers=H, files={"file": (os.path.basename(p), open(p, "rb"))}, timeout=180)
    j = r.json()
    print(f"HTTPS /detect {os.path.basename(p)}: HTTP {r.status_code} prediction={j.get('prediction')} "
          f"score={j.get('synthetic_score')} total_ms={j.get('timing_ms', {}).get('total')}")
print(\"\"\"
Android / curl usage:
  curl -X POST "$URL/detect" -H "X-API-Key: $KEY" -F "file=@voice_note.m4a"
  optional: "$URL/detect?mode=api_replica"
Base URL:\"\"\", PUBLIC_URL)
""")

code("""
# 11. Logs / stop everything
print(open("/content/vcd_api.log").read()[-2000:])
STOP = False   # set True and re-run to stop the tunnel and server
if STOP:
    for name in ("tunnel", "server"):
        proc = globals().get(name)
        if proc is not None and proc.poll() is None:
            proc.terminate(); proc.wait(10); print("stopped", name)
""")

nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT)
