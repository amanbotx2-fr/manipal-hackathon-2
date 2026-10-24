# Reference export for on-device parity checks (paste into ANY Colab runtime that has
# dhwani_infer.py, e.g. notebook 01 or 02). Exports, for both reference clips:
#   decoded native-rate mono, 16 kHz resampled audio, every preprocessed 4 s window for
#   all three modes (float32 little-endian), full-precision logits/scores, settings, hashes.
# Output: /content/vcd_reference_export.zip (~10 MB), downloaded automatically.
import os, sys, glob, json, shutil, zipfile, platform, subprocess, datetime
import numpy as np

for d in ("/content/vcd_app", "/content"):
    if os.path.exists(os.path.join(d, "dhwani_infer.py")) and d not in sys.path:
        sys.path.insert(0, d)
import dhwani_infer as dh
import onnxruntime as ort, librosa, soundfile as sf
from huggingface_hub import hf_hub_download

REF = {
    "genuine_hi_aman.m4a": "8b03de0923525b54aa73cd0c9b940444f652681f9605cfdaa617762970085bae",
    "synthetic__en__Chatterbox__emerald_city_of_oz_20_f000055.wav": "9448a599a4d5cd420191182d20890592e69124c699f704903c22c3614f3fc0ac",
}

# 1) locate the exact reference clips by sha256 (never by name alone)
found = {}
for p in glob.glob("/content/**/*", recursive=True):
    if p.startswith(("/content/drive", "/content/sample_data")):
        continue  # never crawl a mounted Google Drive
    if os.path.isfile(p) and p.lower().endswith((".m4a", ".wav")) and os.path.getsize(p) < 50e6:
        h = dh.sha256_file(p)
        for name, want in REF.items():
            if h == want and name not in found:
                found[name] = p
if "synthetic__en__Chatterbox__emerald_city_of_oz_20_f000055.wav" not in found:
    from google.colab import userdata
    found["synthetic__en__Chatterbox__emerald_city_of_oz_20_f000055.wav"] = hf_hub_download(
        "mueller91/MLAAD", "fake/en/Chatterbox/emerald_city_of_oz_20_f000055.wav", repo_type="dataset",
        revision="30c3dec763fa4803f11c2df1557d3b0828395366", token=userdata.get("HF_TOKEN"))
if "genuine_hi_aman.m4a" not in found:
    from google.colab import files
    print("Upload genuine_hi_aman.m4a (sha256 8b03de09...)")
    for n in files.upload():
        if dh.sha256_file(n) == REF["genuine_hi_aman.m4a"]:
            found["genuine_hi_aman.m4a"] = os.path.abspath(n)
for name, p in found.items():
    assert dh.sha256_file(p) == REF[name], f"hash mismatch for {name}"
print("clips:", json.dumps(found, indent=1))

# 2) model + session exactly as in the baseline (CPU EP, default session options)
mp = globals().get("MODEL_PATH")
if not mp or not os.path.exists(mp):
    try:
        mp = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION, local_files_only=True)
    except Exception:
        mp = hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION)
assert os.path.getsize(mp) == dh.MODEL_BYTES
sess_ref = dh.load_session(mp, verify_sha=True)
in_name = sess_ref.get_inputs()[0].name

# 3) export
OUT = "/content/vcd_reference_export"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
def save(rel, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    path = os.path.join(OUT, rel); os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)
    return {"file": rel, "shape": list(arr.shape), "dtype": "<f4", "sha256": dh.sha256_file(path)}

def decoder_used(p):
    try:
        sf.info(p); return "soundfile"
    except Exception:
        return "librosa/audioread (ffmpeg)"

try:
    ffmpeg_ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
except Exception as e:
    ffmpeg_ver = f"unavailable: {e}"
try:
    import soxr; soxr_ver = soxr.__version__
except Exception:
    soxr_ver = "unknown"

manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model": {"repo": dh.MODEL_REPO, "revision": dh.MODEL_REVISION, "sha256": dh.MODEL_SHA256,
              "providers": sess_ref.get_providers(), "session_options": "onnxruntime defaults (graph_optimization_level=ORT_ENABLE_ALL)",
              "input": [(i.name, i.shape, i.type) for i in sess_ref.get_inputs()],
              "output": [(o.name, o.shape, o.type) for o in sess_ref.get_outputs()]},
    "versions": {"python": platform.python_version(), "onnxruntime": ort.__version__, "librosa": librosa.__version__,
                 "soundfile": sf.__version__, "numpy": np.__version__, "soxr": soxr_ver, "ffmpeg": ffmpeg_ver},
    "preprocessing": {"SR": dh.SR, "WIN": dh.WIN, "HOP": dh.HOP, "MIN_DUR_S": dh.MIN_DUR_S, "MAX_DUR_S": dh.MAX_DUR_S,
                      "PEAK_TARGET_DBFS": dh.PEAK_TARGET_DBFS, "resample": "librosa.resample res_type=soxr_hq",
                      "mono": "mean over channels", "SILENT_PEAK_DBFS": dh.SILENT_PEAK_DBFS,
                      "ACTIVE_FRAME_S": dh.ACTIVE_FRAME_S, "ACTIVE_REL_DB": dh.ACTIVE_REL_DB,
                      "ACTIVE_ABS_DBFS": dh.ACTIVE_ABS_DBFS, "MIN_ACTIVE_S": dh.MIN_ACTIVE_S,
                      "SYNTHETIC_INDEX": dh.SYNTHETIC_INDEX, "DECISION_THRESHOLD": dh.DECISION_THRESHOLD},
    "clips": {},
}
for name, p in found.items():
    cid = "genuine" if name.startswith("genuine") else "synthetic"
    y, sr = dh.load_audio(p)
    native_mono = y.mean(axis=0) if y.shape[0] > 1 else y[0]
    x16 = dh.to_mono_16k(y, sr)
    xv, info = dh.validate(x16)
    clip = {"source_name": name, "source_sha256": REF[name], "decoder": decoder_used(p),
            "native_sr": sr, "channels": int(y.shape[0]), "native_samples": int(y.shape[1]),
            "native_mono": save(f"{cid}/native_mono.f32", native_mono),
            "x16": save(f"{cid}/x16.f32", x16), "validate": info, "modes": {}}
    for mode in dh.PREPROCESSING_MODES:
        wins, starts = dh.make_windows(xv, mode)
        logits = np.array([sess_ref.run(None, {in_name: w[None]})[0][0] for w in wins], dtype=np.float64)
        scores = dh.softmax(logits)[:, dh.SYNTHETIC_INDEX]
        clip["modes"][mode] = {"windows": save(f"{cid}/windows_{mode}.f32", wins), "starts": starts,
                               "logits": logits.tolist(), "scores": scores.tolist(),
                               "clip_score_mean": float(scores.mean()), "clip_score_max": float(scores.max())}
        print(f"{cid:9s} {mode:12s} windows={len(wins)} mean={scores.mean():.10f} logits[0]={logits[0].tolist()}")
    manifest["clips"][cid] = clip

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1, default=str)
zpath = "/content/vcd_reference_export.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(OUT):
        for fn in fs:
            z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), OUT))
print("zip:", zpath, round(os.path.getsize(zpath) / 1e6, 2), "MB | sha256", dh.sha256_file(zpath))
try:
    from google.colab import files; files.download(zpath)
except Exception:
    pass
