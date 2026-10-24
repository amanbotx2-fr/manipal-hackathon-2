# 5b. Fetch ONE pre-registered English synthetic clip + its metadata from MLAAD (gated, CC-BY-NC-4.0)
import os, io, csv, json, random, shutil, hashlib, glob
import soundfile as sf
from huggingface_hub import hf_hub_download
from google.colab import userdata

MLAAD_REPO, MLAAD_REV = "mueller91/MLAAD", "30c3dec763fa4803f11c2df1557d3b0828395366"
MLAAD_GENERATOR, MLAAD_LANG, MLAAD_SEED = "Chatterbox", "en", 20261024
UPLOAD_DIR = globals().get("UPLOAD_DIR", "/content/vcd_uploads")
META_DIR = "/content/vcd_meta"
os.makedirs(UPLOAD_DIR, exist_ok=True); os.makedirs(META_DIR, exist_ok=True)

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception as e:
    raise RuntimeError("Add HF_TOKEN in Colab Secrets (key icon) and enable notebook access") from e

def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

# 1) metadata for this generator only
folder = f"fake/{MLAAD_LANG}/{MLAAD_GENERATOR}"
meta_path = hf_hub_download(MLAAD_REPO, f"{folder}/meta.csv", repo_type="dataset",
                            revision=MLAAD_REV, token=HF_TOKEN)
text = open(meta_path, encoding="utf-8").read()
dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters="|,;\t")
rows_m = list(csv.DictReader(io.StringIO(text), dialect=dialect))
cols = list(rows_m[0].keys())
print(len(rows_m), "metadata rows; columns:", cols)
path_key = next((k for k in cols if k.strip().lower() in ("path", "file", "filename", "audio")), None)
dur_key = next((k for k in cols if k.strip().lower() in ("duration", "duration_s")), None)
assert path_key, f"no path column found in {cols}"

# 2) pre-registered order: sort by path, seeded shuffle; first 5-10 s clip (no detector involved)
cands = sorted(rows_m, key=lambda r: r[path_key])
random.Random(MLAAD_SEED).shuffle(cands)
checked, chosen = [], None
for r in cands:
    if dur_key and r[dur_key]:
        if not (5.0 <= float(r[dur_key]) <= 10.0):
            continue                      # filter on published duration, no download
    elif len(checked) >= 10:
        break                             # no duration column: cap trial downloads
    rel = f"{folder}/{os.path.basename(r[path_key].strip())}"
    local = hf_hub_download(MLAAD_REPO, rel, repo_type="dataset", revision=MLAAD_REV, token=HF_TOKEN)
    info = sf.info(local)
    checked.append((rel, round(info.duration, 3)))
    if 5.0 <= info.duration <= 10.0:
        chosen = (r, rel, local, info); break
assert chosen, f"no 5-10 s clip found; checked: {checked}"

# 3) byte copy into the upload dir (no re-encoding) + provenance sidecar
r, rel, local, info = chosen
sample_id, ext = os.path.splitext(os.path.basename(rel))
out_name = f"synthetic__{MLAAD_LANG}__{MLAAD_GENERATOR}__{sample_id}{ext}"
shutil.copyfile(local, os.path.join(UPLOAD_DIR, out_name))
sidecar = {
    "file_name": out_name, "label": "synthetic", "language": MLAAD_LANG,
    "generator": MLAAD_GENERATOR, "sample_id": sample_id,
    "dataset": MLAAD_REPO, "dataset_revision": MLAAD_REV, "license": "CC-BY-NC-4.0",
    "repo_path": rel, "sha256": _sha256(local),
    "selection_rule": f"meta.csv sorted by '{path_key}', random.Random({MLAAD_SEED}).shuffle, first clip 5-10 s",
    "clips_downloaded": checked,
    "audio": {"samplerate": info.samplerate, "channels": info.channels,
              "duration_s": round(info.duration, 3), "format": info.format, "subtype": info.subtype},
    "meta_csv_row": r,
}
with open(os.path.join(META_DIR, out_name + ".json"), "w", encoding="utf-8") as f:
    json.dump(sidecar, f, indent=2, ensure_ascii=False)
print(json.dumps(sidecar, indent=2, ensure_ascii=False))

paths = sorted(p for p in glob.glob(os.path.join(UPLOAD_DIR, "*")) if os.path.isfile(p))
print(len(paths), "file(s) in", UPLOAD_DIR, ":", [os.path.basename(p) for p in paths])
