# 14. Package the ALREADY-EXPORTED candidate ONNX for the Android hand-off (no re-export, no denoising).
#     Run in the notebook-04 runtime after 12b. Needs: cand, dh, ad, el, WORK, RUN_DIR, DEVICE.
#     Produces /content/vcd_eval/<pkg>.zip with: model file(s), SHA256SUMS, MODEL_SPEC.json, reference inputs +
#     expected outputs for parity, LICENSE_ATTRIBUTION.md, HANDOFF.md. Nothing here changes any score or threshold.
import os, json, time, hashlib, shutil, datetime, platform
import numpy as np, torch, onnx, onnxruntime as ort, soundfile as sf

onnx_path = os.path.join(WORK, "antideepfake_small.onnx")
assert os.path.exists(onnx_path), "run 12b first (exported model not found)"
PKG_NAME = "antideepfake_small_onnx_pkg_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
PKG = os.path.join(WORK, PKG_NAME); os.makedirs(PKG, exist_ok=True)

# --- 1. model file(s): detect external-data tensors (dynamo export may store large weights beside the .onnx)
m = onnx.load(onnx_path, load_external_data=False)
from onnx.external_data_helper import uses_external_data
ext_files = sorted({t.external_data[0].value for t in m.graph.initializer if uses_external_data(t) and t.external_data})
model_files = [onnx_path] + [os.path.join(os.path.dirname(onnx_path), f) for f in ext_files]
for f in model_files:
    shutil.copy2(f, os.path.join(PKG, os.path.basename(f)))
opsets = {(o.domain or "ai.onnx"): o.version for o in m.opset_import}
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
assert inp.name == "wav", inp.name

# --- 2. reference inputs for parity: deterministic arrays (no dataset audio is redistributed) at 2 s, 4 s, 10 s,
#        plus one synthetic 16 kHz PCM16 WAV that exercises the full Android path (decode -> mono -> 16 kHz -> normalise).
def layer_norm(x):  # the exact evaluated preprocessing (model card: F.layer_norm(wav, wav.shape); eps 1e-5, biased variance)
    x = x.astype(np.float64); return ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32)
assert np.allclose(layer_norm(np.arange(10, dtype=np.float32)), ad.AntiDeepfakeHF.preprocess(np.arange(10, dtype=np.float32)))

rng = np.random.default_rng(20261024)
REF = {}
raw_inputs = {
    "ref_noise_2s": (rng.standard_normal(32000) * 0.05).astype(np.float32),
    "ref_noise_4s": (rng.standard_normal(64000) * 0.1).astype(np.float32),
    "ref_tone_chirp_10s": (0.3 * np.sin(2 * np.pi * np.cumsum(np.linspace(100, 3000, 160000)) / 16000)).astype(np.float32),
}
t = np.arange(64000) / 16000.0
wav_sig = (0.25 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(64000)).astype(np.float32)
sf.write(os.path.join(PKG, "ref_wav_16k_pcm16_4s.wav"), wav_sig, 16000, subtype="PCM_16")
wav_back, sr = sf.read(os.path.join(PKG, "ref_wav_16k_pcm16_4s.wav"), dtype="float32")   # what Android will decode (PCM16 -> float)
raw_inputs["ref_wav_16k_pcm16_4s"] = wav_back.astype(np.float32)
os.makedirs(os.path.join(PKG, "reference_inputs"), exist_ok=True)
w = cand.encoder.cpu().eval(); h = cand.head.cpu().eval()
for name, raw in raw_inputs.items():
    x = layer_norm(raw)
    np.save(os.path.join(PKG, "reference_inputs", f"{name}.raw_float32.npy"), raw)
    np.save(os.path.join(PKG, "reference_inputs", f"{name}.normalised_input_float32.npy"), x)
    t0 = time.perf_counter(); zo = sess.run(None, {inp.name: x[None]})[0][0]; ms = (time.perf_counter() - t0) * 1000
    with torch.no_grad(): zt = h(w(torch.from_numpy(x)[None]).last_hidden_state.mean(dim=1))[0].numpy()
    p = ad.softmax(zo)
    REF[name] = {"n_samples": int(len(raw)), "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                 "normalised_input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                 "expected_logits_fake_real_ort": [float(a) for a in zo], "expected_logits_fake_real_torch": [float(a) for a in zt],
                 "expected_synthetic_score": float(p[0]), "expected_genuine_score": float(p[1]),
                 "decision_at_0.5": "synthetic" if p[0] >= 0.5 else "genuine", "ort_cpu_ms": round(ms, 1),
                 "max_abs_logit_diff_ort_vs_torch": float(np.max(np.abs(zo - zt)))}
cand.encoder.to(DEVICE); cand.head.to(DEVICE)
PARITY_TOL = {"logits_abs": 1e-3, "synthetic_score_abs": 1e-4,
              "note": "Android fp32 ORT should match these within tolerance; fp16/int8 builds will not and must be re-evaluated, not just parity-checked"}

# --- 3. spec
prev = json.load(open(os.path.join(RUN_DIR, "metrics.json")))
SPEC = {
    "model": {"name": "AntiDeepfake Wav2Vec2-Small (NII Yamagishi Lab), converted to ONNX by HYPENEXIS evaluation", "source_repo": ad.MODEL_REPO,
              "source_revision": ad.MODEL_REVISION, "source_file": ad.MODEL_FILE, "source_sha256": ad.MODEL_SHA256,
              "params_total": cand.report["n_params_encoder"] + cand.report["n_params_head"], "weights_untouched": True,
              "conversion": "fairseq->transformers key mapping (strict), parity vs authors' fairseq code verified: max |delta fake score| 4.6e-7 on 12 clips (run 20261024T140853Z cell 13)",
              "onnx_export": {"exporter": "torch.onnx dynamo", "torch": torch.__version__, "onnx_ir_version": m.ir_version, "opset_import": opsets,
                              "producer": m.producer_name, "ort_vs_torch_max_abs_fake_score_diff_8_clips": 2.119046221271681e-07}},
    "files": {os.path.basename(f): {"bytes": os.path.getsize(f), "sha256": dh.sha256_file(f)} for f in model_files},
    "external_data_files": ext_files,
    "runtime_verified": {"onnxruntime": ort.__version__, "provider": "CPUExecutionProvider", "colab_cpu_median_ms_per_clip_8_clips": 568.3},
    "io": {"input": {"name": inp.name, "dtype": "float32", "shape": inp.shape, "semantics": "[1, T] normalised mono 16 kHz waveform; batch must be 1",
                     "T_range_exported": [16000, 160000], "T_range_policy": [32000, 160000]},
           "output": {"name": outp.name, "dtype": "float32", "shape": outp.shape, "semantics": "[1, 2] logits in order [fake, real]"}},
    "preprocessing_whole_clip_as_evaluated": [
        "1. decode to float in [-1, 1] (PCM16 / 32768; AAC/Opus via platform decoder)",
        "2. mono: mean over channels",
        "3. resample to 16000 Hz with a high-quality resampler (evaluation used libsoxr HQ; any windowed-sinc resampler is acceptable, record which)",
        "4. policy gates (reject, never pad or substitute): duration < 2.0 s -> reject; > 10.0 s -> analyse first 10.0 s only; "
        "peak < -50 dBFS -> reject (silent); < 1.0 s of 25 ms frames within 35 dB of the loudest frame AND above -55 dBFS -> reject (mostly silent); non-finite samples -> reject",
        "5. normalise over the WHOLE analysed clip: x = (x - mean(x)) / sqrt(var_biased(x) + 1e-5), float32 (epsilon 1e-5; biased variance, i.e. divide by N)",
        "6. ONE forward pass on the whole clip: input [1, T]; no windowing, no cropping, no padding",
        "7. softmax over the 2 logits; synthetic_score = softmax[0] (index 0 = fake); genuine_score = softmax[1]",
        "8. decision: synthetic if synthetic_score >= 0.5 (frozen provisional threshold; NOT tuned; a validation-selected threshold may replace it later)",
        "9. the score is an uncalibrated model output, not a probability; display it as a score"],
    "frozen_threshold": 0.5,
    "do_not": ["batch > 1 (export/eval assume batch 1)", "z-score per window or peak normalisation (that was the Dhwani recipe)", "apply any denoiser (experimental, notebook 06, not part of this package)",
               "use the EER threshold 0.178 from evaluation outputs or the card's 0.8128 for decisions"],
    "evaluation_evidence": {"heldout_itw_run": "20261024T140853Z", "n_scored": 244, "roc_auc": 0.9958, "eer": 0.0246,
                            "at_0.5": {"accuracy": 0.967, "precision": 0.975, "recall": 0.960, "f1": 0.967, "fpr": 0.025},
                            "robustness_run": "20261024T151905Z", "robustness_verdict": "FAIL on white noise 10 dB SNR (EER 15.98 %, ratio 6.50 > 2.5); narrowband 4.10 %, Opus16k 4.10 %, MP3-32k 2.46 %",
                            "caveats": ["In-the-Wild pilot subsets (English, celebrity speech), not a full benchmark", "speakers shared across subsets", "no Hindi/Indic or phone-call evaluation yet",
                                        "loud/quiet audio fine; 1 s silence pad +1.6 pp; first-3 s +2.5 pp; telephone band+mu-law +4.9 pp"]},
    "reference_vectors": REF, "parity_tolerance": PARITY_TOL,
    "licence": {"weights": "CC BY-NC-SA 4.0 (NII) - research/educational use; commercial deployment requires permission from NII",
                "base_model": "facebook/wav2vec2-base, Apache-2.0", "authors_code": "BSD-3 (github.com/nii-yamagishilab/AntiDeepfake)",
                "our_conversion_and_eval_code": "project code (evaluation/)", "citation": "Ge, Wang, Liu, Yamagishi, 'Post-training for Deepfake Speech Detection', arXiv:2506.21090"},
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "generated_in": platform.platform(),
}
el.jdump(SPEC, os.path.join(PKG, "MODEL_SPEC.json"))

# --- 4. checksums, licence/attribution, hand-off text
with open(os.path.join(PKG, "SHA256SUMS"), "w") as f:
    for root, _, files in os.walk(PKG):
        for fn in sorted(files):
            if fn == "SHA256SUMS": continue
            p = os.path.join(root, fn); f.write(f"{dh.sha256_file(p)}  {os.path.relpath(p, PKG)}\n")
open(os.path.join(PKG, "LICENSE_ATTRIBUTION.md"), "w").write(f"""# Attribution and licence

Detector weights: **AntiDeepfake Wav2Vec2-Small**, Yamagishi Lab, National Institute of Informatics (NII), Japan.
Source: https://huggingface.co/{ad.MODEL_REPO} (revision {ad.MODEL_REVISION}, model.safetensors sha256 {ad.MODEL_SHA256}).
Licence of the weights: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** - research and educational use.
A commercial release of the app with these weights requires permission from NII. Derived files (this ONNX) carry the same licence.
Base model: facebook/wav2vec2-base (Apache-2.0). Authors' training/inference code: BSD-3.
Cite: W. Ge, X. Wang, X. Liu, J. Yamagishi, "Post-training for Deepfake Speech Detection", arXiv:2506.21090 (2025).
We did not train or fine-tune these weights; we converted them (fairseq -> transformers -> ONNX) and evaluated them.
""")
open(os.path.join(PKG, "HANDOFF.md"), "w").write(f"""# Android hand-off: AntiDeepfake Wav2Vec2-Small ONNX (fp32)

**Files:** {', '.join(os.path.basename(f) for f in model_files)} ({sum(os.path.getsize(f) for f in model_files)/1e6:.1f} MB total; checksums in SHA256SUMS).
Everything else you need is in `MODEL_SPEC.json`; this file is the short version.

**Input** `{inp.name}`: float32 `[1, T]`, T between 32000 and 160000 samples (2-10 s at 16 kHz; the export allows 16000-160000). Batch size 1 only.
**Output** `{outp.name}`: float32 `[1, 2]` = logits `[fake, real]`. `synthetic_score = softmax(logits)[0]`. Decision: synthetic if `synthetic_score >= 0.5`.

**Preprocessing (whole clip, exactly as evaluated):** decode -> mono (mean) -> 16 kHz -> policy gates (reject < 2 s, analyse first 10 s, reject silent / mostly-silent)
-> `x = (x - mean) / sqrt(var_biased + 1e-5)` over the whole clip -> one forward pass. No windowing, no peak normalisation, no per-window z-score, no denoiser.
The score is uncalibrated; show it as a score, not a probability. 0.5 is a frozen provisional threshold.

**Parity check:** `reference_inputs/*.normalised_input_float32.npy` are ready-to-feed inputs; `*.raw_float32.npy` are the pre-normalisation signals;
`ref_wav_16k_pcm16_4s.wav` tests your whole decode+normalise path. Expected logits/scores and tolerances are in `MODEL_SPEC.json` -> `reference_vectors`
(fp32 ORT should match within 1e-3 on logits). Verified here: ORT vs PyTorch max |delta score| 2.1e-7 on 8 real clips.

**Size/latency:** 378.8 MB fp32; Colab 2-vCPU ORT median 568 ms per clip (not a phone number). Expect several hundred ms to a few seconds on a phone CPU.
fp16/int8 quantisation is **not** evaluated: if you quantise, the model must be re-evaluated (notebook 04/05 protocol), not just parity-checked.

**Evidence:** held-out In-the-Wild pilot subset (244 clips): ROC-AUC 0.996, EER 2.5 %, accuracy 0.967 at 0.5. Robustness: fine for 8 kHz band-limiting and Opus/MP3,
**fails under additive noise at 10 dB SNR** (EER 16 %). Not yet evaluated on Hindi/Indic speech or real phone calls. Denoising is experimental and not part of this package.

**Licence:** CC BY-NC-SA 4.0 (research use). See LICENSE_ATTRIBUTION.md. Attribute NII in the app's about screen.
""")
zip_path = shutil.make_archive(os.path.join(WORK, PKG_NAME), "zip", PKG)
shutil.copy2(os.path.join(PKG, "MODEL_SPEC.json"), os.path.join(RUN_DIR, "android_package_MODEL_SPEC.json"))
print(open(os.path.join(PKG, "SHA256SUMS")).read())
print(json.dumps({"input": SPEC["io"]["input"], "output": SPEC["io"]["output"], "opsets": opsets, "external_data_files": ext_files,
                  "reference_vectors": {k: {kk: v[kk] for kk in ("expected_logits_fake_real_ort", "expected_synthetic_score", "max_abs_logit_diff_ort_vs_torch")} for k, v in REF.items()}}, indent=1))
print("package:", zip_path, f"{os.path.getsize(zip_path)/1e6:.1f} MB")
try:
    from google.colab import files; files.download(zip_path)
except ImportError: pass
