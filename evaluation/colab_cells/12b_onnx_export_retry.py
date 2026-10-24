# 12b. ONNX export retry (run in the SAME runtime after cells 1-11; the first attempt failed with
#      "No module named 'onnxscript'": torch >= 2.9's default exporter is the dynamo one and needs onnxscript).
#      Deployment-feasibility evidence only. Nothing here changes any score.
import subprocess, sys, os, time, json
import numpy as np, torch, onnxruntime as ort
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "onnxscript", "onnx"], check=True)

class Wrapped(torch.nn.Module):
    def __init__(s, enc, head): super().__init__(); s.enc, s.head = enc, head
    def forward(s, wav): return s.head(s.enc(wav).last_hidden_state.mean(dim=1))

w = Wrapped(cand.encoder, cand.head).eval().cpu()
onnx_path = os.path.join(WORK, "antideepfake_small.onnx")
ONNX_REPORT = {"attempted": True, "exporter": None}
for exporter in ("dynamo", "torchscript"):
    try:
        dummy = torch.randn(1, 64000)
        if exporter == "dynamo":
            prog = torch.onnx.export(w, (dummy,), dynamo=True, dynamic_shapes={"wav": {1: torch.export.Dim("time", min=16000, max=160000)}})
            prog.save(onnx_path)
        else:
            torch.onnx.export(w, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17,
                              dynamic_axes={"input": {1: "time"}}, dynamo=False)
        ONNX_REPORT["exporter"] = exporter
        break
    except Exception as e:
        ONNX_REPORT[f"{exporter}_error"] = f"{type(e).__name__}: {str(e)[:400]}"
if ONNX_REPORT["exporter"]:
    s2 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = s2.get_inputs()[0].name
    diffs, lat = [], []
    for m in OK.head(8).itertuples():
        x16 = dh.to_mono_16k(*dh.load_audio(os.path.join(WORK, "audio", "heldout", os.path.basename(m.clip_id))))
        x16, _ = dh.validate(x16); x = ad.AntiDeepfakeHF.preprocess(x16)
        t0 = time.perf_counter(); zo = s2.run(None, {in_name: x[None]})[0][0]; lat.append((time.perf_counter() - t0) * 1000)
        with torch.no_grad(): zt = w(torch.from_numpy(x)[None]).numpy()[0]
        diffs.append(float(abs(ad.softmax(zo)[0] - ad.softmax(zt)[0])))
    ONNX_REPORT.update(ok=True, file_mb=round(os.path.getsize(onnx_path) / 1e6, 1), input_name=in_name,
                       max_abs_fake_score_diff=max(diffs), ort_cpu_ms_median=float(np.median(lat)), n_clips=len(diffs),
                       clip_seconds_median=float(np.median([len(dh.to_mono_16k(*dh.load_audio(os.path.join(WORK, "audio", "heldout", os.path.basename(m.clip_id)))))/16000 for m in OK.head(8).itertuples()])),
                       note="fp32 ONNX, dynamic time axis, Colab CPU (2 vCPU); int8/fp16 and on-device latency not measured")
else:
    ONNX_REPORT["ok"] = False
cand.encoder.to(DEVICE); cand.head.to(DEVICE)
print(json.dumps(ONNX_REPORT, indent=1))
# append to the saved metrics without touching predictions
mp = os.path.join(RUN_DIR, "metrics.json"); M = json.load(open(mp)); M["onnx_retry"] = ONNX_REPORT; el.jdump(M, mp)
el.jdump(ONNX_REPORT, os.path.join(RUN_DIR, "onnx_retry.json"))
print("saved onnx_retry.json; download it (or re-zip RUN_DIR) and add it to the run folder")
