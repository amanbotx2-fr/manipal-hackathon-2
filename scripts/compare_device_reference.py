"""Stage-by-stage comparison of on-device dumps against the Colab reference export.

Usage: python3 scripts/compare_device_reference.py <reference_dir> <device_dumps_dir> [out.json]
  reference_dir: unzipped vcd_reference_export.zip (manifest.json + genuine/ + synthetic/)
  device_dumps_dir: adb-pulled files/dumps/ (one <sha12>/ folder per clip with native_mono.f32,
                    x16.f32, windows_<mode>.f32, result_<mode>.json)
Standard library only. Compares arrays; runs no model.
"""

import json
import math
import os
import sys
from array import array


def load_f32(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def diff(dev, ref, lag=0):
    n = 0; mx = 0.0; ss = 0.0; rs = 0.0
    for i in range(len(dev)):
        j = i + lag
        if 0 <= j < len(ref):
            d = dev[i] - ref[j]
            ad = abs(d)
            if ad > mx: mx = ad
            ss += d * d; rs += ref[j] * ref[j]; n += 1
    return {"device_len": len(dev), "colab_len": len(ref), "overlap": n, "lag": lag,
            "max_abs_diff": mx, "rms_diff": math.sqrt(ss / n) if n else None,
            "snr_db": (10 * math.log10(rs / ss) if ss > 0 else float("inf")) if n else None}


def main():
    ref_dir, dump_dir = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    m = json.load(open(os.path.join(ref_dir, "manifest.json")))
    report = {"reference_versions": m["versions"], "clips": {}}
    for cid, c in m["clips"].items():
        ddir = os.path.join(dump_dir, c["source_sha256"][:12])
        r = {"source_sha256": c["source_sha256"], "colab_decoder": c["decoder"]}
        dev_native = load_f32(os.path.join(ddir, "native_mono.f32"))
        ref_native = load_f32(os.path.join(ref_dir, c["native_mono"]["file"]))
        r["decode_native_mono"] = diff(dev_native, ref_native)
        dev_x16 = load_f32(os.path.join(ddir, "x16.f32"))
        ref_x16 = load_f32(os.path.join(ref_dir, c["x16"]["file"]))
        r["resample_x16"] = diff(dev_x16, ref_x16)
        r["modes"] = {}
        for mode, mm in c["modes"].items():
            res_path = os.path.join(ddir, f"result_{mode}.json")
            win_path = os.path.join(ddir, f"windows_{mode}.f32")
            if not (os.path.exists(res_path) and os.path.exists(win_path)):
                r["modes"][mode] = {"missing": True}
                continue
            dev_res = json.load(open(res_path))
            dev_w = load_f32(win_path)
            ref_w = load_f32(os.path.join(ref_dir, mm["windows"]["file"]))
            k = len(mm["logits"]); win = len(ref_w) // k
            per = []
            for i in range(k):
                wd = diff(dev_w[i * win:(i + 1) * win], ref_w[i * win:(i + 1) * win])
                dl = dev_res["windows"][i]["logits"]; rl = mm["logits"][i]
                wd["device_logits"] = dl; wd["colab_logits"] = rl
                wd["max_abs_logit_diff"] = max(abs(a - b) for a, b in zip(dl, rl))
                wd["device_score"] = dev_res["windows"][i]["synthetic_score"]; wd["colab_score"] = mm["scores"][i]
                per.append(wd)
            r["modes"][mode] = {
                "window_count": {"device": len(dev_res["windows"]), "colab": k},
                "starts_equal": [round(w["start_s"] * 16000) for w in dev_res["windows"]] == mm["starts"],
                "windows": per,
                "max_abs_window_diff": max(p["max_abs_diff"] for p in per),
                "max_abs_logit_diff": max(p["max_abs_logit_diff"] for p in per),
                "clip_score": {"device": dev_res["synthetic_score"], "colab": mm["clip_score_mean"],
                               "abs_diff": abs(dev_res["synthetic_score"] - mm["clip_score_mean"])},
                "prediction": {"device": dev_res["prediction"],
                               "colab": "synthetic" if round(mm["clip_score_mean"], 4) >= 0.5 else "genuine"},
            }
        report["clips"][cid] = r

    for cid, r in report["clips"].items():
        print(f"== {cid}  (Colab decoder: {r['colab_decoder']})")
        for stage in ("decode_native_mono", "resample_x16"):
            d = r[stage]
            print(f"  {stage:20s} len dev/colab {d['device_len']}/{d['colab_len']}  max|d| {d['max_abs_diff']:.3e}  "
                  f"rms {d['rms_diff']:.3e}  SNR {d['snr_db']:.1f} dB")
        for mode, mm in r["modes"].items():
            if mm.get("missing"):
                print(f"  {mode:12s} MISSING device dump"); continue
            print(f"  {mode:12s} windows {mm['window_count']} starts_equal={mm['starts_equal']}  "
                  f"max|window d| {mm['max_abs_window_diff']:.3e}  max|logit d| {mm['max_abs_logit_diff']:.3e}  "
                  f"score dev/colab {mm['clip_score']['device']:.8f}/{mm['clip_score']['colab']:.8f}  "
                  f"pred {mm['prediction']['device']}/{mm['prediction']['colab']}")
            for i, w in enumerate(mm["windows"]):
                print(f"      w{i}: window max|d| {w['max_abs_diff']:.3e} SNR {w['snr_db']:.1f} dB  logit d {w['max_abs_logit_diff']:.3e}")
    if out_path:
        json.dump(report, open(out_path, "w"), indent=1)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
