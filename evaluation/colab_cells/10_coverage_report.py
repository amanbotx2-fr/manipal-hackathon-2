# 10. Load saved predictions (READ-ONLY: nothing is re-scored or rewritten); dataset status, coverage, exclusions
import os, json
import numpy as np, pandas as pd

PLANNED_N = {"asvspoof2019_la_eval": ASV_N_GENUINE + 13 * ASV_N_PER_ATTACK,   # 104 + 8 x 13 = 208
             "in_the_wild": ITW_N_GENUINE + ITW_N_SYNTHETIC}                  # 200
PRED_PATH = os.path.join(RUN_DIR, "predictions.jsonl")
_raw = [json.loads(l) for l in open(PRED_PATH, encoding="utf-8") if l.strip()]
_acc_path = os.path.join(RUN_DIR, "sources_and_access.json")
_acc = json.load(open(_acc_path, encoding="utf-8")) if os.path.exists(_acc_path) else {}
_blockers = _acc.get("blockers", []) + [b for b in globals().get("BLOCKERS", []) if b not in _acc.get("blockers", [])]

# --- which datasets are actually present, and why -------------------------------------------
DATASET_STATUS = {}
for ds in DATASETS:
    man = os.path.join(RUN_DIR, f"manifest_{ds}.csv")
    n_man = len(pd.read_csv(man)) if os.path.exists(man) else 0
    n_pred = sum(1 for r in _raw if r.get("dataset") == ds)
    errs = sorted({b["error"] for b in _blockers if b.get("dataset") == ds})
    if n_pred == 0:
        status = "NOT TESTED"
        why = ("access blocker: " + " | ".join(errs)) if errs else (
            "no manifest written (dataset never reached selection)" if not n_man else "selected but never scored")
    elif n_pred < n_man:
        status, why = "INCOMPLETE", f"{n_pred}/{n_man} manifest clips have predictions"
    else:
        status, why = "scored", ""
    DATASET_STATUS[ds] = {"status": status, "reason": why, "planned_clips": PLANNED_N.get(ds),
                          "manifest_clips": n_man, "prediction_rows": n_pred,
                          "metadata": _acc.get("metadata", {}).get(ds), "access": _acc.get("access", {}).get(ds),
                          "blockers": errs}
print(f"predictions.jsonl: {len(_raw)} rows (read only) | planned {sum(PLANNED_N.values())} | "
      f"in-memory MANIFEST: {len(MANIFEST)}")
for ds, s in DATASET_STATUS.items():
    print(f"  {ds:22s} {s['status']:10s} planned {s['planned_clips']}, manifest {s['manifest_clips']}, "
          f"predictions {s['prediction_rows']}  {s['reason']}")
TESTED = [ds for ds, s in DATASET_STATUS.items() if s["prediction_rows"] > 0]
assert TESTED, f"no dataset has predictions: {DATASET_STATUS}"
NOT_TESTED = [ds for ds in DATASETS if ds not in TESTED]
el.jdump(DATASET_STATUS, os.path.join(RUN_DIR, "dataset_status.json"))   # new file; predictions untouched

# --- predictions of present datasets only ----------------------------------------------------
P = pd.DataFrame([r for r in _raw if r.get("dataset") in TESTED])
P = P.drop_duplicates(["dataset", "clip_id"], keep="last").sort_values(["dataset", "selection_rank"]).reset_index(drop=True)
for ds in TESTED:
    assert (P.dataset == ds).sum() == DATASET_STATUS[ds]["manifest_clips"], f"{ds}: predictions != manifest"
for c in ("attack_id", "reject_category", "reject_reason", "window_scores", "truncated",
          "duration_s", "container_duration_s"):
    if c not in P:
        P[c] = np.nan
P["attack_id"] = P["attack_id"].fillna("").astype(str)
P["reject_category"] = P["reject_category"].fillna("").astype(str)
P["n_windows"] = P["window_scores"].map(lambda s: len(json.loads(s)) if isinstance(s, str) else 0)
P["attack_or_label"] = np.where(P.label == "synthetic", P.attack_id.where(P.attack_id != "", "synthetic"), "genuine")
P["_truncated"] = P["truncated"].map(lambda v: v is True or str(v).lower() == "true")

COV_COLS = ["selected", "fetch_failed", "rejected", "scored", "coverage", "truncated_to_10s", "single_padded_window(<=4s)"]
def _cov(g):
    ok = g.status == "ok"
    return {"selected": len(g), "fetch_failed": int((g.status == "fetch_failed").sum()),
            "rejected": int((g.status == "rejected").sum()), "scored": int(ok.sum()),
            "coverage": round(float(ok.mean()), 4) if len(g) else None,
            "truncated_to_10s": int(g._truncated.sum()),
            "single_padded_window(<=4s)": int((ok & (g.n_windows == 1)).sum())}

COV = pd.DataFrame([{"dataset": ds, "label": lab, **_cov(g)} for (ds, lab), g in P.groupby(["dataset", "label"])],
                   columns=["dataset", "label"] + COV_COLS)
_asv = P[P.dataset == "asvspoof2019_la_eval"]
COV_ATTACK = pd.DataFrame([{"attack_or_label": k, **_cov(g)} for k, g in _asv.groupby("attack_or_label")],
                          columns=["attack_or_label"] + COV_COLS)
_bad = P[P.status != "ok"]
REJ = pd.DataFrame([{"dataset": d, "label": l, "reject_category": c, "n": len(g)}
                    for (d, l, c), g in _bad.groupby(["dataset", "label", "reject_category"])],
                   columns=["dataset", "label", "reject_category", "n"])
EXCLUDED = _bad.reindex(columns=["dataset", "clip_id", "label", "attack_id", "speaker_id", "status",
                                 "reject_category", "reject_reason", "container_duration_s"])
_okp = P[P.status == "ok"]
DUR = (_okp.groupby(["dataset", "label"])["duration_s"].describe(percentiles=[.05, .5, .95]).round(2)
       if len(_okp) else pd.DataFrame())

print("\nCoverage by dataset x label:\n", COV.to_string(index=False))
print("\nASVspoof coverage by attack:\n", COV_ATTACK.to_string(index=False) if len(COV_ATTACK)
      else "  none: ASVspoof 2019 LA was NOT TESTED in this run")
print("\nRejection / fetch-failure reasons:\n", REJ.to_string(index=False) if len(REJ) else "  none")
print("\nDurations of scored clips (s):\n", DUR.to_string() if len(DUR) else "  none")
print("\nPolicy: < 2.0 s rejected (not replaced); > 10.0 s -> first 10 s only; <= 4 s -> one zero-padded window; "
      "> 4 s -> 4 s windows, 2 s hop, last aligned to end. Metrics use scored clips only.")
for ds in NOT_TESTED:
    print(f"\n*** {ds}: NOT TESTED in this run ({DATASET_STATUS[ds]['reason']}). "
          "No metric below refers to it; do not report it as evaluated. ***")
