"""Stdlib-only checks for evaluation/eval_lib.py (no ML deps, no network, no datasets).

Run: python3 -I evaluation/tests/test_eval_lib.py
"""

import io
import itertools
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_lib as el  # noqa: E402


def brute_auc(y, s):
    pos = [b for a, b in zip(y, s) if a == 1]
    neg = [b for a, b in zip(y, s) if a == 0]
    tot = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg))
    return tot / (len(pos) * len(neg))


def brute_eer(y, s):
    best = None
    for t in sorted(set(s)) + [float("inf")]:
        P = sum(y); N = len(y) - P
        fnr = sum(1 for a, b in zip(y, s) if a == 1 and b < t) / P
        fpr = sum(1 for a, b in zip(y, s) if a == 0 and b >= t) / N
        if best is None or abs(fnr - fpr) < best[0]:
            best = (abs(fnr - fpr), (fnr + fpr) / 2, t)
    return best[1], best[2]


def test_metrics():
    rng = random.Random(1)
    for trial in range(200):
        n = rng.randint(2, 40)
        y = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y)) < 2:
            y[0], y[1] = 0, 1
        s = [round(rng.random(), rng.choice([1, 2, 4])) for _ in range(n)]  # rounding forces ties
        assert abs(el.roc_auc(y, s) - brute_auc(y, s)) < 1e-12
        assert el.eer(y, s) == brute_eer(y, s), (y, s)
    # perfect / inverted / single-class
    assert el.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert el.eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])[0] == 0.0
    assert el.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert el.eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])[0] == 1.0
    assert el.roc_auc([1, 1], [0.3, 0.4]) is None and el.eer([1, 1], [0.3, 0.4]) == (None, None)
    m = el.at_threshold([1, 1, 1, 0, 0], [0.9, 0.5, 0.2, 0.6, 0.1], 0.5)
    c = m["confusion"]
    assert (c["tp_synthetic_as_synthetic"], c["fn_synthetic_as_genuine"],
            c["fp_genuine_as_synthetic"], c["tn_genuine_as_genuine"]) == (2, 1, 1, 1)  # 0.5 counts as synthetic
    assert abs(m["synthetic_recall"] - 2 / 3) < 1e-12 and m["genuine_fpr"] == 0.5
    assert abs(m["balanced_accuracy"] - (2 / 3 + 0.5) / 2) < 1e-12
    lo, hi = el.wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29
    e = el.evaluate([1, 1, 0, 0] * 10, [0.9, 0.6, 0.4, 0.1] * 10, 0.5, n_boot=50, seed=3)
    assert e["bootstrap95"]["roc_auc"] == (1.0, 1.0)
    assert e == el.evaluate([1, 1, 0, 0] * 10, [0.9, 0.6, 0.4, 0.1] * 10, 0.5, n_boot=50, seed=3)


PROTO = """LA_0039 LA_E_2834763 - A11 spoof
LA_0014 LA_E_8877452 - A14 spoof
LA_0040 LA_E_6828287 - - bonafide
"""


def test_parsers():
    r = el.parse_asvspoof_cm_protocol(PROTO)
    assert r[0] == {"clip_id": "LA_E_2834763", "speaker_id": "LA_0039", "attack_id": "A11",
                    "label": "synthetic", "original_label": "spoof"}
    assert r[2]["label"] == "genuine" and r[2]["attack_id"] == ""
    for bad in ("LA_1 LA_E_1 - A11\n", "LA_1 LA_E_1 - A11 fake\n", "LA_1 LA_E_1 - - spoof\n"):
        try:
            el.parse_asvspoof_cm_protocol(bad)
            raise AssertionError("should fail: " + bad)
        except ValueError:
            pass
    m = el.parse_itw_meta("file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Alec Guinness,bona-fide\n")
    assert [x["label"] for x in m] == ["synthetic", "genuine"] and m[0]["speaker_id"] == "Alec Guinness"
    for bad in ("file,speaker,label\n0.wav,X,maybe\n", "name,who,label\n0.wav,X,spoof\n"):
        try:
            el.parse_itw_meta(bad)
            raise AssertionError("should fail")
        except ValueError:
            pass


def fake_records(seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(3000):
        spoof = rng.random() < 0.85
        recs.append({"clip_id": f"LA_E_{1000000 + i}", "speaker_id": f"LA_{rng.randint(1, 48):04d}",
                     "attack_id": f"A{rng.randint(7, 19):02d}" if spoof else "",
                     "label": "synthetic" if spoof else "genuine",
                     "original_label": "spoof" if spoof else "bonafide"})
    return recs


def test_selection():
    recs = fake_records()
    sel, meta = el.select_subset(recs, "asvspoof2019_la_eval", 20261025, 104, n_per_attack=8)
    assert len(sel) == 208 and meta["shortfalls"] == {}
    assert meta["selected"]["by_label"] == {"genuine": 104, "synthetic": 104}
    assert set(meta["selected"]["by_attack"].values()) == {8} and len(meta["selected"]["by_attack"]) == 13
    assert len({r["clip_id"] for r in sel}) == 208
    # speaker spread: round-robin means max - min per-speaker genuine count <= 1 when enough clips
    from collections import Counter
    cnt = Counter(r["speaker_id"] for r in sel if r["label"] == "genuine")
    avail = Counter(r["speaker_id"] for r in recs if r["label"] == "genuine")
    top = max(cnt.values())
    assert all(cnt[s] >= top - 1 for s in avail if avail[s] >= top), (cnt, avail)
    # reproducible and independent of input order
    shuffled = recs[:]
    random.Random(99).shuffle(shuffled)
    sel2, meta2 = el.select_subset(shuffled, "asvspoof2019_la_eval", 20261025, 104, n_per_attack=8)
    assert [r["clip_id"] for r in sel] == [r["clip_id"] for r in sel2]
    assert meta["clip_id_list_sha256"] == meta2["clip_id_list_sha256"]
    # a different seed gives a different sample
    sel3, _ = el.select_subset(recs, "asvspoof2019_la_eval", 1, 104, n_per_attack=8)
    assert [r["clip_id"] for r in sel] != [r["clip_id"] for r in sel3]
    # shortfall reported, not padded
    few = [r for r in recs if r["attack_id"] != "A07" or r["clip_id"] < "LA_E_1000300"]
    _, m4 = el.select_subset(few[:400], "asvspoof2019_la_eval", 20261025, 104, n_per_attack=8)
    assert m4["shortfalls"], m4["shortfalls"]
    # detector outputs may not influence selection
    try:
        el.select_subset([dict(recs[0], clip_score_mean=0.3)] + recs[1:], "x", 1, 5, n_synthetic=5)
        raise AssertionError("score column accepted")
    except AssertionError as e:
        assert "detector-derived" in str(e)
    # held-out: exclude a previous pilot's clips and speakers; sample must be disjoint
    pilot_ids = {r["clip_id"] for r in sel}
    pilot_spk = {r["speaker_id"] for r in sel[:40]}
    sel5, m5 = el.select_subset(recs, "asvspoof2019_la_eval", 7, 50, n_per_attack=4,
                                exclude_clip_ids=pilot_ids, exclude_speakers=pilot_spk)
    assert not ({r["clip_id"] for r in sel5} & pilot_ids) and not ({r["speaker_id"] for r in sel5} & pilot_spk)
    assert m5["exclusion"]["records_after"] < m5["exclusion"]["records_before"]
    full, mf = el.select_subset(recs, "asvspoof2019_la_eval", 20261025, 0, full=True)
    assert len(full) == len(recs) and mf["full"]


def test_range_zip():
    buf = io.BytesIO()
    payload = {f"LA/ASVspoof2019_LA_eval/flac/LA_E_{i}.flac": os.urandom(5000 + i) for i in range(50)}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", PROTO)
        z.writestr("__MACOSX/LA/ASVspoof2019_LA_cm_protocols/._ASVspoof2019.LA.cm.eval.trl.txt", b"junk")
        for k, v in payload.items():
            z.writestr(k, v)
    blob = buf.getvalue()
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        return blob[a:b + 1]

    raw = el.HTTPRangeFile(size=len(blob), fetch=fetch)
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 12))
    names = zf.namelist()
    proto = el.find_member(names, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    assert "__MACOSX" not in proto
    assert zf.read(proto).decode() == PROTO
    k = "LA/ASVspoof2019_LA_eval/flac/LA_E_7.flac"
    before = raw.bytes_fetched
    assert zf.read(k) == payload[k]
    assert raw.bytes_fetched - before < 3 * 4096 + len(payload[k])  # only that member was transferred
    # CRC check: corrupt one byte of member data -> zipfile must raise
    off = zf.getinfo(k).header_offset + 30 + len(k) + 100
    bad = blob[:off] + bytes([blob[off] ^ 0xFF]) + blob[off + 1:]
    zf2 = zipfile.ZipFile(io.BufferedReader(el.HTTPRangeFile(size=len(bad), fetch=lambda a, b: bad[a:b + 1])))
    try:
        zf2.read(k)
        raise AssertionError("corruption not detected")
    except zipfile.BadZipFile:
        pass


def test_reject_category():
    assert el.reject_category("too short: 1.20 s < 2.0 s minimum") == "too_short"
    assert el.reject_category("mostly silent: 0.40 s active signal < 1.0 s") == "mostly_silent"
    assert el.reject_category("silent: peak -60 dBFS < -50.0 dBFS") == "silent"
    assert el.reject_category("unreadable audio (soundfile: X)") == "unreadable"
    assert el.reject_category("audio contains NaN/Inf samples") == "non_finite"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all eval_lib tests passed")
