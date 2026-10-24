"""Helpers for the Dhwani generalisation evaluation (evaluation only, no training).

Standard library only, so the selection, archive access and metric code can be
unit-tested without ML dependencies (evaluation/tests/test_eval_lib.py).
Model execution, preprocessing and windowing are NOT reimplemented here: the
notebook calls src/vcd/dhwani_infer.py (embedded verbatim, read-only).

Labels follow dhwani_infer.LABELS: "genuine" (0) / "synthetic" (1).
Scores are uncalibrated synthetic-class softmax means, not probabilities.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

# ---- pinned dataset sources (verified 2026-10-24, see evaluation/README.md) --------
SOURCES = {
    "asvspoof2019_la_eval": {
        "title": "ASVspoof 2019 Logical Access (LA), evaluation partition",
        "landing_page": "https://datashare.ed.ac.uk/handle/10283/3336",
        "doi": "10.7488/ds/2555",
        "archive": "LA.zip",
        "url": "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
               "a9f87c35-f055-4015-80e2-2fdff0d46269/content",
        "bytes": 7640952520,
        "md5": "30c98f11d8b2bc21f2c257bfd78bb5c5",  # DSpace checkSum (MD5) == HTTP ETag
        "last_modified": "Tue, 04 Jun 2019 09:38:20 GMT",
        "license": "ODC-By 1.0 (archive README.txt, COPYING section)",
        "protocol_suffix": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir_suffix": "ASVspoof2019_LA_eval/flac/",
        "audio_ext": ".flac",
        # Literature values, checked against the protocol in the notebook (warn on mismatch).
        "expected_counts": {"genuine": 7355, "synthetic": 63882},
        "expected_attacks": [f"A{i:02d}" for i in range(7, 20)],
    },
    "in_the_wild": {
        "title": "In-the-Wild audio deepfake dataset (Mueller et al., Interspeech 2022)",
        "landing_page": "https://huggingface.co/datasets/mueller91/In-The-Wild",
        "project_page": "https://deepfake-total.com/in_the_wild",
        "hf_repo": "mueller91/In-The-Wild",
        "revision": "eee168f92c367f8c82ff2cf42b6f61e362fd6211",
        "archive": "release_in_the_wild.zip",
        "url": "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/"
               "eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip",
        "bytes": 8161489023,
        "sha256": "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a",  # LFS oid
        "xet_hash": "ec530d2076f973d40a8e388e559888f1b4fea467db8126c811d34dd508ee106f",  # CDN ETag
        "license": "CONFLICT: HF card says CC-BY-SA-4.0; Fraunhofer project page says Apache-2.0",
        "meta_suffix": "meta.csv",
        "expected_counts": {"genuine": 19963, "synthetic": 11816},
    },
}

GENUINE, SYNTHETIC = "genuine", "synthetic"


# ---- remote zip access via HTTP range requests -------------------------------------
class RangeUnsupported(RuntimeError):
    """Server ignored the Range header (would have streamed the whole archive)."""


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests.

    zipfile.ZipFile can open it directly, so only the central directory and the
    selected members are transferred. zipfile checks each member's CRC-32 on read.
    `fetch(start, end_inclusive) -> bytes` can be injected for tests.
    """

    def __init__(self, url: str | None = None, size: int | None = None, fetch=None,
                 timeout: float = 60.0, retries: int = 4):
        super().__init__()
        self.url, self.timeout, self.retries = url, timeout, retries
        self._final_url = None
        self._fetch = fetch or self._http_fetch
        self.headers = {}
        if size is None:
            size = self._probe_size()
        self.size, self.pos = int(size), 0
        self.n_requests, self.bytes_fetched = 0, 0

    def _resolve(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise RangeUnsupported(f"HTTP {r.status} to a Range request: {self.url}")
            self._final_url = r.geturl()
            self.headers = {k.lower(): v for k, v in r.headers.items()}
            r.read()
        return self.headers

    def _probe_size(self) -> int:
        h = self._resolve()
        return int(h["content-range"].rsplit("/", 1)[1])

    def _http_fetch(self, start: int, end: int) -> bytes:
        last = None
        for attempt in range(self.retries):
            if self._final_url is None:
                self._resolve()
            req = urllib.request.Request(self._final_url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise RangeUnsupported(f"HTTP {r.status} to a Range request")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end - start + 1}")
                return data
            except RangeUnsupported:
                raise
            except (urllib.error.URLError, IOError, TimeoutError) as e:
                last = e
                self._final_url = None  # signed redirect URLs expire: resolve again
                time.sleep(2 ** attempt)
        raise IOError(f"range fetch {start}-{end} failed after {self.retries} attempts: {last}")

    # RawIOBase interface
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        if self.pos < 0:
            raise ValueError("negative seek position")
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._fetch(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        self.n_requests += 1
        self.bytes_fetched += len(data)
        return len(data)


def open_remote_zip(url: str, expected_bytes: int | None = None, buffer_size: int = 1 << 18):
    """Return (ZipFile, raw HTTPRangeFile) for a remote archive."""
    import zipfile

    raw = HTTPRangeFile(url)
    if expected_bytes is not None and raw.size != expected_bytes:
        raise IOError(f"remote size {raw.size} != recorded {expected_bytes}")
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=buffer_size))
    return zf, raw


def find_member(names, suffix: str) -> str:
    hits = [n for n in names if n.endswith(suffix) and not n.split("/")[-1].startswith("._")]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one member ending in {suffix!r}, found {hits[:5]}")
    return hits[0]


def file_digest(path: str, algo: str = "sha256", chunk: int = 1 << 22) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---- metadata parsing ------------------------------------------------------------
def parse_asvspoof_cm_protocol(text: str) -> list[dict]:
    """ASVspoof 2019 LA CM protocol: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"protocol line {i}: expected 5 columns, got {len(parts)}: {line!r}")
        spk, utt, _unused, system, key = parts
        if key not in ("bonafide", "spoof"):
            raise ValueError(f"protocol line {i}: unknown key {key!r}")
        if key == "spoof" and system == "-":
            raise ValueError(f"protocol line {i}: spoof without attack ID")
        out.append({
            "clip_id": utt, "speaker_id": spk,
            "attack_id": system if key == "spoof" else "",
            "label": SYNTHETIC if key == "spoof" else GENUINE,
            "original_label": key,
        })
    return out


ITW_LABELS = {"bona-fide": GENUINE, "bonafide": GENUINE, "spoof": SYNTHETIC}


def parse_itw_meta(text: str) -> list[dict]:
    """In-the-Wild meta.csv (expected columns: file, speaker, label)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("meta.csv is empty")
    cols = {c.strip().lower(): c for c in rows[0].keys()}
    missing = {"file", "speaker", "label"} - set(cols)
    if missing:
        raise ValueError(f"meta.csv lacks columns {missing}; has {list(rows[0].keys())}")
    out = []
    for i, r in enumerate(rows, 2):
        lab = r[cols["label"]].strip()
        if lab.lower() not in ITW_LABELS:
            raise ValueError(f"meta.csv line {i}: unknown label {lab!r}")
        fn = r[cols["file"]].strip()
        out.append({
            "clip_id": fn, "speaker_id": r[cols["speaker"]].strip(), "attack_id": "",
            "label": ITW_LABELS[lab.lower()], "original_label": lab,
        })
    return out


def describe_records(records: list[dict]) -> dict:
    by_label = Counter(r["label"] for r in records)
    attacks = Counter(r["attack_id"] for r in records if r["label"] == SYNTHETIC and r["attack_id"])
    spk = defaultdict(Counter)
    for r in records:
        spk[r["label"]][r["speaker_id"]] += 1
    return {
        "n": len(records),
        "by_label": dict(sorted(by_label.items())),
        "by_attack": dict(sorted(attacks.items())),
        "n_speakers_by_label": {k: len(v) for k, v in sorted(spk.items())},
        "n_speakers_total": len({r["speaker_id"] for r in records}),
        "duplicate_clip_ids": len(records) - len({r["clip_id"] for r in records}),
    }


# ---- score-free, reproducible selection ------------------------------------------
FORBIDDEN_SELECTION_KEYS = ("score", "logit", "pred", "prob")


def _rng(seed: int, *stratum) -> random.Random:
    # str seeds are hashed with SHA-512 by random.seed (version 2): stable across
    # runs, machines and PYTHONHASHSEED. One RNG per stratum => order-independent.
    return random.Random(":".join([str(seed), *map(str, stratum)]))


def speaker_round_robin(records: list[dict], n: int, seed: int, *stratum) -> list[dict]:
    """Take n clips spreading over speakers: shuffled speaker order, shuffled clips per
    speaker, then one clip per speaker per round. Uses metadata only."""
    by_spk = defaultdict(list)
    for r in sorted(records, key=lambda r: r["clip_id"]):
        by_spk[r["speaker_id"]].append(r)
    spks = sorted(by_spk)
    _rng(seed, *stratum, "speakers").shuffle(spks)
    for s in spks:
        _rng(seed, *stratum, "clips", s).shuffle(by_spk[s])
    out, rnd = [], 0
    while len(out) < n and any(len(by_spk[s]) > rnd for s in spks):
        for s in spks:
            if len(out) >= n:
                break
            if len(by_spk[s]) > rnd:
                out.append(by_spk[s][rnd])
        rnd += 1
    return out


def select_subset(records: list[dict], dataset: str, seed: int, n_genuine: int,
                  n_synthetic: int | None = None, n_per_attack: int | None = None,
                  full: bool = False, exclude_clip_ids=(), exclude_speakers=()) -> tuple[list[dict], dict]:
    """Score-free stratified sample. full=True: every record (sorted by ID).

    exclude_clip_ids / exclude_speakers: clips already used as development
    evidence (e.g. an earlier pilot) are removed BEFORE sampling, so a fresh
    held-out subset never shares clips (or speakers) with them.
    """
    for r in records[:1]:
        bad = [k for k in r if any(f in k.lower() for f in FORBIDDEN_SELECTION_KEYS)]
        assert not bad, f"selection input contains detector-derived fields: {bad}"
    ex_ids, ex_spk = set(exclude_clip_ids), set(exclude_speakers)
    n_before = len(records)
    records = [r for r in records if r["clip_id"] not in ex_ids and r["speaker_id"] not in ex_spk]
    exclusion = {"excluded_clip_ids": len(ex_ids), "excluded_speakers": len(ex_spk),
                 "records_before": n_before, "records_after": len(records)}
    gen = [r for r in records if r["label"] == GENUINE]
    syn = [r for r in records if r["label"] == SYNTHETIC]
    if full:
        sel = [dict(r, stratum="all") for r in sorted(records, key=lambda r: r["clip_id"])]
        rule = "full evaluation partition: every protocol/metadata row, sorted by clip_id"
    else:
        sel = [dict(r, stratum="genuine") for r in speaker_round_robin(gen, n_genuine, seed, dataset, GENUINE)]
        if n_per_attack is not None:
            by_atk = defaultdict(list)
            for r in syn:
                by_atk[r["attack_id"]].append(r)
            for atk in sorted(by_atk):
                take = speaker_round_robin(by_atk[atk], n_per_attack, seed, dataset, SYNTHETIC, atk)
                sel += [dict(r, stratum=f"synthetic/{atk}") for r in take]
            rule = (f"genuine: {n_genuine} by speaker round-robin; synthetic: {n_per_attack} per attack ID "
                    f"({len(by_atk)} attacks) by speaker round-robin within attack")
        else:
            sel += [dict(r, stratum="synthetic") for r in speaker_round_robin(syn, n_synthetic, seed, dataset, SYNTHETIC)]
            rule = f"genuine: {n_genuine}, synthetic: {n_synthetic}, each by speaker round-robin"
        rule += (f"; seed {seed}; per-stratum random.Random('{seed}:{dataset}:<label>[:<attack>]:...'); "
                 "input sorted by clip_id; metadata only (no audio, no detector output)")
    for i, r in enumerate(sel):
        r["selection_rank"] = i
        r["dataset"] = dataset
    ids = "\n".join(r["clip_id"] for r in sel)
    meta = {
        "dataset": dataset, "seed": seed, "full": full, "rule": rule,
        "n_selected": len(sel), "selected": describe_records(sel), "available": describe_records(records),
        "clip_id_list_sha256": hashlib.sha256(ids.encode()).hexdigest(),
        "shortfalls": {}, "exclusion": exclusion,
    }
    assert not ({r["clip_id"] for r in sel} & ex_ids) and not ({r["speaker_id"] for r in sel} & ex_spk)
    if not full:
        got = Counter(r["stratum"] for r in sel)
        if got["genuine"] < n_genuine:
            meta["shortfalls"]["genuine"] = f"{got['genuine']} < {n_genuine}"
        for k, v in got.items():
            if k.startswith("synthetic/") and n_per_attack and v < n_per_attack:
                meta["shortfalls"][k] = f"{v} < {n_per_attack}"
    return sel, meta


# ---- metrics (positive class = synthetic; higher score = more synthetic) --------
def _split(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    return pos, neg


def roc_auc(labels, scores):
    """Mann-Whitney U / (n_pos * n_neg), ties counted 1/2. None if a class is empty."""
    pos, neg = _split(labels, scores)
    if not pos or not neg:
        return None
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum_pos, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # ranks i+1..j
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def eer(labels, scores):
    """EER with the ASVspoof 2019 reference convention (compute_eer in the official
    tDCF package): sweep thresholds over all scores, take the point minimising
    |FNR - FPR| and report their mean. Decision: synthetic if score >= t.
    Returns (eer, threshold_at_eer) or (None, None). The EER threshold is
    descriptive only; it is NOT an operating point for this project."""
    pos, neg = _split(labels, scores)
    if not pos or not neg:
        return None, None
    pts = sorted(set(pos) | set(neg)) + [math.inf]  # inf: FNR = 1, FPR = 0
    pos_s, neg_s = sorted(pos), sorted(neg)
    best = None
    ip = ineg = 0
    for t in pts:  # FNR(t) = #pos < t / P ; FPR(t) = #neg >= t / N
        while ip < len(pos_s) and pos_s[ip] < t:
            ip += 1
        while ineg < len(neg_s) and neg_s[ineg] < t:
            ineg += 1
        fnr, fpr = ip / len(pos_s), (len(neg_s) - ineg) / len(neg_s)
        d = abs(fnr - fpr)
        if best is None or d < best[0]:
            best = (d, (fnr + fpr) / 2.0, t)
    return best[1], best[2]


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def at_threshold(labels, scores, thr: float) -> dict:
    tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= thr)
    fn = sum(1 for l, s in zip(labels, scores) if l == 1 and s < thr)
    fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= thr)
    tn = sum(1 for l, s in zip(labels, scores) if l == 0 and s < thr)
    P, N = tp + fn, fp + tn
    rec = tp / P if P else None
    fpr = fp / N if N else None
    spec = tn / N if N else None
    return {
        "threshold": thr,
        "confusion": {"tp_synthetic_as_synthetic": tp, "fn_synthetic_as_genuine": fn,
                      "fp_genuine_as_synthetic": fp, "tn_genuine_as_genuine": tn},
        "synthetic_recall": rec, "synthetic_recall_wilson95": wilson(tp, P),
        "genuine_fpr": fpr, "genuine_fpr_wilson95": wilson(fp, N),
        "genuine_specificity": spec,
        "balanced_accuracy": (rec + spec) / 2 if rec is not None and spec is not None else None,
        "accuracy": (tp + tn) / (P + N) if P + N else None,
        "n_synthetic": P, "n_genuine": N,
    }


def bootstrap_ci(labels, scores, thr: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Stratified (within-class) percentile bootstrap, 95 %. Reflects clip sampling
    only; it ignores speaker clustering, so intervals are optimistic."""
    pos, neg = _split(labels, scores)
    if not pos or not neg or n_boot <= 0:
        return {}
    rng = random.Random(f"bootstrap:{seed}")
    acc = defaultdict(list)
    for _ in range(n_boot):
        p = [pos[rng.randrange(len(pos))] for _ in pos]
        q = [neg[rng.randrange(len(neg))] for _ in neg]
        lab, sc = [1] * len(p) + [0] * len(q), p + q
        acc["roc_auc"].append(roc_auc(lab, sc))
        acc["eer"].append(eer(lab, sc)[0])
        acc["balanced_accuracy@thr"].append(at_threshold(lab, sc, thr)["balanced_accuracy"])
    out = {}
    for k, v in acc.items():
        v = sorted(v)
        out[k] = (v[int(0.025 * (len(v) - 1))], v[int(math.ceil(0.975 * (len(v) - 1)))])
    out["n_boot"], out["seed"] = n_boot, seed
    return out


def evaluate(labels, scores, thr: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """labels: 1 = synthetic, 0 = genuine (scored clips only)."""
    e, et = eer(labels, scores)
    return {
        "n_scored": len(labels),
        "threshold_free": {"roc_auc": roc_auc(labels, scores), "eer": e,
                           "eer_threshold_descriptive_only": et},
        "at_threshold": at_threshold(labels, scores, thr),
        "bootstrap95": bootstrap_ci(labels, scores, thr, n_boot, seed),
    }


REJECT_CATEGORIES = (
    ("too short", "too_short"), ("mostly silent", "mostly_silent"), ("silent", "silent"),
    ("unreadable", "unreadable"), ("empty", "empty"), ("NaN", "non_finite"),
)


def reject_category(reason: str) -> str:
    for needle, cat in REJECT_CATEGORIES:
        if reason.startswith(needle) or needle in reason[:40]:
            return cat
    return "other"


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
