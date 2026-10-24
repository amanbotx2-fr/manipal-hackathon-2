"""Inference wrapper for the third-party Dhwani ONNX checkpoint.

Model: ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model (MIT),
file best_model.onnx at HF revision MODEL_REVISION. We did NOT train it.

Preprocessing is reproduced from the upstream training notebook
(github.com/ayush2635/Dhwani-Deepfake-Audio-Detection-API @ efd512c,
training/training_notebook.ipynb cell 4), which is the notebook whose cell 14
exports best_model.onnx with a (1, 64000) dummy input. See
docs/UPSTREAM_AUDIT.md for why we do not use the model card's z-score recipe
as the default.

Scores are softmax outputs of an UNCALIBRATED classifier. They are model
scores, not probabilities that a clip is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np

# ---- pinned provenance -----------------------------------------------------
MODEL_REPO = "ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model"
MODEL_FILE = "best_model.onnx"
MODEL_REVISION = "ee0187142973e1f02050e47d2a31c55ee0263065"
MODEL_SHA256 = "d1c232bf4d7990526804b375c15804c0ee4e9b566478b908c5a57431491b7342"
MODEL_BYTES = 1262760439

# ---- preprocessing / windowing policy ---------------------------------------
SR = 16000
WIN = 64000            # 4.0 s: training crop length and ONNX export shape
HOP = 32000            # 2.0 s: same hop as the upstream API's sliding window
MIN_DUR_S = 2.0        # brief: 2-10 s clips
MAX_DUR_S = 10.0       # longer clips: analyse the first 10 s, flagged
PEAK_TARGET_DBFS = -3.0

# ---- rejection gates (input validation only; never fed to the model) --------
SILENT_PEAK_DBFS = -50.0   # whole clip quieter than this -> silent
ACTIVE_FRAME_S = 0.025
ACTIVE_REL_DB = 35.0       # frame counts as active if within 35 dB of loudest frame
ACTIVE_ABS_DBFS = -55.0    # ...and above this absolute level
MIN_ACTIVE_S = 1.0

# Class order from upstream dataset.py: labels = [0]*real + [1]*fake.
# Documented in source; must still be confirmed empirically on known clips.
LABELS = {0: "genuine", 1: "synthetic"}
SYNTHETIC_INDEX = 1
DECISION_THRESHOLD = 0.5   # placeholder, NOT tuned on any validation set

# Controlled comparison (docs/UPSTREAM_AUDIT.md):
#   train_peak   - training recipe: peak -3 dBFS, 4 s windows             (A)
#   train_zscore - identical windows, z-score instead of peak norm        (A vs B isolates normalisation)
#   api_replica  - deployed API: first 3 s, z-score, zero-pad to 4 s     (B vs C isolates crop/padding)
PREPROCESSING_MODES = ("train_peak", "train_zscore", "api_replica")


class AudioRejected(Exception):
    """Input cannot be analysed (unreadable, empty, silent, too short)."""


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _dbfs(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


# ---- loading ------------------------------------------------------------------
def load_audio(path: str):
    """Return (samples [channels, n] float32 at native rate, native_sr).

    Tries soundfile first (wav/flac/ogg), then librosa/audioread (mp3, m4a, opus
    via ffmpeg). Never substitutes zeros on failure.
    """
    import soundfile as sf

    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
        y = y.T
    except Exception as sf_err:
        try:
            import warnings

            import librosa

            with warnings.catch_warnings():  # expected audioread fallback for m4a/aac
                warnings.simplefilter("ignore", (UserWarning, FutureWarning))
                y, sr = librosa.load(path, sr=None, mono=False)
            y = np.atleast_2d(y).astype(np.float32)
        except Exception as lr_err:
            raise AudioRejected(
                f"unreadable audio (soundfile: {type(sf_err).__name__}; "
                f"librosa: {type(lr_err).__name__}: {lr_err})"
            ) from lr_err
    if y.size == 0 or y.shape[-1] == 0:
        raise AudioRejected("empty audio (0 samples)")
    if not np.all(np.isfinite(y)):
        raise AudioRejected("audio contains NaN/Inf samples")
    return y, int(sr)


def to_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
    mono = y.mean(axis=0) if y.shape[0] > 1 else y[0]
    if sr != SR:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return mono.astype(np.float32)


def signal_stats(x: np.ndarray) -> dict:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    flen = int(ACTIVE_FRAME_S * SR)
    nfr = len(x) // flen
    if nfr:
        fr = x[: nfr * flen].reshape(nfr, flen).astype(np.float64)
        fdb = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-20)
        active = (fdb > fdb.max() - ACTIVE_REL_DB) & (fdb > ACTIVE_ABS_DBFS)
        active_s = float(active.sum() * ACTIVE_FRAME_S)
    else:
        active_s = 0.0
    return {
        "peak_dbfs": round(_dbfs(peak), 2),
        "rms_dbfs": round(_dbfs(rms), 2),
        "active_s": round(active_s, 3),
        "clipping_frac": round(float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0, 5),
    }


def validate(x16: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply the 2-10 s policy and silence gates. Returns (analysed audio, info)."""
    dur = len(x16) / SR
    info = {"duration_s": round(dur, 3), "truncated": False}
    if dur < MIN_DUR_S:
        raise AudioRejected(f"too short: {dur:.2f} s < {MIN_DUR_S} s minimum")
    if dur > MAX_DUR_S:
        x16 = x16[: int(MAX_DUR_S * SR)]
        info["truncated"] = True
    info["analysed_duration_s"] = round(len(x16) / SR, 3)
    st = signal_stats(x16)
    info.update(st)
    if st["peak_dbfs"] < SILENT_PEAK_DBFS:
        raise AudioRejected(f"silent: peak {st['peak_dbfs']} dBFS < {SILENT_PEAK_DBFS} dBFS")
    if st["active_s"] < MIN_ACTIVE_S:
        raise AudioRejected(
            f"mostly silent: {st['active_s']:.2f} s active signal < {MIN_ACTIVE_S} s"
        )
    return x16, info


# ---- preprocessing --------------------------------------------------------------
def window_starts(n: int) -> list[int]:
    """4 s windows, 2 s hop, last window aligned to the clip end (full coverage)."""
    if n <= WIN:
        return [0]
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return starts


def make_windows(x16: np.ndarray, mode: str) -> tuple[np.ndarray, list[int]]:
    """Return (windows [k, WIN] float32, start sample indices)."""
    if mode in ("train_peak", "train_zscore"):
        if mode == "train_peak":
            # Training notebook: peak-normalise the clip to -3 dBFS, then crop/zero-pad
            # the END to 64000 samples. No mean/variance normalisation.
            peak = float(np.max(np.abs(x16)))
            x = x16 * ((10 ** (PEAK_TARGET_DBFS / 20)) / (peak + 1e-6))
        else:
            # Same windows; z-score over the analysed clip (before padding).
            x = (x16 - x16.mean()) / np.sqrt(x16.var() + 1e-5)
        starts = window_starts(len(x))
        wins = []
        for s in starts:
            w = x[s : s + WIN]
            if len(w) < WIN:
                w = np.pad(w, (0, WIN - len(w)))
            wins.append(w)
        return np.stack(wins).astype(np.float32), starts
    if mode == "api_replica":
        # Deployed upstream API (src/api/inference.py): first 48000 samples,
        # zero-pad to 48000, z-score, then zero-pad to 64000 in predict().
        # Kept only as a comparison; it differs from training.
        w = x16[:48000]
        w = np.pad(w, (0, 48000 - len(w)))
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-5)
        w = np.pad(w, (0, WIN - len(w)))
        return w[None, :].astype(np.float32), [0]
    raise ValueError(f"unknown preprocessing mode {mode!r}; use {PREPROCESSING_MODES}")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---- model ------------------------------------------------------------------------
def load_session(model_path: str, verify_sha: bool = True):
    import onnxruntime as ort

    if verify_sha:
        got = sha256_file(model_path)
        if got != MODEL_SHA256:
            raise RuntimeError(f"model sha256 mismatch: {got} != pinned {MODEL_SHA256}")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess


def describe_session(sess) -> dict:
    import onnxruntime as ort

    return {
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "session_providers": sess.get_providers(),
        "inputs": [(i.name, i.shape, i.type) for i in sess.get_inputs()],
        "outputs": [(o.name, o.shape, o.type) for o in sess.get_outputs()],
    }


def interface_self_test(sess, seed: int = 0) -> dict:
    """Probe what the checkpoint actually accepts (metadata alone proves nothing)."""
    rng = np.random.default_rng(seed)
    name = sess.get_inputs()[0].name
    out = {}
    base = (rng.standard_normal(WIN) * 0.1).astype(np.float32)
    ref = sess.run(None, {name: base[None]})[0]
    out["len_64000_batch1"] = {"ok": True, "shape": list(ref.shape)}
    for n in (32000, 48000, 96000):
        try:
            r = sess.run(None, {name: base[:n][None] if n <= WIN else np.tile(base, 2)[:n][None]})[0]
            out[f"len_{n}_batch1"] = {"ok": True, "shape": list(r.shape)}
        except Exception as e:
            out[f"len_{n}_batch1"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    try:
        b = np.stack([base, base[::-1].copy()])
        r = sess.run(None, {name: b})[0]
        r1 = sess.run(None, {name: base[::-1].copy()[None]})[0]
        # Colab run 20261024T102407Z: batch of 2 returned shape [1, 2] (a row is
        # silently dropped), so "ok" requires one output row per input row.
        out["batch2"] = {
            "ok": r.shape[0] == 2,
            "shape": list(r.shape),
            "max_abs_diff_vs_individual": float(
                max(np.abs(r[0] - ref[0]).max(), np.abs(r[1] - r1[0]).max())
            ) if r.shape[0] == 2 else None,
        }
    except Exception as e:
        out["batch2"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    # determinism
    again = sess.run(None, {name: base[None]})[0]
    out["deterministic"] = bool(np.allclose(again, ref))
    return out


@dataclass
class Result:
    file: str
    status: str = "ok"
    reject_reason: str = ""
    file_sha256: str = ""
    native_sr: int | None = None
    channels: int | None = None
    info: dict = field(default_factory=dict)
    preprocessing: str = ""
    window_starts_s: list = field(default_factory=list)
    window_logits: list = field(default_factory=list)
    window_scores: list = field(default_factory=list)
    clip_score_mean: float | None = None
    clip_score_max: float | None = None
    predicted_label: str = ""
    latency_ms_model: float | None = None
    latency_ms_total: float | None = None


def analyse(sess, path: str, mode: str = "train_peak") -> Result:
    """Score an audio file (any format load_audio can decode)."""
    t0 = time.perf_counter()
    res = Result(file=path, preprocessing=mode)
    try:
        res.file_sha256 = sha256_file(path)
        y, sr = load_audio(path)
    except AudioRejected as e:
        res.status, res.reject_reason = "rejected", str(e)
        res.latency_ms_total = round((time.perf_counter() - t0) * 1000, 1)
        return res
    return _score(sess, res, y, sr, t0)


def analyse_waveform(sess, samples: np.ndarray, sr: int, mode: str = "train_peak",
                     label: str = "<waveform>") -> Result:
    """Score in-memory PCM ([n] or [channels, n], float in [-1, 1]).

    Same validation, preprocessing and windowing as analyse(); intended for
    audio windows that never touch disk (e.g. remote-call PCM later).
    """
    t0 = time.perf_counter()
    res = Result(file=label, preprocessing=mode)
    y = np.atleast_2d(np.asarray(samples, dtype=np.float32))
    if y.size == 0:
        res.status, res.reject_reason = "rejected", "empty audio (0 samples)"
    elif not np.all(np.isfinite(y)):
        res.status, res.reject_reason = "rejected", "audio contains NaN/Inf samples"
    if res.status == "rejected":
        res.latency_ms_total = round((time.perf_counter() - t0) * 1000, 1)
        return res
    return _score(sess, res, y, int(sr), t0)


def _score(sess, res: Result, y: np.ndarray, sr: int, t0: float) -> Result:
    try:
        res.native_sr, res.channels = sr, int(y.shape[0])
        x16 = to_mono_16k(y, sr)
        x16, info = validate(x16)
        res.info = info
        wins, starts = make_windows(x16, res.preprocessing)
    except AudioRejected as e:
        res.status, res.reject_reason = "rejected", str(e)
        res.latency_ms_total = round((time.perf_counter() - t0) * 1000, 1)
        return res

    name = sess.get_inputs()[0].name
    logits = []
    tm = time.perf_counter()
    for w in wins:  # batch=1 only: batch>1 drops rows (see interface_self_test)
        logits.append(sess.run(None, {name: w[None]})[0][0])
    res.latency_ms_model = round((time.perf_counter() - tm) * 1000, 1)
    logits = np.asarray(logits, dtype=np.float64)
    scores = softmax(logits)[:, SYNTHETIC_INDEX]
    res.window_starts_s = [round(s / SR, 3) for s in starts]
    res.window_logits = [[round(float(a), 4) for a in row] for row in logits]
    res.window_scores = [round(float(s), 4) for s in scores]
    res.clip_score_mean = round(float(scores.mean()), 4)
    res.clip_score_max = round(float(scores.max()), 4)
    res.predicted_label = LABELS[SYNTHETIC_INDEX] if res.clip_score_mean >= DECISION_THRESHOLD else LABELS[0]
    res.latency_ms_total = round((time.perf_counter() - t0) * 1000, 1)
    return res


def result_row(res: Result, **extra) -> dict:
    d = asdict(res)
    info = d.pop("info")
    for k in ("window_starts_s", "window_logits", "window_scores"):
        d[k] = json.dumps(d[k])
    row = {**extra, **d, **info}
    row.update(
        decision_threshold=DECISION_THRESHOLD,
        label_mapping=json.dumps(LABELS),
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        model_sha256=MODEL_SHA256,
    )
    return row
