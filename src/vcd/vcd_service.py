"""Detector service: model lifecycle + serialised inference. No HTTP here.

Wraps the third-party Dhwani ONNX checkpoint via dhwani_infer (pinned repo,
revision and sha256 live there). Used by vcd_api (FastAPI) now and intended
for remote-call audio windows later (detect_waveform).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

import dhwani_infer as dh

ATTRIBUTION = {
    "model": dh.MODEL_REPO,
    "revision": dh.MODEL_REVISION,
    "sha256": dh.MODEL_SHA256,
    "licence": "MIT (per model card)",
    "authors": "ayush2635 (Dhwani, HCL Guvi Hackathon); not trained by this project",
    "upstream_code": "github.com/ayush2635/Dhwani-Deepfake-Audio-Detection-API@efd512c",
    "encoder": "facebook/wav2vec2-xls-r-300m (Apache-2.0)",
}
SCORE_NOTE = (
    "synthetic_score is the mean softmax output for the synthetic class over 4 s windows. "
    "It is uncalibrated: a model score, not a probability. The 0.5 threshold is a placeholder."
)
MODE_NOTE = (
    "train_peak is a provisional default (matches upstream training code); "
    "it has not been validated beyond a two-clip smoke test."
)


class ServiceBusy(Exception):
    """Another request held the inference slot longer than the wait limit."""


class ServiceNotReady(Exception):
    """Model failed to load or has not loaded yet."""


def resolve_model_path(allow_download: bool = False) -> str:
    """Pinned checkpoint path: VCD_MODEL_PATH, else the local HF cache.

    Downloads only if allow_download is set (once, at startup - never per request).
    """
    explicit = os.environ.get("VCD_MODEL_PATH")
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION,
                               local_files_only=True)
    except Exception:
        if not allow_download:
            raise FileNotFoundError(
                "pinned checkpoint not in local HF cache; set VCD_MODEL_PATH or VCD_ALLOW_DOWNLOAD=1"
            )
    return hf_hub_download(dh.MODEL_REPO, dh.MODEL_FILE, revision=dh.MODEL_REVISION)


def verify_checkpoint(path: str) -> str:
    """Size + sha256 check; reuses the notebook's verification marker when the file is unchanged."""
    size = os.path.getsize(path)
    if size != dh.MODEL_BYTES:
        raise RuntimeError(f"unexpected checkpoint size {size} != {dh.MODEL_BYTES}")
    marker = os.path.join(os.path.dirname(os.path.realpath(path)), ".dhwani_sha256_ok")
    stamp = f"{size}:{os.path.getmtime(path)}"
    if os.path.exists(marker) and open(marker).read() == stamp:
        return "previously verified"
    got = dh.sha256_file(path)
    if got != dh.MODEL_SHA256:
        raise RuntimeError(f"checkpoint sha256 mismatch: {got}")
    try:
        with open(marker, "w") as f:
            f.write(stamp)
    except OSError:
        pass
    return "verified now"


class DetectorService:
    def __init__(self, default_mode: str = "train_peak", allow_download: bool = False,
                 max_wait_s: float = 60.0):
        if default_mode not in dh.PREPROCESSING_MODES:
            raise ValueError(f"unknown preprocessing mode {default_mode!r}")
        self.default_mode = default_mode
        self.allow_download = allow_download
        self.max_wait_s = max_wait_s
        self._lock = threading.Lock()  # one inference at a time: avoids overlapping CPU work
        self.sess = None
        self.model_path = None
        self.load_error = None
        self.load_info = {}

    # ---- lifecycle --------------------------------------------------------------
    def load(self) -> None:
        """Load once. Errors are kept for /health instead of crashing the server."""
        t0 = time.perf_counter()
        try:
            self.model_path = resolve_model_path(self.allow_download)
            sha_status = verify_checkpoint(self.model_path)
            self.sess = dh.load_session(self.model_path, verify_sha=False)
            warm = (np.random.default_rng(0).standard_normal(dh.WIN) * 0.1).astype(np.float32)
            tw = time.perf_counter()
            self.sess.run(None, {self.sess.get_inputs()[0].name: warm[None]})
            self.load_info = {
                "sha256_check": sha_status,
                "load_s": round(time.perf_counter() - t0, 2),
                "warmup_window_ms": round((time.perf_counter() - tw) * 1000, 1),
                **dh.describe_session(self.sess),
            }
        except Exception as e:
            self.sess = None
            self.load_error = f"{type(e).__name__}: {e}"

    @property
    def ready(self) -> bool:
        return self.sess is not None

    def health(self) -> dict:
        return {
            "status": "ready" if self.ready else "not_ready",
            "model_ready": self.ready,
            "load_error": self.load_error,
            "default_preprocessing": self.default_mode,
            "preprocessing_modes": list(dh.PREPROCESSING_MODES),
            "busy": self._lock.locked(),
            "model": ATTRIBUTION,
            "load_info": {k: self.load_info[k] for k in
                          ("sha256_check", "load_s", "warmup_window_ms", "session_providers")
                          if k in self.load_info},
        }

    # ---- inference ----------------------------------------------------------------
    def detect_file(self, path: str, mode: str | None = None) -> dict:
        mode = self._mode(mode)
        return self._serialised(lambda: dh.analyse(self.sess, path, mode=mode))

    def detect_waveform(self, samples: np.ndarray, sr: int, mode: str | None = None) -> dict:
        mode = self._mode(mode)
        return self._serialised(lambda: dh.analyse_waveform(self.sess, samples, sr, mode=mode))

    def _mode(self, mode: str | None) -> str:
        mode = mode or self.default_mode
        if mode not in dh.PREPROCESSING_MODES:
            raise ValueError(f"unknown preprocessing mode {mode!r}; use one of {list(dh.PREPROCESSING_MODES)}")
        return mode

    def _serialised(self, fn) -> dict:
        if not self.ready:
            raise ServiceNotReady(self.load_error or "model not loaded")
        tq = time.perf_counter()
        if not self._lock.acquire(timeout=self.max_wait_s):
            raise ServiceBusy(f"inference slot busy for more than {self.max_wait_s:.0f} s")
        queue_ms = round((time.perf_counter() - tq) * 1000, 1)
        acquired = time.monotonic()
        try:
            res = fn()
        finally:
            released = time.monotonic()
            self._lock.release()
        out = to_response(res)
        out["timing_ms"]["queue_wait"] = queue_ms
        out["inference_slot"] = {"acquired_monotonic": round(acquired, 4), "released_monotonic": round(released, 4)}
        return out


def to_response(res: dh.Result) -> dict:
    ok = res.status == "ok"
    info = res.info or {}
    return {
        "status": res.status,
        "prediction": res.predicted_label if ok else None,
        "synthetic_score": res.clip_score_mean,
        "synthetic_score_max": res.clip_score_max,
        "score_is_calibrated": False,
        "decision_threshold": dh.DECISION_THRESHOLD,
        "threshold_is_tuned": False,
        "score_note": SCORE_NOTE,
        "reject_reason": res.reject_reason or None,
        "duration_s": info.get("duration_s"),
        "analysed_duration_s": info.get("analysed_duration_s"),
        "truncated": info.get("truncated"),
        "audio": {
            "native_sample_rate": res.native_sr,
            "channels": res.channels,
            "peak_dbfs": info.get("peak_dbfs"),
            "rms_dbfs": info.get("rms_dbfs"),
            "active_s": info.get("active_s"),
        },
        "windows": [{"start_s": s, "synthetic_score": w}
                    for s, w in zip(res.window_starts_s, res.window_scores)],
        "preprocessing": res.preprocessing,
        "preprocessing_note": MODE_NOTE,
        "timing_ms": {"model": res.latency_ms_model, "pipeline": res.latency_ms_total},
        "model": {"repo": dh.MODEL_REPO, "revision": dh.MODEL_REVISION},
    }
