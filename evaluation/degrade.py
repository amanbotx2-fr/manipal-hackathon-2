"""Paired audio degradations for the robustness evaluation (evaluation only).

Every function takes a mono 16 kHz float32 waveform (already decoded and validated by
dhwani_infer) and returns a mono 16 kHz float32 waveform of the same clip, so the comparison
between conditions is paired per clip. Codecs go through ffmpeg (Colab ships it); their exact
settings are fixed here and recorded in CONDITIONS. Noise is white Gaussian at a documented
SNR with a per-clip seed derived from (global seed, clip_id), so it is reproducible.

Required ("official ratio") conditions:  clean, narrowband_8k, codec_1 (Opus 16 kb/s),
codec_2 (MP3 32 kb/s), noise (white, 10 dB SNR).
Sensitivity checks (reported separately): loud_minus20db, loud_peak_minus1db, silence_pad_1s,
duration_first3s, telephone_quality (300-3400 Hz band-pass + 8-bit mu-law).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

SR = 16000

CONDITIONS = {
    "clean": {"group": "required", "desc": "decoded 16 kHz mono, no change"},
    "narrowband_8k": {"group": "required", "desc": "resample 16 kHz -> 8 kHz -> 16 kHz, librosa soxr_hq both ways"},
    "codec_1": {"group": "required", "desc": "Opus (libopus) 16 kb/s VBR, application=voip, 16 kHz in/out via ffmpeg",
                "ffmpeg_encode": ["-c:a", "libopus", "-b:a", "16k", "-vbr", "on", "-application", "voip"], "ext": ".opus"},
    "codec_2": {"group": "required", "desc": "MP3 (libmp3lame) 32 kb/s CBR mono, 16 kHz via ffmpeg",
                "ffmpeg_encode": ["-c:a", "libmp3lame", "-b:a", "32k"], "ext": ".mp3"},
    "noise": {"group": "required", "desc": "additive white Gaussian noise at 10 dB SNR (SNR vs clip RMS), seed sha256(seed:clip_id)",
              "snr_db": 10.0},
    "loud_minus20db": {"group": "sensitivity", "desc": "gain -20 dB (quiet recording)"},
    "loud_peak_minus1db": {"group": "sensitivity", "desc": "peak-normalise to -1 dBFS (loud recording, no clipping)"},
    "silence_pad_1s": {"group": "sensitivity", "desc": "1.0 s of digital silence prepended and appended"},
    "duration_first3s": {"group": "sensitivity", "desc": "first 3.0 s only (clips shorter than 3 s unchanged, flagged)"},
    "telephone_quality": {"group": "sensitivity", "desc": "300-3400 Hz 4th-order Butterworth band-pass + 8-bit mu-law quantisation"},
}
NOISE_SEED = 20261027


def _rng(clip_id: str) -> np.random.Generator:
    h = hashlib.sha256(f"{NOISE_SEED}:{clip_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def narrowband_8k(x: np.ndarray, **_) -> np.ndarray:
    import librosa

    y = librosa.resample(x, orig_sr=SR, target_sr=8000, res_type="soxr_hq")
    y = librosa.resample(y, orig_sr=8000, target_sr=SR, res_type="soxr_hq")
    return _fit(y, len(x))


def ffmpeg_available() -> dict:
    """Return {'ffmpeg': path or None, 'libopus': bool, 'libmp3lame': bool}."""
    exe = shutil.which("ffmpeg")
    out = {"ffmpeg": exe, "libopus": False, "libmp3lame": False}
    if exe:
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        out["libopus"], out["libmp3lame"] = "libopus" in enc, "libmp3lame" in enc
    return out


def codec_roundtrip(x: np.ndarray, cond: str, **_) -> np.ndarray:
    import soundfile as sf

    spec = CONDITIONS[cond]
    d = tempfile.mkdtemp(prefix="vcd_codec_")
    try:
        src, enc, dec = os.path.join(d, "in.wav"), os.path.join(d, "enc" + spec["ext"]), os.path.join(d, "out.wav")
        sf.write(src, x, SR, subtype="PCM_16")
        for cmd in ([ "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, *spec["ffmpeg_encode"], "-ar", "16000", "-ac", "1", enc],
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", enc, "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le", dec]):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed ({cond}): {r.stderr.strip()[-300:]}")
        y, sr = sf.read(dec, dtype="float32", always_2d=True)
        assert sr == SR
        y = y.mean(axis=1)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # codecs add a few ms of delay/padding: align length to the source (crop/pad at the end) so the clip
    # stays paired; the encoder delay itself is part of what a real transmission would do.
    return _fit(y, len(x))


def add_noise(x: np.ndarray, clip_id: str = "", snr_db: float | None = None, **_) -> np.ndarray:
    snr_db = CONDITIONS["noise"]["snr_db"] if snr_db is None else snr_db
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-6
    noise = _rng(clip_id).standard_normal(len(x)).astype(np.float32)
    noise *= rms / (10 ** (snr_db / 20)) / (float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) or 1e-6)
    return (x + noise).astype(np.float32)


def loud_minus20db(x: np.ndarray, **_) -> np.ndarray:
    return (x * 10 ** (-20 / 20)).astype(np.float32)


def loud_peak_minus1db(x: np.ndarray, **_) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1e-6
    return (x * (10 ** (-1 / 20) / peak)).astype(np.float32)


def silence_pad_1s(x: np.ndarray, **_) -> np.ndarray:
    z = np.zeros(SR, dtype=np.float32)
    return np.concatenate([z, x, z])


def duration_first3s(x: np.ndarray, **_) -> np.ndarray:
    return x[: 3 * SR].astype(np.float32)


def telephone_quality(x: np.ndarray, **_) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    peak = float(np.max(np.abs(y))) or 1e-6
    y = y / peak * 0.9                       # avoid clipping before quantisation
    mu = 255.0
    comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round(comp * 127) / 127           # 8-bit mu-law levels
    y = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu))) / mu
    return (y * peak / 0.9).astype(np.float32)


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


APPLY = {
    "clean": lambda x, **k: x.astype(np.float32),
    "narrowband_8k": narrowband_8k,
    "codec_1": lambda x, **k: codec_roundtrip(x, "codec_1"),
    "codec_2": lambda x, **k: codec_roundtrip(x, "codec_2"),
    "noise": add_noise,
    "loud_minus20db": loud_minus20db,
    "loud_peak_minus1db": loud_peak_minus1db,
    "silence_pad_1s": silence_pad_1s,
    "duration_first3s": duration_first3s,
    "telephone_quality": telephone_quality,
}


def apply(cond: str, x: np.ndarray, clip_id: str = "") -> np.ndarray:
    return APPLY[cond](x, clip_id=clip_id)
