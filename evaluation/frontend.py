"""Lightweight denoising front-ends under test (experiment, not an assumed fix).

Each front-end maps a mono 16 kHz float32 waveform to a mono 16 kHz float32 waveform of the
same length. In the experiment the SAME front-end is applied to every input, genuine and
synthetic, clean and degraded, exactly as it would run in front of the detector on a phone.
Settings are fixed here (no per-clip or per-condition tuning) and recorded in FRONTENDS.

  none            : identity (baseline)
  nr_stationary   : noisereduce spectral gating, stationary profile, prop_decrease 0.8,
                    n_fft 512 / hop 128 (32 ms / 8 ms at 16 kHz), 1.5 std threshold. Pure DSP,
                    CPU, no weights -> portable to Android (Kotlin re-implementation possible).
  deepfilternet3  : DeepFilterNet3 (Schroeter et al.), ~2 M params, MIT/Apache, default
                    post-filter off, attenuation limit default (100 dB). Runs at 48 kHz internally
                    (resampled in/out). Has an ONNX/LADSPA deployment story. Optional: only if
                    `pip install deepfilternet` works in the runtime.
"""

from __future__ import annotations

import numpy as np

SR = 16000

FRONTENDS = {
    "none": {"desc": "identity", "params": 0},
    "nr_stationary": {"desc": "noisereduce spectral gating (stationary=True, prop_decrease=0.8, n_fft=512, hop_length=128, "
                              "n_std_thresh_stationary=1.5)", "params": 0,
                      "kwargs": dict(stationary=True, prop_decrease=0.8, n_fft=512, hop_length=128, n_std_thresh_stationary=1.5)},
    "deepfilternet3": {"desc": "DeepFilterNet3 default model via `df.enhance`, 48 kHz internal, post-filter off", "params": 2_000_000},
}

_DF = {}


def available() -> dict:
    out = {"none": True}
    try:
        import noisereduce  # noqa: F401
        out["nr_stationary"] = True
    except Exception as e:
        out["nr_stationary"] = f"unavailable: {type(e).__name__}"
    try:
        from df.enhance import init_df  # noqa: F401
        out["deepfilternet3"] = True
    except Exception as e:
        out["deepfilternet3"] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    return out


def _fit(y, n):
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def nr_stationary(x: np.ndarray) -> np.ndarray:
    import noisereduce as nr

    y = nr.reduce_noise(y=x.astype(np.float32), sr=SR, **FRONTENDS["nr_stationary"]["kwargs"])
    return _fit(y, len(x))


def deepfilternet3(x: np.ndarray) -> np.ndarray:
    import torch
    from df.enhance import enhance, init_df
    import librosa

    if "model" not in _DF:
        model, df_state, _ = init_df(post_filter=False)
        _DF.update(model=model, state=df_state, sr=df_state.sr())
    y48 = librosa.resample(x.astype(np.float32), orig_sr=SR, target_sr=_DF["sr"], res_type="soxr_hq")
    with torch.no_grad():
        out = enhance(_DF["model"], _DF["state"], torch.from_numpy(y48)[None])
    y = librosa.resample(out[0].cpu().numpy(), orig_sr=_DF["sr"], target_sr=SR, res_type="soxr_hq")
    return _fit(y, len(x))


APPLY = {"none": lambda x: x.astype(np.float32), "nr_stationary": nr_stationary, "deepfilternet3": deepfilternet3}


def apply(name: str, x: np.ndarray) -> np.ndarray:
    return APPLY[name](x)
