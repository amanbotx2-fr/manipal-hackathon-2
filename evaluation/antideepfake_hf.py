"""Candidate detector: AntiDeepfake Wav2Vec2-Small (NII Yamagishi Lab), loaded WITHOUT fairseq.

Model: nii-yamagishilab/wav2vec-small-anti-deepfake @ MODEL_REVISION, model.safetensors
(380,210,632 B, sha256 MODEL_SHA256). Licence CC BY-NC-SA 4.0 (research/educational use;
NOT for commercial deployment without permission). Base: facebook/wav2vec2-base (Apache-2.0).
Paper: Ge, Wang, Liu, Yamagishi, "Post-training for Deepfake Speech Detection", arXiv:2506.21090.
We did not train or fine-tune this model.

Why this loader exists: the published inference code builds the encoder with
fairseq 0.12.2 (Python <= 3.8 wheels; the project is archived), which does not install on
current Colab runtimes. The checkpoint's encoder keys are fairseq wav2vec2 names, so we map
them onto transformers' Wav2Vec2Model using the same correspondence as HF's official
convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py, and FAIL LOUDLY if any
parameter is left over or missing. Numerical identity with the fairseq path is NOT
proven here (see verify_notes()); the notebook checks behaviour on development evidence
before any held-out scoring.

Architecture correspondence, checked against fairseq v0.12.2 fairseq/models/wav2vec/wav2vec2.py and
transformers modeling_wav2vec2.py (code read side by side; see architecture_check()):
  * conv feature extractor: 7 Conv1d(no bias) [(512,10,5),(512,3,2)x4,(512,2,2)x2]; fairseq
    extractor_mode='default' = GroupNorm(512,512) after conv 0 only, GELU after every conv
    == HF feat_extract_norm='group', conv_bias=False, feat_extract_activation='gelu'.
  * LayerNorm(512) on frames, then Linear(512,768) (fairseq model.layer_norm + post_extract_proj
    == HF feature_projection.layer_norm + .projection).
  * positional conv: Conv1d(768,768,k=128,pad=64,groups=16), weight_norm(dim=2), drop last frame
    (SamePad == Wav2Vec2SamePadLayer), GELU; x = x + pos.
  * post-LN transformer (fairseq layer_norm_first=False == HF do_stable_layer_norm=False):
    encoder.layer_norm BEFORE the 12 layers; each layer: attn -> residual -> layer_norm -> FFN(GELU,
    3072) -> residual -> final_layer_norm; 12 heads, scaling head_dim**-0.5; exact (erf) GELU both.
    No final norm after the last layer in either implementation. Output = last_hidden_state == fairseq 'x'.
  * fairseq pads frames to a multiple of 2 with a key-padding mask and removes the pad afterwards;
    with the pad masked this is a no-op for the result. HF does no such padding. No attention mask is
    used by either side for a single unpadded clip (batch size 1).
  * head: mean over time (AdaptiveAvgPool1d(1)) -> Linear(768,2) -> softmax; fake = index 0.
  * eval mode: no time/feature masking, no dropout, no layerdrop on either side.
  Differences that remain: the authors decode with torchaudio and resample with torchaudio.functional.resample
  (our path: soundfile/librosa + soxr; irrelevant at 16 kHz input), and they score whole files while we apply
  the 2-10 s policy (first 10 s). Numerical parity is only established by the optional fairseq parity cell.

Published recipe (model card), reproduced here:
  mono (mean) -> 16 kHz -> layer_norm over the whole waveform (zero mean, unit var, eps 1e-5)
  -> encoder last hidden state -> mean over time -> Linear(768, 2) -> softmax.
  Output order: index 0 = FAKE, index 1 = REAL (card: "<Fake score, Real score>").
  Arbitrary input length (no cropping in the published code).
"""

from __future__ import annotations

import time

try:
    import numpy as np
except ImportError:  # the key-mapping functions are stdlib-only and testable without numpy
    np = None

MODEL_REPO = "nii-yamagishilab/wav2vec-small-anti-deepfake"
MODEL_REVISION = "9a13264b5dcc827a8d5a4f8e01fccefa392f886b"
MODEL_FILE = "model.safetensors"
MODEL_SHA256 = "828ee456122f86d5d631cb7895a10e5c62c78a4fcb8a8b1c42cb5838a9abcfe0"
MODEL_BYTES = 380210632
LICENSE = "CC-BY-NC-SA-4.0 (weights); base facebook/wav2vec2-base Apache-2.0; code BSD-3"
FAKE_INDEX = 0          # softmax index of the synthetic class (card: <Fake, Real>)
# The card reports EER 4.24 % "@ 0.8128" on In-the-Wild. That threshold is the EER point found ON In-the-Wild
# with the authors' whole-file pipeline, i.e. it was selected on the dataset we test on. It is therefore NOT a
# pre-registered operating point for us and is reported only as context, never as a decision threshold.
CARD_EER_THRESHOLD_ITW_CONTEXT_ONLY = 0.8128
SR = 16000

# Keys of the fairseq checkpoint that belong to pre-training only (quantiser, projections,
# mask embedding). They have no counterpart in inference and are dropped on purpose.
PRETRAIN_ONLY_PREFIXES = ("quantizer.", "project_q.", "final_proj.", "mask_emb")


def _map_encoder_key(k: str) -> str | None:
    """fairseq Wav2Vec2Model key (prefix 'm_ssl.model.' removed) -> transformers Wav2Vec2Model key."""
    if k.startswith(PRETRAIN_ONLY_PREFIXES):
        return None
    if k.startswith("feature_extractor.conv_layers."):
        _, _, layer, typ, param = k.split(".")
        if typ == "0":
            return f"feature_extractor.conv_layers.{layer}.conv.{param}"
        if typ == "2":  # group norm (layer 0 only, feat_extract_norm='group')
            return f"feature_extractor.conv_layers.{layer}.layer_norm.{param}"
        raise KeyError(k)
    if k.startswith("post_extract_proj."):
        return "feature_projection.projection." + k.split(".", 1)[1]
    if k.startswith("layer_norm."):
        return "feature_projection.layer_norm." + k.split(".", 1)[1]
    if k.startswith("encoder.pos_conv.0."):
        param = k.rsplit(".", 1)[1]           # weight_g / weight_v / bias
        return "encoder.pos_conv_embed.conv." + param
    if k.startswith("encoder.layer_norm."):
        return k
    if k.startswith("encoder.layers."):
        _, _, i, rest = k.split(".", 3)
        sub = {
            "self_attn.k_proj": "attention.k_proj", "self_attn.v_proj": "attention.v_proj",
            "self_attn.q_proj": "attention.q_proj", "self_attn.out_proj": "attention.out_proj",
            "self_attn_layer_norm": "layer_norm", "fc1": "feed_forward.intermediate_dense",
            "fc2": "feed_forward.output_dense", "final_layer_norm": "final_layer_norm",
        }
        for src, dst in sub.items():
            if rest.startswith(src + "."):
                return f"encoder.layers.{i}.{dst}.{rest[len(src) + 1:]}"
        raise KeyError(k)
    raise KeyError(k)


def convert_state_dict(sd: dict, hf_keys: set) -> tuple[dict, dict, dict]:
    """Return (encoder_state_dict, head_state_dict, report). Raises on anything unexplained."""
    enc, head, dropped, unexplained = {}, {}, [], []
    for k, v in sd.items():
        if k.startswith("proj_fc."):
            head[k.split(".", 1)[1]] = v
            continue
        if not k.startswith("m_ssl.model."):
            unexplained.append(k)
            continue
        kk = k[len("m_ssl.model."):]
        try:
            mk = _map_encoder_key(kk)
        except KeyError:
            unexplained.append(k)
            continue
        if mk is None:
            dropped.append(k)
            continue
        # weight-norm naming differs across torch/transformers versions
        if mk.endswith(("weight_g", "weight_v")) and mk not in hf_keys:
            alt = mk.replace("weight_g", "parametrizations.weight.original0").replace(
                "weight_v", "parametrizations.weight.original1")
            if alt in hf_keys:
                mk = alt
        enc[mk] = v
    missing = sorted(hf_keys - set(enc))
    unexpected = sorted(set(enc) - hf_keys)
    report = {"n_checkpoint_tensors": len(sd), "n_encoder_mapped": len(enc), "n_head": len(head),
              "dropped_pretrain_only": dropped, "unexplained": unexplained,
              "missing_in_checkpoint": missing, "unexpected_for_hf": unexpected}
    if unexplained or missing or unexpected or set(head) != {"weight", "bias"}:
        raise RuntimeError(f"checkpoint/model mismatch: {report}")
    return enc, head, report


# fairseq Wav2Vec2Config defaults that the authors' code relies on (only 5 fields are set explicitly there)
EXPECTED_HF_CONFIG = {
    "model_type": "wav2vec2", "feat_extract_norm": "group", "conv_bias": False, "feat_extract_activation": "gelu",
    "conv_dim": [512] * 7, "conv_kernel": [10, 3, 3, 3, 3, 2, 2], "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072,
    "hidden_act": "gelu", "do_stable_layer_norm": False, "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16, "layer_norm_eps": 1e-05, "add_adapter": False,
}


def architecture_check(cfg) -> dict:
    """Compare the HF config against the fairseq-default architecture the checkpoint was trained with."""
    got = {k: getattr(cfg, k) for k in EXPECTED_HF_CONFIG}
    bad = {k: (got[k], v) for k, v in EXPECTED_HF_CONFIG.items() if got[k] != v}
    if bad:
        raise RuntimeError(f"HF config does not match the fairseq architecture: {bad}")
    return got


class AntiDeepfakeHF:
    """Thin inference wrapper (torch + transformers). Build with .load()."""

    def __init__(self, encoder, head, device, report):
        self.encoder, self.head, self.device, self.report = encoder, head, device, report

    @classmethod
    def load(cls, safetensors_path: str, config_source: str, device: str = "cpu"):
        import torch
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        cfg = Wav2Vec2Config.from_pretrained(config_source)
        arch = architecture_check(cfg)
        # inference only: no masking, no layerdrop, no dropout
        cfg.mask_time_prob = 0.0
        cfg.mask_feature_prob = 0.0
        cfg.layerdrop = 0.0
        cfg.apply_spec_augment = False
        encoder = Wav2Vec2Model(cfg)
        sd = load_file(safetensors_path)
        hf_keys = set(encoder.state_dict().keys()) - {"masked_spec_embed"}
        enc_sd, head_sd, report = convert_state_dict(sd, hf_keys)
        res = encoder.load_state_dict(enc_sd, strict=False)
        assert not res.unexpected_keys and set(res.missing_keys) <= {"masked_spec_embed"}, res
        head = torch.nn.Linear(768, 2)
        head.load_state_dict(head_sd)
        encoder.eval().to(device)
        head.eval().to(device)
        report["architecture"] = arch
        report["n_params_encoder"] = sum(p.numel() for p in encoder.parameters())
        report["n_params_head"] = sum(p.numel() for p in head.parameters())
        return cls(encoder, head, device, report)

    @staticmethod
    def preprocess(x16: np.ndarray) -> np.ndarray:
        """Published recipe: layer_norm over the whole waveform (eps 1e-5)."""
        x = x16.astype(np.float64)
        return ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32)

    def logits(self, wave: np.ndarray) -> np.ndarray:
        """wave: preprocessed mono 16 kHz float32 [n]. Returns logits [2] (index 0 = fake)."""
        import torch

        with torch.no_grad():
            t = torch.from_numpy(wave)[None].to(self.device)
            h = self.encoder(t).last_hidden_state          # [1, T, 768]
            z = self.head(h.mean(dim=1))                    # [1, 2]
        return z[0].float().cpu().numpy()


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score_clip(model: AntiDeepfakeHF, x16: np.ndarray, window=None, hop=None) -> dict:
    """Primary: whole analysed clip in one pass (as published). If window/hop are given,
    additionally scores 4 s windows (mean of fake scores) for comparability with Dhwani."""
    t0 = time.perf_counter()
    x = model.preprocess(x16)
    z = model.logits(x)
    p = softmax(z)
    out = {"logits_fake_real": [round(float(a), 4) for a in z], "fake_score_whole": round(float(p[FAKE_INDEX]), 6),
           "n_samples": int(len(x16))}
    if window:
        n = len(x)
        starts = [0] if n <= window else list(range(0, n - window + 1, hop))
        if n > window and starts[-1] + window < n:
            starts.append(n - window)
        ws = []
        for s in starts:
            w = x[s:s + window]
            if len(w) < window:
                w = np.pad(w, (0, window - len(w)))
            ws.append(float(softmax(model.logits(w))[FAKE_INDEX]))
        out.update(window_starts_s=[round(s / SR, 3) for s in starts], window_fake_scores=[round(v, 6) for v in ws],
                   fake_score_window_mean=round(float(np.mean(ws)), 6))
    out["latency_ms_model"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def verify_notes() -> str:
    return ("Weights were loaded through a fairseq->transformers key mapping (every checkpoint tensor accounted "
            "for) and the architecture fields were checked against fairseq defaults. This does NOT prove numerical "
            "parity with the authors' fairseq code; parity is 'unverified' unless the fairseq parity cell ran "
            "and reported a small max difference. The development-evidence gate (pilot clips) is a stop-and-"
            "investigate check: a low AUC does not by itself prove the mapping is wrong, and a high AUC does not "
            "prove parity.")
