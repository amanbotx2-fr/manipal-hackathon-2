"""Reference-implementation scorer for the parity check. Runs in a SEPARATE Python 3.8 venv with
fairseq==0.12.2 (the authors' dependency), never in the main Colab kernel.

Usage: python fairseq_parity_ref.py <model.safetensors> <inputs.npz> <out.npz>
inputs.npz: arrays w0, w1, ... = already layer-normed mono 16 kHz float32 waveforms (identical to
what the transformers path receives). Writes logits (fake, real) per input, computed exactly like the
model card's code: fairseq Wav2Vec2Model(features_only=True)['x'] -> mean over time -> proj_fc.
"""
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config

model_path, in_path, out_path = sys.argv[1:4]
cfg = Wav2Vec2Config(encoder_layers=12, encoder_embed_dim=768, quantize_targets=True, latent_dim=256, final_dim=256)
ssl = Wav2Vec2Model(cfg)
sd = load_file(model_path)
enc = {k[len("m_ssl.model."):]: v for k, v in sd.items() if k.startswith("m_ssl.model.")}
missing, unexpected = ssl.load_state_dict(enc, strict=False)
assert not unexpected, unexpected
assert not missing, missing   # the checkpoint carries every fairseq tensor, incl. quantizer
head = torch.nn.Linear(768, 2)
head.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("proj_fc.")})
ssl.eval(); head.eval()
inp = np.load(in_path)
out = {}
with torch.no_grad():
    for k in sorted(inp.files):
        x = torch.from_numpy(inp[k].astype(np.float32))[None]
        feats = ssl(x, mask=False, features_only=True)["x"]        # [1, T, 768]
        out[k] = head(feats.mean(dim=1))[0].numpy()
np.savez(out_path, **out)
print("fairseq", getattr(__import__("fairseq"), "__version__", "?"), "torch", torch.__version__, "scored", len(out))
