"""Stdlib-only test of the fairseq->transformers key mapping in antideepfake_hf.py (no torch, no weights).

Run: python3 -I evaluation/tests/test_antideepfake_mapping.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import antideepfake_hf as ad  # noqa: E402

fs = ["m_ssl.model.mask_emb", "m_ssl.model.post_extract_proj.weight", "m_ssl.model.post_extract_proj.bias",
      "m_ssl.model.layer_norm.weight", "m_ssl.model.layer_norm.bias",
      "m_ssl.model.encoder.pos_conv.0.bias", "m_ssl.model.encoder.pos_conv.0.weight_g", "m_ssl.model.encoder.pos_conv.0.weight_v",
      "m_ssl.model.encoder.layer_norm.weight", "m_ssl.model.encoder.layer_norm.bias",
      "m_ssl.model.feature_extractor.conv_layers.0.0.weight", "m_ssl.model.feature_extractor.conv_layers.0.2.weight",
      "m_ssl.model.feature_extractor.conv_layers.0.2.bias"]
fs += [f"m_ssl.model.feature_extractor.conv_layers.{i}.0.weight" for i in range(1, 7)]
for i in range(12):
    for p in ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj",
              "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"):
        fs += [f"m_ssl.model.encoder.layers.{i}.{p}.weight", f"m_ssl.model.encoder.layers.{i}.{p}.bias"]
fs += ["m_ssl.model.quantizer.vars", "m_ssl.model.quantizer.weight_proj.weight", "m_ssl.model.quantizer.weight_proj.bias",
       "m_ssl.model.project_q.weight", "m_ssl.model.project_q.bias", "m_ssl.model.final_proj.weight",
       "m_ssl.model.final_proj.bias", "proj_fc.weight", "proj_fc.bias"]

# expected transformers Wav2Vec2Model(wav2vec2-base) keys, parametrised weight-norm naming
hf = {"feature_projection.projection.weight", "feature_projection.projection.bias",
      "feature_projection.layer_norm.weight", "feature_projection.layer_norm.bias",
      "encoder.pos_conv_embed.conv.bias", "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
      "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
      "encoder.layer_norm.weight", "encoder.layer_norm.bias",
      "feature_extractor.conv_layers.0.conv.weight", "feature_extractor.conv_layers.0.layer_norm.weight",
      "feature_extractor.conv_layers.0.layer_norm.bias"}
hf |= {f"feature_extractor.conv_layers.{i}.conv.weight" for i in range(1, 7)}
for i in range(12):
    for p in ("attention.k_proj", "attention.v_proj", "attention.q_proj", "attention.out_proj", "layer_norm",
              "feed_forward.intermediate_dense", "feed_forward.output_dense", "final_layer_norm"):
        hf |= {f"encoder.layers.{i}.{p}.weight", f"encoder.layers.{i}.{p}.bias"}

enc, head, rep = ad.convert_state_dict({k: k for k in fs}, hf)
assert set(enc) == hf and set(head) == {"weight", "bias"} and len(rep["dropped_pretrain_only"]) == 8, rep
# legacy weight-norm naming also accepted
hf_old = (hf - {"encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                "encoder.pos_conv_embed.conv.parametrizations.weight.original1"}) | {
    "encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv_embed.conv.weight_v"}
enc2, _, _ = ad.convert_state_dict({k: k for k in fs}, hf_old)
assert set(enc2) == hf_old
for bad in (fs + ["m_ssl.model.encoder.layers.0.bogus.weight"], fs[2:], fs + ["stray.weight"]):  # fs[2:] drops a real tensor
    try:
        ad.convert_state_dict({k: k for k in bad}, hf)
        raise SystemExit("mapping accepted a bad checkpoint")
    except RuntimeError:
        pass
print(f"ok: {rep['n_encoder_mapped']} encoder tensors mapped, {len(rep['dropped_pretrain_only'])} pre-training-only dropped, "
      "mismatches rejected")
