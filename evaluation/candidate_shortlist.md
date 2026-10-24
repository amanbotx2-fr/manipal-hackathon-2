# Replacement detector shortlist (research, 2026-10-24)

Compiled from web sources by a research subagent and spot-checked by me (HF API: revision, file size, LFS sha256, licence tag; model card text; PyPI). Claims marked *agent* were not re-verified by me. "SR" = self-reported by the authors; "IND" = independent re-evaluation.

Independent sources: Speech DF Arena paper (<https://arxiv.org/html/2509.02859>, Table 2) and the lab260 "Speech Anti-Spoofing Arena" leaderboard on Hugging Face (*agent*).

| Model | ITW EER | 19LA EER | Trained on | Weights | Licence (code / weights) | Params | Android path |
|---|---|---|---|---|---|---|---|
| **AntiDeepfake Wav2Vec2-Small** (NII) | 4.24 % SR (card, AUC 0.993, EER threshold 0.8128); 5.31 % IND (*agent*) | not reported (19LA in training) | 56k h real + 18k h fake, 27+ corpora incl. 19LA, 21LA/DF, ASVspoof5, MLAAD, SpoofCeleb, VoxCeleb2; RawBoost. ITW **not** in training table (verified on card) | HF `nii-yamagishilab/wav2vec-small-anti-deepfake` @ `9a13264b…`, `model.safetensors` 380,210,632 B, sha256 `828ee456…` (verified) | BSD-3 / **CC BY-NC-SA 4.0** (verified) | 95 M | wav2vec2-base; Optimum supports ONNX export of Wav2Vec2; int8 wav2vec2 Android demo exists (*agent*). Published loader needs fairseq 0.12.2 (wheels ≤ cp38; verified on PyPI) → our transformers mapping |
| AntiDeepfake W2V-Large | 1.91 % SR; 2.94 % IND (NDA) | n/a | same | HF, ≈ 1.27 GB | BSD-3 / CC BY-NC-SA 4.0 | 317 M | same path, Dhwani-sized |
| DF Arena 500M V1 | 1.76 % SR; 1.87 % IND | 1.09 % SR (in training) | ASVspoof 19/24, Codecfake, LibriSeVoc, DFADD, CtrSVDD, SpoofCeleb, MLAAD, EnvSDD | HF `Speech-Arena-2025/DF_Arena_500M_V_1`, 1.75 GB `.bin`, custom code | custom / **non-commercial** | 436 M | XLS-R + Conformer; export unverified; heavy |
| XLSR-Mamba | 6.71 % SR; 6.70 % IND | 0.42 % IND | 19LA + RawBoost | HF `AustinXiao/XLSR-Mamba-DF`, 1.28 GB | MIT / MIT | 319 M | blocked: mamba-ssm CUDA kernels |
| Nes2Net-X (XLS-R) | 5.15 % SR; 7.75 % IND | 0.12 % IND | 19LA + RawBoost ("Val Aug" undefined) | Google Drive ckpt, size unknown | Apache-2.0 / Apache-2.0 | 318 M | plausible ONNX; Dhwani-sized |
| XLSR-SLS | 7.46 % SR; 7.45 % IND | 0.23 % IND | 19LA + RawBoost | Google Drive `.pth` | **no licence** | 340 M | plausible; legally unusable as shipped |
| SSL-AASIST (Tak et al.) | 11.19 % IND | 0.22 % IND | 19LA + RawBoost | Google Drive `.pth` | MIT / unstated | 318 M | fairseq XLS-R; GAT ops |
| AASIST / AASIST-L | 43 % IND | 0.83 % / 0.99 % SR | 19LA | in repo, tiny | MIT | 0.3 M / 85 k | trivial, but unusable on ITW |
| RawNet2 | 34–49 % | 4.4 % | 19LA | none official | MIT | 17.6 M | easy, poor |
| Whisper-MesoNet / LCNN / SpecRNet | 26.7 % IND | 5.8 % | 21DF subset | Drive / none | MIT | 7.6 M | easy, poor |
| HF "popular" classifiers (MelodyMachine, Gustking, garystafford) | **none report ITW** | — | undocumented | HF | Apache-2.0 | 95–300 M | easy; no generalisation evidence |

## Decision
**AntiDeepfake Wav2Vec2-Small**: the only phone-sized model with strong *and independently reproduced* In-the-Wild evidence, pinned HF weights, documented recipe, and a known base architecture with an ONNX path. Costs accepted for the hackathon: non-commercial licence (must be resolved before any product use), fairseq-free loading that we verify ourselves (notebook 04 cell 9), possible celebrity-speaker overlap via VoxCeleb2.
Fallback if a permissive licence becomes mandatory: Nes2Net-X (Apache-2.0), at Dhwani's footprint.

## Caveats that apply to every candidate
- Published ITW numbers use full-length clips; the card shows 4 s inputs are worse than 10 s on Deepfake-Eval-2024 (17.7 % vs 14.0 % EER after fine-tuning). Our 2–10 s policy is in between.
- Deepfake-Eval-2024 zero-shot EER for AntiDeepfake-Small is 33 %: real-world 2024 deepfakes remain hard for all of these.
- No on-phone latency is published for any model; the ONNX step in notebook 04 measures Colab CPU only.
