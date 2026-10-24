# Project context — AI-01 Voice-Clone Detection for Phone Scams

## Brief
- Classify 2–10 s speech clips as genuine vs synthetic.
- Evaluate on voice generators **excluded from training**.
- Robust to language and speaker variation.
- No shortcuts from silence, loudness, encoding or dataset origin.
- Pretrained self-supervised speech encoders allowed.

## Product (Android)
1. Import a voice note / recording and analyse it.
2. Protected in-app internet calls (Aman ↔ Harsh); detector analyses the **remote** participant's incoming audio.
   Not intercepting cellular/WhatsApp calls. Calling layer is being researched separately — not built yet.

Later (not now): flagged windows, audio-quality checks, history, transcript-based scam cues.

## Team / workflow
- Claude Code: implementation. ChatGPT: planning, review, debugging.
- Colab (T4) for heavy runs; Claude Code cannot access Colab or Drive — it creates notebooks locally, user uploads and runs them.
- Local: MacBook Air M4, 16 GB.

## Rules we hold ourselves to
- Never fabricate scores, logs, accuracy or provenance. Attribute third-party code/models.
- Only call weights "ours" after we train them.
- Softmax outputs are uncalibrated scores, not probabilities.
- Splits grouped by speaker / source+target reference / generator; augmentations and crops stay with their source clip.
- Separate validation (model + threshold selection) from an untouched final test with held-out generators.
- Respect dataset licences (IndicSynth, MLAAD are CC-BY-NC-4.0; MLAAD gated). No secrets in chat or repo. No large automatic downloads.

## Plan
1. **Baseline inference** with third-party Dhwani ONNX (current).
2. Small multilingual evaluation manifest (genuine + synthetic, seen + unseen generators).
3. Choose training approach from baseline results: frozen XLS-R + our own head first; partial fine-tuning only if justified.
4. Android integration (on-device vs server TBD), then calling layer.

See `UPSTREAM_AUDIT.md` for pins and known problems with the Dhwani checkpoint.
