# Voice Check — voice-clone detection for phone scams

**Hypnexis '26 hackathon, problem AI-01.** An Android app that gives you a second check on any voice:
pick a voice note and get a genuine / synthetic / mixed verdict **computed on the phone**, or make an
in-app call where the **other person's voice is checked live** while you talk. Audio never leaves the device.

<p align="center">
  <img src="ppt%20screenshots/01_home.png" width="180" alt="Home screen">
  <img src="ppt%20screenshots/06_result_mixed_voices_timeline.png" width="180" alt="Mixed voices result with timeline">
  <img src="ppt%20screenshots/10_in_call_ai_voice_toggle.png" width="180" alt="In-app call with live check">
  <img src="ppt%20screenshots/14_web_ai_caller_page.png" width="180" alt="AI caller demo page">
</p>

## What it does

| Feature | How |
|---|---|
| **Check a voice note** | Pick any audio file (WAV, M4A/AAC, MP3…). 2–10 s clips get one verdict: *Likely genuine* or *Likely synthetic*. |
| **Mixed clips** | Clips longer than 10 s (up to 60 s) get a per-segment timeline. A 20 s clip with 7 s of AI speech in the middle comes back as **Mixed voices** with the AI span marked. |
| **Live call check** | Start a call, share the invite link; it opens the app on the friend's phone (Android App Links) and auto-joins. The remote participant's audio is scored on your phone in 4-second windows every 2 s. Your own microphone is never scored. |
| **Verdict relay** | Each phone tells the other side (and the web call page) what it concluded about that side's voice. |
| **AI caller demo** | "Demo: call an AI voice" gives a link; whoever opens it streams a synthetic voice into the room. Inside a call, "Play AI voice into call" swaps your mic for the AI sample so the other phone's live check can be shown flagging it. |
| **Offline and private** | One-time model download (1.26 GB) into app-private storage, SHA-256 verified. Inference runs in ONNX Runtime on the phone's CPU. No uploads, no API. |
| **Dev screen** | Model status and checksum, raw scores, per-window results, preprocessing mode, timings, memory, reference-tensor parity check. |

## Architecture

```
Voice note ──┐                                   ┌─ Result screen (genuine / synthetic / mixed + timeline)
             ├─► decode → mono → 16 kHz → peak −3 dBFS → 4 s windows / 2 s hop ─► ONNX model (on device) ─┤
Call audio ──┘   (remote participant, 16 kHz PCM tapped from WebRTC)                                     └─ Live strip in the call + verdict relayed to the peer
```

- **App** (`mobile/`): React Native + TypeScript (Expo dev build), Android only. Screens: Home, Result, Dev, Call.
- **Native detector module** (`mobile/modules/vcd-detector/`, Kotlin): audio decode (WAV parser + MediaCodec),
  Kaiser-sinc resampler, preprocessing (a port of `src/vcd/dhwani_infer.py`), ONNX Runtime session (batch 1),
  model store with resumable download and checksum, timeline analysis, and a `DetectorAdapter` / `AdapterSpec`
  contract so the model can be swapped without touching the UI.
- **Calling** (`calling/`): vendored and extended [LiveCalling](https://github.com/DevanshModi09/LiveCalling) kit.
  WebView-hosted WebRTC call page, Node.js signaling server over WebSocket, STUN + self-hosted TURN (coturn),
  ICE restart and reconnect, Android App Links (`/.well-known/assetlinks.json`) and a `voicecheck://` scheme.
  Deployed on AWS Lightsail behind Caddy HTTPS (`calling/deploy/`).
- **Reference pipeline** (`src/vcd/`, `notebooks/`): the Python inference used to validate the checkpoint in
  Google Colab, a FastAPI service used before the app went offline, and a reference-tensor export used to prove
  the on-device port matches Colab (max logit difference 2.4e-6 for the runtime, ≤ 3.3e-3 end to end).
- **Evaluation** (`evaluation/`): score-free dataset selection, generalisation and robustness notebooks, candidate
  comparison, and an Android hand-off package for the next detector.

## The detector, honestly

The app ships the open **Dhwani** checkpoint
(`ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model`, revision `ee01871`, MIT; XLS-R 300M + graph
attention head). **We did not train it.** We validated its preprocessing against the author's training code,
ported it to Android bit-for-bit, and built the product around a replaceable adapter.

Our evaluation (`evaluation/PROGRESS.md`) found that this checkpoint **does not generalise**: on an In-the-Wild
held-out set it scored at chance (ROC-AUC 0.46, EER 0.56, 155 clips). The same protocol put
**AntiDeepfake Wav2Vec2-Small** (NII, CC BY-NC-SA weights) at AUC 0.996 / EER 2.5 % on clean audio, with a
known weakness under heavy noise (EER 16 % at 10 dB SNR). That model has been exported to ONNX with reference
tensors and a hand-off package (`evaluation/handoff/ANDROID_HANDOFF.md`); swapping it in is the next step and
needs only a new adapter.

So: the pipeline, the on-device engine, the live-call check and the demos are real and verified; the verdict
quality is bounded by the baseline checkpoint. Scores are uncalibrated model outputs, never probabilities, and a
"genuine" result is not proof that a caller is safe.

## Try it

**Install the app:** download `call apk.apk` from the GitHub Releases page of this repository (or build it,
below) and install it on an Android phone (arm64). Open the app once and tap *Download model* (1.26 GB, one
time). After that it works offline. If you update the app later, install over the old one; uninstalling removes
the model.

**Test clips:** `testing apk/` (local, not in git) holds the two reference clips and `mixed_20s_gen8_syn7_gen5.wav`
(8 s real + 7 s AI + 5 s real) for the mixed-voice demo.

**Call a friend:** Home → *Call a friend · live check* → *Start a call* → *Share invite*. The friend taps the link;
the app opens and joins. The live strip shows the verdict on their voice every 2 s.

**AI caller demo:** Call → *Demo: call an AI voice* → open the AI caller page on any device and tap *Start AI
caller*, or during a normal call tap *Demo: play AI voice into call*.

## Build

```bash
cd mobile && npm install
export ANDROID_HOME=~/Library/Android/sdk
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
npx expo prebuild --platform android            # only if android/ is missing
cd android && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
adb install -r app/build/outputs/apk/release/app-release.apk
./gradlew --stop
```

Calling server: `calling/README.md` (local run) and `calling/deploy/LIGHTSAIL.md` (Ubuntu + Caddy + coturn +
systemd; `setup-lightsail.sh` once, `update-lightsail.sh` to redeploy from this repository). The app's default
server is set in `mobile/src/screens/CallScreen.tsx` and in `mobile/app.json` (App Links host).

## Repository map

| Path | Contents |
|---|---|
| `mobile/` | The Android app (Expo / React Native + Kotlin native module). `mobile/README.md` has the module layout. |
| `calling/` | Call page, signaling server, AI caller page, deployment scripts. |
| `src/vcd/` | Python reference inference, validation, windowing; FastAPI service. |
| `notebooks/`, `runs/` | Executed Colab notebooks and their outputs (baseline inference, API, reference export). |
| `evaluation/` | Dataset pins, evaluation notebooks, run records, candidate shortlist, Android hand-off. |
| `docs/` | `PROJECT_CONTEXT.md` (brief and rules), `UPSTREAM_AUDIT.md` (pins and preprocessing evidence), `PROGRESS.md` (everything verified, in order), `DETECTOR_INTEGRATION_HANDOFF.md` (how to add a detector). |
| `designfordevansh/` | UI design hand-off (theme, components, screen code). |
| `ppt screenshots/` | Screenshots of the real build used in the presentation. |

Models, datasets, audio clips, APKs and build outputs are git-ignored. All model work ran in Google Colab.

## Attribution and licences

- Detector: Dhwani by ayush2635 (MIT), Hugging Face revision `ee01871`. Candidate: AntiDeepfake Wav2Vec2-Small by
  NII Yamagishi Lab (code BSD-3, weights CC BY-NC-SA 4.0).
- Calling kit: LiveCalling by Devansh Modi, vendored at `1a01138` and extended.
- AI voice sample for the demo: Chatterbox TTS speech from the MLAAD dataset, CC BY-NC 4.0, demo use only
  (`calling/web/samples/ATTRIBUTION.md`).
- Evaluation data: ASVspoof 2019 LA (ODC-By) and In-the-Wild (Müller et al. 2022); nothing redistributed.
- ONNX Runtime (MIT), Expo / React Native (MIT), coturn (BSD), Caddy (Apache-2.0).

## Limitations

- Verdicts come from an unchanged third-party checkpoint with weak generalisation (see above); treat every result
  as a hint and verify the request through another channel.
- Android only; iOS module is a stub. The live check needs the 1.26 GB model installed on the checking phone.
- Calls are foreground-only (keep the screen open), two participants, no ringing or push.
- Mixed-clip boundaries are approximate to about ±2 s because windows are 4 s long.
