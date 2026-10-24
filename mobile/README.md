# Voice Clone Check — mobile demo (React Native + Expo, TypeScript)

Current app: **offline, on-device detection** with the third-party Dhwani ONNX checkpoint
(ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model @ `ee01871`, MIT; not trained by us).
After a one-time model install, audio is decoded, preprocessed and scored on the phone. No audio
upload and no API calls. Scores are **uncalibrated** model outputs, not probabilities or accuracy;
a "genuine" result is not proof that a caller is safe.

The earlier server-upload version is kept for reference: `reference/server-upload-v1/` (source)
and `dist/voice-clone-check-arm64-release-debugsigned.apk` (built APK).

## Replaceable detector

- **UI / calling code** use only `src/detector` (`Detector` interface: `spec`, `status`, `install`, `load`,
  `analyseFile(uri, opts)`, `analysePcm(float32Samples, sampleRate, channels, opts)`). Neither knows the model.
- **Native engine** (`DetectorEngine`, `ModelStore`, `OnnxWindowSession`, `AudioDecode`, `Resampler`) is model-agnostic.
- **Model adapters** implement `DetectorAdapter` and declare an `AdapterSpec`: id/version, pinned artifact
  (repo, revision, file, sha256, bytes, licence), sample rate, window/hop, duration limits, batch size,
  preprocessing modes + default, output label mapping + positive index, aggregation, threshold
  (+ `thresholdTuned`), `scoreCalibrated`, attribution.
- `DhwaniAdapter` is the baseline and the only registered adapter (`Adapters` in `DetectorAdapter.kt`).
  Add another only after the evaluation work supplies a verified candidate: write its adapter, register it,
  and check it against a Colab reference export with the same tooling. Step-by-step: `../docs/DETECTOR_INTEGRATION_HANDOFF.md`.
- **Status:** the adapter layer compiles (`tsc`, `compileReleaseKotlin`) but is unverified at runtime. `dist/voice-clone-check-offline-arm64-release-debugsigned.apk` is the verified pre-refactor build.

## Layout

- `App.tsx`: UI (model install/load, reference check, mode selector, file picker, results, timings, memory).
- `modules/vcd-detector/`: local Expo module.
  - `android/.../ModelStore.kt`: pinned download to app-private storage (`filesDir/models/dhwani-ee01871/`), resumable `.part` + HTTP Range, SHA-256 verify, atomic rename.
  - `DetectorAdapter.kt` / `DhwaniAdapter.kt`: adapter contract + Dhwani baseline adapter.
  - `OnnxWindowSession.kt`: one ONNX Runtime session (onnxruntime-android 1.31.0, CPU, ALL_OPT, batch 1).
  - `AudioDecode.kt`: WAV parser (soundfile scaling) + MediaCodec for m4a/AAC etc. (16-bit PCM like audioread).
  - `Resampler.kt`: polyphase Kaiser-sinc approximating soxr "HQ" (not bit-identical).
  - `Preprocess.kt`: Dhwani preprocessing, a port of `src/vcd/dhwani_infer.py` (validation, windows, train_peak / train_zscore / api_replica, softmax, mean aggregation).
  - `DetectorEngine.kt`: single inference thread; `analyseFile`, `analysePcm` (in-memory PCM for future in-app calls), reference comparison, memory metrics.
  - `ios/VcdDetectorModule.swift`: **stub only**; every inference call rejects ("not implemented on iOS yet").
- The ONNX Runtime telemetry content provider is removed in the module's AndroidManifest (`tools:node="remove"`).

## Build and install (Android)

```bash
cd mobile && npm install
export ANDROID_HOME=~/Library/Android/sdk
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
npx expo prebuild --platform android          # only if android/ is missing
cd android && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
adb install -r app/build/outputs/apk/release/app-release.apk
./gradlew --stop   # always: idle Gradle daemons held ~2.7 GB RAM and pushed macOS swap (and disk use) up
```

The release APK embeds the JS bundle (no Metro) and is signed with the debug keystore. arm64-v8a only.
The model is **not** in the APK, the JS bundle or git.

## Model install

In the app: **Download model (1.26 GB)**. It downloads once over HTTPS from the pinned Hugging Face revision
into app-private storage, shows progress, resumes after interruption, verifies SHA-256
`d1c232bf…7342`, then enables analysis. Needs ~1.3 GB free on the device.

## Reference parity check

1. In Colab, run `../notebooks/colab_cells/reference_export_for_device.py` and download `vcd_reference_export.zip`.
2. Unzip it and push it into the app's external files dir (adb can write there even for release builds):

```bash
adb push vcd_reference_export/. /sdcard/Android/data/com.hypenex.voiceclonecheck/files/reference/
```

3. Stage-by-stage comparison on the Mac (no inference): `adb pull /sdcard/Android/data/com.hypenex.voiceclonecheck/files/dumps/ dumps/`
   then `python3 scripts/compare_device_reference.py vcd_reference_export dumps report.json` (from the repo root).
4. In the app: **Run reference tensors** runs the exported windows on-device and compares logits. Analysing a
   reference clip with "Compare with Colab reference" enabled compares decoded audio, 16 kHz audio, windows,
   logits and the clip score, and writes the device tensors to `files/dumps/<sha12>/`.

## Offline test

```bash
adb shell cmd connectivity airplane-mode enable
adb shell am force-stop com.hypenex.voiceclonecheck
adb shell am start -n com.hypenex.voiceclonecheck/.MainActivity
```

Then pick each clip from Downloads and analyse.

## In-app calls with a live check (branch `live-calling`)

Built on Devansh Modi's **LiveCalling** kit (https://github.com/DevanshModi09/LiveCalling, commit `1a01138`):
a WebView-hosted WebRTC audio call whose JS bridge delivers the *other* participant's 16 kHz mono PCM
to native in 320-sample chunks. Our additions:

- `modules/vcd-detector/android/.../vcdcalling/WebRtcCallView.kt`: vendored from the kit (package renamed).
- `LiveAnalyser.kt`: rolling 10 s buffer of remote PCM; every 2 s scores the latest 4 s window via the
  **shared** `DetectorEngine` (same model and session as file analysis), skipping a window if the previous
  one is still running. The local microphone is never scored.
- `VcdCallView.kt` + `VcdCallingModule.kt`: Expo view (`serverUrl`, `liveCheck`, `onCallState`, `onLiveResult`).
- `src/screens/CallScreen.tsx`: server/invitation entry, microphone permission, the call view, and a live
  strip with the latest window's verdict and uncalibrated score. Reached from Home ("start an in-app call").
- `../calling/`: the kit's signaling server and web page for deployment (see its README).

Requirements: an **HTTPS** calling server (the WebView rejects anything else), both people in the same
room, screen kept in the foreground. STUN/TURN for cross-network calls. Android only; no ringing,
identity or background calls. Live-check results are model outputs at an untuned threshold.

## iOS

Not implemented or tested. The Swift module is a stub. The plan is `onnxruntime-objc` 1.31.0 + AVAudioFile decoding +
the same preprocessing port, built with full Xcode (this Mac has only the Command Line Tools) or EAS Build.
