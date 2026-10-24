# Detector integration handoff

For whoever registers the next on-device model once the evaluation session supplies a verified candidate.
Written 2026-10-24. State of play: the Dhwani baseline adapter is the only registered detector. The
adapter layer **compiles** (`tsc --noEmit`, `:vcd-detector:compileReleaseKotlin`) but has **not been
rebuilt into an APK or run on a device**. The verified demo APK predates the refactor.

## Frozen artefacts (do not overwrite)

| Item | Where | Identity |
|---|---|---|
| Verified offline demo APK (pre-refactor) | `mobile/dist/voice-clone-check-offline-arm64-release-debugsigned.apk` | sha256 `e51a2244…a49c0`, 57 MB |
| Server-upload APK (reference) | `mobile/dist/voice-clone-check-arm64-release-debugsigned.apk` | sha256 `867da954…5887` |
| Installed model on the Pixel_10a AVD | `/data/user/0/com.hypenex.voiceclonecheck/files/models/dhwani-ee01871/best_model.onnx` | sha256 `d1c232bf…7342`, verified in-app |
| Colab reference export | `vcd_reference_export.zip` (repo root) | sha256 `7615a446…9aede` |

## The adapter files

All under `mobile/modules/vcd-detector/android/src/main/java/com/hypenex/vcddetector/`:

| File | Role | Touch when adding a model? |
|---|---|---|
| `DetectorAdapter.kt` | `DetectorAdapter` interface, `AdapterSpec` + `ModelArtifact` data classes, `AudioRejected`, `ValidatedAudio`, and the `Adapters` registry | **Yes**: add the new adapter to `Adapters.all` (and `Adapters.default` if it should replace Dhwani) |
| `DhwaniAdapter.kt` | Baseline adapter: spec + delegation to `Preprocess` | No. Copy it as the template for the new adapter |
| `Preprocess.kt` | Dhwani-specific validation, windowing, modes, softmax, label | No. A new model gets its own preprocessing object |
| `DetectorEngine.kt` | Model-agnostic pipeline: decode → resample to `spec.sampleRate` → `adapter.validate` → `adapter.windows` → one ONNX call per window → `adapter.windowScore` → `adapter.aggregate` → `adapter.label`; reference comparison; timing; memory | No |
| `ModelStore.kt` | Pinned download into `filesDir/models/<id>-<rev7>/`, resumable, sha256-verified | No |
| `OnnxWindowSession.kt` | One ORT session, runs `[1, n]` float32 windows, batch 1 | No, unless the new model needs a different input layout |
| `AudioDecode.kt`, `Resampler.kt` | WAV/MediaCodec decode; Kaiser-sinc resampler | No |
| `VcdDetectorModule.kt` | Expo bridge; exposes `getAdapterSpec()`, `listAdapters()` | No |

JS side: `mobile/src/detector/index.ts` (the `Detector` interface the UI uses) and
`mobile/modules/vcd-detector/src/VcdDetector.types.ts` (`AdapterSpec` type). Neither needs changes for a
new adapter unless the spec gains fields.

iOS: `mobile/modules/vcd-detector/ios/VcdDetectorModule.swift` is a stub that rejects every call. Nothing
on iOS exists to integrate into yet.

## What the engine feeds the model and expects back

**Input** (what `OnnxWindowSession.run` sends):
- One ONNX input, float32, shape `[1, spec.windowSamples]`.
- Mono audio at `spec.sampleRate`, already passed through `adapter.validate` and `adapter.windows`, so any
  normalisation, cropping or padding is the adapter's job.
- Windows are run **sequentially, batch size 1**. If the candidate supports batching, it still runs one at a
  time today; `spec.batchSize` is declarative.

**Output** (what the engine hands to `adapter.windowScore`):
- The first ONNX output, first row, as a `FloatArray` of any length. Dhwani returns 2 logits; a sigmoid model
  returning 1 value works too, as long as `windowScore` maps it to a positive-class score in `[0, 1]`.
- The engine checks the output batch dimension is 1 and nothing else.

**Scoring chain:** `windowScore(output) -> Double` per window, `aggregate(DoubleArray) -> Double` for the clip,
`label(Double) -> String` using `spec.decisionThreshold`. The result JSON carries `synthetic_score`,
`synthetic_score_max`, per-window logits/scores, `score_is_calibrated = spec.scoreCalibrated` and
`threshold_is_tuned = spec.thresholdTuned`.

**Entry points:** `analyseFile(uri, mode, compare, dump)` for picked files and `analysePcm(bytes, sampleRate,
channels, mode)` for in-memory float32 PCM (future in-app calls). Both reach the same `analyseDecoded` path.

## Registering a new detector: required steps

1. **Pin the artefact.** Repo, revision, filename, sha256, byte count, licence. Fill a `ModelArtifact`.
   The download URL must serve the exact pinned revision. No model file goes into the APK, JS bundle or git.
2. **Inspect the graph before writing code.** IR version, opset, operator list, input/output names and shapes.
   `scratchpad/onnx_peek.py` from the 2026-10-24 session did this for Dhwani over HTTP Range reads without a
   local download; reuse that approach. Confirm every operator is in onnxruntime-android 1.31.0's CPU kernel
   set, or bump the ORT version in `android/build.gradle` **and** `ORT_VERSION` in `OnnxWindowSession.kt`
   together, and re-verify Dhwani afterwards.
3. **Write `<Name>Adapter.kt`** implementing `DetectorAdapter`, with its own preprocessing object. Every
   `AdapterSpec` field is required: `id`, `displayName`, `version`, `model`, `sampleRate`, `windowSamples`,
   `hopSamples`, `minDurationS`, `maxDurationS`, `batchSize`, `preprocessingModes`, `defaultMode`,
   `outputLabels`, `positiveIndex`, `aggregation`, `decisionThreshold`, `thresholdTuned`, `scoreCalibrated`,
   `attribution`. Leave `thresholdTuned = false` and `scoreCalibrated = false` until the evaluation session says otherwise.
4. **Register it** in `Adapters.all`. Change `Adapters.default` only if it is meant to replace Dhwani in the UI.
5. **Produce a Colab reference export for it**: adapt `notebooks/colab_cells/reference_export_for_device.py`
   to the new preprocessing (same manifest layout: `native_mono.f32`, `x16.f32` or the model's rate,
   `windows_<mode>.f32`, full-precision logits). Use the two pinned reference clips (sha `8b03de09…`,
   `9448a599…`) plus whatever the evaluation session adds.
6. **Compile:** `cd mobile && npx tsc --noEmit`, then
   `cd android && ./gradlew :vcd-detector:compileReleaseKotlin --offline`, then `./gradlew --stop`.
7. **Build and install only when the Mac has ≥2 GB free**, then run the checks below.

## Checks before calling a new detector integrated

Each of these was run for Dhwani; the numbers are its results, to be beaten or matched.

| Check | How | Dhwani result |
|---|---|---|
| Install + verify | In-app download, kill mid-way, resume, sha256 ok | Resumed at byte 564,810,910; verified |
| Cold load + memory | `Load model`; sample `dumpsys meminfo` | 4–44 s load; 1.5 GB peak RSS, 1.44 GB PSS |
| Runtime parity | Push reference export, **Run reference tensors** | max \|Δlogit\| 2.38e-6 over 16 windows |
| Pipeline parity | Analyse both clips with Compare on; `scripts/compare_device_reference.py` on pulled dumps | logits ≤3.3e-3, clip score ≤5.2e-5, identical predictions |
| Offline | Airplane mode, cold restart, delete dumps, re-analyse; app network counters unchanged | Logits identical to online run |
| No crash / LMK | `logcat` for `Fatal signal`, `lowmemorykiller: Kill 'com.hypenex…` | None after the R8 keep rules |
| Rejections | Silent, short (<2 s), corrupt, empty input → `status: rejected` with reason | Covered by `Preprocess.validate` (server tests passed 21/21; on-device not re-run) |
| Dhwani regression | Re-run the parity checks for `dhwani` after registering the new adapter | Must stay within the numbers above |

Two clips establish numerical compatibility only. Accuracy, multilingual robustness and real-phone memory
and speed are separate questions for the evaluation session and a physical device.
