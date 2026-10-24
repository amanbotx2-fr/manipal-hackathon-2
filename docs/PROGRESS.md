# Progress log

## 2026-10-24 — baseline inference notebook ready (not yet run)

**Done (local, by Claude Code)**
- Pinned upstream revisions + audit: `docs/UPSTREAM_AUDIT.md`.
- Audit findings: preprocessing conflict (training = peak −3 dBFS, 4 s; API/card = z-score, 3 s → padded to 4 s); augmentations applied only to genuine class (shortcut risk: noisy/band-limited ⇒ genuine); genuine vs fake Indic audio from different corpora (dataset-origin confound); random file-level split; zero-waveform fallback; unpublished "calibrated" T = 1.362; simplified GAT, not official AASIST; only ONNX published (no trainable `.pth`).
- Inference module `src/vcd/dhwani_infer.py`, embedded into `notebooks/01_dhwani_baseline_inference.ipynb` (built by `scripts/build_inference_notebook.py`).
- Controlled comparison of 3 preprocessing modes per clip: A `train_peak`, B `train_zscore`, C `api_replica`.
- Windowing: reject < 2 s; first 10 s only if longer; ≤ 4 s → one zero-padded window; > 4 s → 4 s windows, 2 s hop, last aligned to end; clip score = mean (max also reported).
- Rejects unreadable / empty / NaN / silent (peak < −50 dBFS) / mostly silent (< 1 s active) inputs.

**Checked locally**
- Notebook JSON generated; every code cell parses as Python (shell/magic lines excluded).
- Embedded module text is identical to `src/vcd/dhwani_infer.py`.
- Notebook never references a stand-in model; it loads only the pinned Dhwani file (sha256 enforced).

**NOT verified (needs Colab run)**
- That the module runs end-to-end (no execution happened locally — the stand-in run was stopped before it executed).
- Download/cache cell, sha256 match, ONNX session load in Colab.
- Interface probe: variable lengths (32k/48k/96k), batch = 2, determinism.
- Audio decoding of your formats (m4a/opus need ffmpeg; Colab has it).
- Silence/activity thresholds on real recordings.
- Label mapping 0 = genuine / 1 = synthetic (from upstream source only).
- Which preprocessing mode the checkpoint actually expects.
- Any prediction quality. No scores exist yet.

**Next**: run the notebook in Colab with 1 genuine + 1 synthetic clip; paste outputs of steps 3, 5, 10 and the CSV.

## 2026-10-24 — test clips

- Genuine: `data/colab_upload/genuine__en__aman.m4a`. Renamed from `genuine_hi_aman 2.m4a` (it's English); byte-identical, sha256 `f827f1b1c85fb7f0655ef02bd6c6cf28899edc6b0591c2156f88d97a38b160e9`, AAC mono 48 kHz, 8.21 s, 64 kbps. Recorded by Aman.
- Synthetic: **not fetched locally.** MLAAD audio and `meta.csv` are gated (HTTP 401 without an HF login that has accepted the terms). ASVspoof 2019 LA is only distributed as a full archive. New notebook step **5b** fetches one pre-registered clip in Colab instead: MLAAD @ `30c3dec`, `fake/en/Chatterbox`, rows sorted by path then shuffled with seed 20261024, first clip of 5–10 s; full metadata row saved to a sidecar JSON and the CSV column `source_meta`.
- Why Chatterbox: zero-shot voice cloning, and not one of the six generators in Dhwani's English training source (garystafford/deepfake-audio-detection: ElevenLabs, Kokoro, Hume, Luvvoice, Polly, Speechify, per that dataset's current card). Not verified against the exact dataset version Dhwani used.

## 2026-10-24 — baseline smoke test completed in Colab (run 20261024T102407Z)

Preserved in `runs/20261024T102407Z_dhwani_baseline_smoke/`: the executed notebook (byte-identical to `notebooks/01_dhwani_baseline_inference.ipynb`, which stays untouched), the predictions CSV, and the MLAAD sidecar JSON pulled from the notebook output.

| Clip | sha256 | Duration | train_peak | train_zscore | api_replica |
|---|---|---|---|---|---|
| `genuine_hi_aman.m4a` (English, Aman; label overridden to genuine/en) | `8b03de09…` | 8.256 s, AAC 48 kHz mono | 0.0001 | 0.0001 | 0.0001 |
| MLAAD `fake/en/Chatterbox/emerald_city_of_oz_20_f000055.wav` | `9448a599…` | 7.44 s, PCM16 22.05 kHz mono | 0.9998 | 0.9901 | 0.9916 |

Scores are uncalibrated synthetic-class softmax means. All 6 runs status ok; both clips are on the correct side of 0.5 under all three modes. **Two clips are a smoke test, not an accuracy benchmark, and they do not establish which preprocessing mode is correct.** `train_peak` stays the provisional default.

Other findings from this run:
- Label mapping 0 = genuine / 1 = synthetic is consistent with both clips (weak evidence, n = 2).
- **Batch of 2 drops a row** (output stays `[1, 2]`), so inference must use batch = 1. The probe in `dhwani_infer.interface_self_test` has been fixed to report this as not ok.
- Latency on Colab CPU: about 1.8 s per 4 s window, so 5.6–8.8 s per 7–8 s clip in train modes and about 1.8 s in api_replica.
- m4a decodes via the audioread fallback (soundfile can't read AAC); those warnings are now silenced.
- **File note:** the genuine clip used in Colab (`./genuine_hi_aman.m4a`, sha `8b03de09…`; `./genuine_hi_aman 2.m4a` is identical) is **not** the file I renamed earlier to `data/colab_upload/genuine__en__aman.m4a` (sha `f827f1b1…`, a different recording). All three files left untouched.
- Provenance of the MLAAD clip is in the CSV only via filename; the full `meta.csv` row is in the run folder's sidecar JSON. Newer builds also write it to the CSV column `source_meta`.

Synced into source: the executed 5b cell verbatim (`src/notebook_cells/mlaad_fetch_5b.py`). The builder now writes to `notebooks/build/` and never touches the executed notebook. `dhwani_infer.analyse_waveform()` added for in-memory audio (same validation, preprocessing and windowing as files).

## 2026-10-24 — FastAPI backend prepared (not yet run)

- `src/vcd/vcd_service.py`: model lifecycle + serialised inference, no HTTP. `detect_file()` now; `detect_waveform()` for remote-call windows later. Loads once and verifies sha256 (reusing notebook 01's marker); never downloads unless `VCD_ALLOW_DOWNLOAD=1`.
- `src/vcd/vcd_api.py`: `GET /health` (open: readiness, attribution, no audio) and `POST /detect` (multipart `file`, `X-API-Key` header, optional `?mode=`).
  - Status codes: 200 ok, 422 rejected (with `reject_reason`), 400 bad mode, 401 key, 413 > 8 MiB (checked before spooling; chunked bodies counted), 415 extension, 503 not ready or busy.
  - Single worker + lock = one inference at a time. The upload goes to a temp file that is deleted in `finally`.
  - The response carries `prediction`, `synthetic_score` (+max, `score_is_calibrated: false`, `threshold_is_tuned: false`), duration, `timing_ms`, `preprocessing`, per-window scores and model attribution.
- `notebooks/02_detector_api_colab.ipynb` (built by `scripts/build_api_notebook.py`): reuses this runtime's HF cache if present (a separate Colab notebook normally gets its own VM, so expect one 1.26 GB download unless you use the Drive cache).
  - The API key comes from Colab secret `VCD_API_KEY`, or is generated per session.
  - Step 8 runs 20+ checks: /health; 401s; both clips in train_peak and api_replica matched against the run 20261024T102407Z scores (±0.002, keyed by sha256); 400/413/415; silent, short, corrupt and empty → 422; 2 concurrent requests don't overlap; temp dir empty.
  - Step 9: Cloudflare quick tunnel (cloudflared 2026.9.3) → public `https://*.trycloudflare.com`. Step 10 re-tests /health and /detect over HTTPS.

**Checked locally:** all source files and every notebook code cell parse; the embedded sources are identical to `src/vcd/*`; `notebooks/build/01…` contains the verbatim executed 5b cell; the executed notebook's sha256 is unchanged (`3d7a4b20…`); the cloudflared 2026.9.3 linux-amd64 asset URL resolves (HTTP 200).
**Not verified (Colab):** nothing in the API has executed yet: FastAPI/uvicorn startup, the middleware's 413 behaviour, auth, the score match, serialisation, temp cleanup, the tunnel, and reachability from the phones. The backend is not complete until step 8 prints all checks passed and step 10 succeeds.

**Next action:** open `notebooks/02_detector_api_colab.ipynb` in Colab and run steps 1–8 (upload `genuine_hi_aman.m4a` at step 7). Paste the step 6 health JSON and the step 8 check lines.

## 2026-10-24 — React Native (Expo) upload demo: built, partly verified on emulator

Backend reported verified by Aman in Colab: 21/21 checks, public HTTPS /health + /detect with both clips (backend code unchanged since).

**Built**
- `mobile/`: Expo SDK 58, React Native 0.88, TypeScript; development-build setup (expo-dev-client), not Expo Go. `App.tsx` is the single screen; `src/api.ts` (types mirror `vcd_api.py`/`vcd_service.py` JSON), `src/storage.ts` (expo-secure-store), `src/mime.ts`. Run instructions and iOS notes are in `mobile/README.md`.
- Upload: XMLHttpRequest + FormData `{uri, name, type}` field `file`, header `X-API-Key`, no manual Content-Type (native boundary), 180 s timeout, cancel button, double-submit guard (ref + disabled buttons). Client pre-check for files over 8 MB.
- Errors mapped: 401 auth, 422 rejected (server's reason), 413, 415, 400, 503 busy / not ready, 502/530 tunnel down, network failure, timeout.
- Result screen: prediction ("Likely synthetic/genuine", untuned 0.5 threshold), "Synthetic score (uncalibrated)" with a not-a-probability note, max window score, duration, server processing time, total incl. upload, preprocessing (provisional), window count, request ID.
- APK: `mobile/dist/voice-clone-check-arm64-release-debugsigned.apk` (22.7 MB, sha256 `867da954…`). Release variant, JS bundle embedded (no Metro), signed with the debug keystore, arm64-v8a only.

**Verified on the Pixel_10a emulator (Android API 37), against the live tunnel**
- `npx tsc --noEmit` passes; `npx expo-doctor` 20/20; Gradle `assembleRelease` succeeds.
- APK installs and launches; no crash in logcat.
- URL entry + Save (SecureStore) works; **Check connection → "Connected · model ready"** (real `GET /health` over HTTPS).
- System document picker (DocumentsUI) → Downloads → `genuine_hi_aman.m4a`: filename preserved, size 71 KB, MIME `audio/mp4`.
- Upload reaches the server: with a deliberately wrong key the app shows **"Authentication failed (HTTP 401)"**.
- Test clips pushed to `/sdcard/Download/` (genuine `8b03de09…`, synthetic `9448a599…`, both matching the baseline hashes).

**NOT verified yet**
- A successful `/detect` from the app (prediction/score display) for either clip. With the key entered by Aman, the server still returned 401. The field held 39 characters, while a Colab-generated session key is 43, so the key was probably mistyped. Then the emulator was stopped.
- Multipart body acceptance by the server (a 401 is decided before the body is used), 422/413/415/503/timeout UI paths, the analysing-state screen, cancel.
- Real phones (Aman's and Harsh's), mobile-data networks, other file providers (WhatsApp audio, Google Drive, Files by Google), and armeabi-v7a-only devices.
- iOS: not built or tested (this Mac has no full Xcode). See `mobile/README.md`.
- `npx expo lint` not run (skipped to save disk).

**Next action:** start the emulator (or install the APK on a phone: `adb install -r mobile/dist/voice-clone-check-arm64-release-debugsigned.apk`), paste the exact API key into the app (long-press → Paste), Save, then analyse `genuine_hi_aman.m4a` and the Chatterbox clip from Downloads. Expected from the baseline run: synthetic score ≈ 0.0001 and ≈ 0.9998 (train_peak).

## 2026-10-24 — Offline on-device detection (Android): working on emulator; reference-tensor check pending

Server-upload app preserved: `mobile/reference/server-upload-v1/` + `mobile/dist/voice-clone-check-arm64-release-debugsigned.apk`. Colab backend unchanged.

**Compatibility (inspected without downloading the model to the Mac: HTTP Range reads of the pinned file)**
- ONNX IR 8, opset 18 (`ai.onnx` only), producer PyTorch 2.8.0, 1,888 nodes, standard ops only (Conv, MatMul, LayerNormalization, Erf, Softmax, Gather, Where, ReduceL2, Gemm, Elu, LeakyRelu, …).
- Runtime chosen: `onnxruntime-android` **1.31.0** (same version as the Colab reference run) via a native Kotlin Expo module (`mobile/modules/vcd-detector`). Not `onnxruntime-react-native`, which is at 1.24.3 and uses React Native's old bridge.

**Implemented**
- Pinned one-time model install into app-private storage (`filesDir/models/dhwani-ee01871/best_model.onnx`): `.part` + HTTP Range resume, SHA-256 verify, atomic rename, progress UI. Not in the APK, JS bundle or git.
- One ORT session (CPU, ALL_OPT, 4 intra-op threads, sequential, batch 1) on a dedicated inference thread.
- Native decode (WAV parser; MediaCodec for m4a/AAC), mono, Kaiser-sinc resampler approximating soxr_hq, Kotlin port of `dhwani_infer.py` (validation, windows, 3 modes, softmax, mean).
- `analysePcm()` for future in-app call audio (in-memory float32 PCM).
- Removed ORT's `TelemetryInitializer` (it starts an HTTP client at app launch) via `tools:node="remove"`.
- Added R8 keep rules for `ai.onnxruntime.**`: release minification had stripped JNI-constructed classes and the app aborted in `OrtSession.getInputInfo`.
- APK: `mobile/dist/voice-clone-check-offline-arm64-release-debugsigned.apk` (57 MB, arm64-v8a, sha256 `e51a2244…`).

**Verified on the Pixel_10a emulator (Android API 37, arm64, 4 GB guest RAM, on the MacBook Air M4)**
- In-app download from the pinned HF revision; resume after reinstall at byte 564,810,910 (HTTP 206); SHA-256 verified; finalised.
- Cold model load 3.99 s (right after install) and 43.9 s (after app restart, with the host under heavy memory pressure). Peak RSS during load ~1.50 GB; with model loaded + inference: VmHWM 1,574 MB, PSS 1,436 MB (`dumpsys meminfo`). No crash or LMK kill of the app after the R8 fix.
- Both exact reference clips via the real system picker (DocumentsUI → Downloads; hashes `8b03de09…`, `9448a599…`), all three modes. Results are in `runs/20261024_android_emulator_ondevice/`:

| clip | mode | windows | device score | Colab CSV | max abs logit diff vs CSV (4 dp) |
|---|---|---|---|---|---|
| genuine | train_peak | 4 | 0.00012348 | 0.0001 | 7.2e-05 |
| genuine | train_zscore | 4 | 0.00012148 | 0.0001 | 3.3e-04 |
| genuine | api_replica | 1 | 0.00013221 | 0.0001 | 4.2e-04 |
| synthetic | train_peak | 3 | 0.99980446 | 0.9998 | 2.1e-04 |
| synthetic | train_zscore | 3 | 0.99003038 | 0.9901 | 3.3e-03 |
| synthetic | api_replica | 1 | 0.99155436 | 0.9916 | 2.5e-03 |

  Same predictions, durations, window starts and validation stats (peak/RMS/active) in every case. The AAC decode length matched Colab exactly (396,288 frames at 48 kHz).
- **Offline:** airplane mode on (no default network, ping unreachable), app cold-restarted, old dumps deleted, both clips re-analysed. The logits were identical to the online run, and the app's network counters were unchanged before/after. No ORT telemetry log lines after the manifest fix.
- Speed (emulator): steady state 1.9–2.6 s per 4 s window (synthetic WAV total 6.8 s; WAV decode 2 ms, resample 75 ms; m4a decode ~1 s). Outliers of 13–37 s per window coincided with the Mac at ~145 MB free RAM, 6 GB compressed and heavy swap. These are emulator/host effects, not phone numbers.

**Not verified / open**
- **Milestone 1 reference-tensor check not run yet:** waiting for `vcd_reference_export.zip` from `notebooks/colab_cells/reference_export_for_device.py`. Without it, logit diffs up to 3.3e-3 (z-score modes, 22.05 kHz WAV) can't be attributed to resampling (soxr vs ours) versus ARM vs x86 floating point.
- Real phones: memory headroom (~1.5 GB RSS for the app alone), speed, thermal throttling, LMK behaviour on 6 GB devices, cold load from flash. **Not established that this suits 6 GB phones.**
- iOS: module is a stub; nothing built or tested.
- Two clips verify numerical compatibility only, not accuracy or multilingual robustness. Scores are uncalibrated; "genuine" is not proof a caller is safe.
- Mac storage: free space fell to 1.0 GB during testing (macOS swap under memory pressure) and recovered to 1.4 GB after the emulator stopped. Further rebuilds or emulator sessions need ≥2 GB free; this is a practical blocker.

## 2026-10-24 — Reference-tensor check done; detector made replaceable

**Reference export:** `vcd_reference_export.zip` (sha256 `7615a446…`), made in a separate Colab runtime with onnxruntime 1.31.0 (CPU, ORT_ENABLE_ALL), librosa 0.11.0, soxr 1.1.0, ffmpeg 6.1.1. All 10 arrays matched their manifest sha256 and shapes. A second baseline CSV (`dhwani_predictions_20261024T124533Z.csv`) reproduced the earlier logits. Evidence is in `runs/20261024_android_emulator_ondevice/` (`reference_stage_comparison.json`, `reference_export_manifest.json`, `screen_reference_tensor_check.png`).

**Runtime only:** all 16 exported Colab windows (both clips × 3 modes) run through onnxruntime-android 1.31.0 on the emulator gave **max |Δlogit| = 2.38e-6** (per window 0 to 2.4e-6). ~600 ms/window, cold load 5.3 s in this session (host not memory-starved).

**Preprocessing stages** (device vs Colab, full precision):

| Stage | genuine (AAC m4a 48 kHz) | synthetic (PCM16 WAV 22.05 kHz) |
|---|---|---|
| decode | max 3.05e-5 (= 1 LSB of int16), RMS 2.4e-6, same length, lag 0, SNR 73.2 dB | bit-exact |
| resample (ours vs soxr_hq) | max 4.3e-5, SNR 72.3 dB | max 1.98e-4, SNR 77.4 dB |
| windows | train_peak max 3.9e-4; train_zscore max 3.8e-3 (normalisation gain); SNR 72–75 dB | train_peak max 1.5e-4, SNR 75–80 dB |
| logits | train_peak 3.4e-5; train_zscore 3.6e-4; api_replica 3.9e-4 | train_peak 1.6e-4; train_zscore 3.3e-3; api_replica 2.5e-3 |
| clip score abs diff | ≤ 2e-7 | ≤ 5.2e-5 |
| prediction | identical (all modes) | identical (all modes) |

Interpretation: the runtime contributes ≤2.4e-6. The synthetic WAV decodes bit-exactly, so its logit gaps (up to 3.3e-3) come from resampling (soxr_hq vs our Kaiser-sinc), amplified by z-score normalisation. Genuine gaps combine 1-LSB AAC decoder rounding with resampling. Preprocessing was not changed to force agreement. Working tolerances observed: runtime ≤1e-5 logit; full pipeline ≤5e-3 logit and ≤1e-4 clip score on these two clips. These are observations on two clips, not guarantees.

Not covered: device window dumps for genuine api_replica and synthetic train_zscore/api_replica (UI automation missed those taps; their logits were compared from earlier runs with identical preprocessing code).

**Replaceable detector (small refactor, Dhwani preserved as baseline):**
- Native `DetectorAdapter` + declarative `AdapterSpec` (version, pinned artifact + sha256, sample rate, window/hop, duration limits, batch size, modes + default, output labels + positive index, aggregation, threshold + tuned flag, calibration flag, attribution). `DhwaniAdapter` is the only registered adapter.
- `DetectorEngine`, `ModelStore` (directory `models/<id>-<rev7>/`, so the installed `dhwani-ee01871` model is reused) and `OnnxWindowSession` (renamed from DhwaniSession) are model-agnostic. New JS calls: `getAdapterSpec()`, `listAdapters()`.
- TS `src/detector` `Detector` interface (`analyseFile`, `analysePcm`, lifecycle). `App.tsx` uses only that; modes and default come from the adapter spec.
- Verified: `tsc --noEmit` passes; `:vcd-detector:compileReleaseKotlin` succeeds. **Not verified:** a rebuilt APK on the emulator. The installed APK and `mobile/dist/voice-clone-check-offline-arm64-release-debugsigned.apk` (sha256 `e51a2244…`) are the pre-refactor, verified build. Rebuild + emulator smoke test once the Mac has ≥2 GB free.

**Disk root cause:** three idle Gradle daemons (one at 2.7 GB RAM) plus the emulator pushed macOS swap to 7.6/8 GB, and free disk fell to 486 MB. `gradlew --stop` restored it to 2.4 GB; it was 1.2 GB at the end of this session. Always run `./gradlew --stop` after builds; shut the emulator down when idle.

## 2026-10-24 — Paused; integration handoff written

- Builds and emulator runs are paused until the Mac has ≥2 GB free and the evaluation session supplies a candidate. No further refactoring.
- Preserved unchanged: pre-refactor offline APK (sha256 `e51a2244…`), server-upload APK (`867da954…`), the model installed on the Pixel_10a AVD, and `vcd_reference_export.zip`. Nothing deleted, nothing downloaded.
- Adapter layer status unchanged: **compiles, unverified at runtime.**
- `docs/DETECTOR_INTEGRATION_HANDOFF.md`: exact adapter files, the model input/output contract the engine enforces, the registration steps, and the parity/offline/crash checks (with Dhwani's measured results as the bar).
- Note: two Gradle daemons from VS Code's Java extension (not from this project's builds) were running at the end of the session, reserving up to 2.5 GB of RAM. Left alone.

## 2026-10-24 (evening) — Landing screen rebuilt; compact result screen; manual testing handed over

**Installed on the Pixel_10a emulator and in `testing apk/`:** `voice-check-landing-arm64-release-debugsigned.apk`, sha256 `58bf36b2…`, versionName 1.0.0, package `com.hypenex.voiceclonecheck` (same package as the verified demo build, so the installed model is reused). The earlier demo APKs are untouched in `mobile/dist/` (`e51a2244…` offline demo, `867da954…` server-upload).

**UI (React Native, no new dependencies):** `src/screens/HomeScreen.tsx` (home / selected / analysing), `src/screens/ResultScreen.tsx` (dedicated result: Back, "Analysis result", one prediction, plain-language explanation, duration + analysis + model-load row, Check another file, disclaimer "Experimental result. Verify the request through another channel."), `src/screens/DiagnosticsScreen.tsx` (Dev: now also shows the last check's raw score, threshold note, full filename, preprocessing, per-window scores/logits/ms, timing breakdown, attribution). `src/components/Dial.tsx`, `Icons.tsx`, `theme.ts`. Root `App.tsx` switches screens and handles Android hardware back. Inference, adapter and preprocessing code unchanged.

**Verified on the emulator (screenshot-first replay, run 20:17–20:21; evidence in `runs/20261024_landing_replay/`):**
- Home renders with the primary action on screen; Dev opens; hardware back returns home; app process stable (pid 13873 throughout).
- genuine_hi_aman.m4a → result screen "Likely genuine", duration 8.3 s, **analysis 15.9 s, model load 21.3 s** (shown separately; cold load in this process). Result screen replaced the home content (home not visible). Back → result; Check another → home idle.
- synthetic Chatterbox clip → "Likely synthetic", 7.4 s, analysis 10.3 s (model already loaded, no load row).
- too_short_1s.wav → rejection screen "This clip couldn’t be checked".
- Analysing state visible: screenshots at 2 s and 6 s after tapping Analyse show the busy row with "2 s elapsed" → "6 s elapsed", the dial's orange ticks moved, Dev greyed. The counter advancing during native inference shows the JS thread stays responsive.
- Dev shows "Model loaded" after loading from the landing screen (shared native session; Dev re-reads status on show).

**Timing note (56.7 s question):** that figure was analysis only; model load is a separate call and was not included. The UI now shows them separately. Measured today on this emulator: analysis 10–16 s per 7–8 s clip when the host is lightly loaded, 57–68 s when the Mac was swapping heavily (Gradle daemon + emulator). Cold load 4–44 s under the same conditions. These are emulator numbers, not phone numbers.

**Not verified / open:**
- The replay was stopped before the offline (airplane-mode) cold restart on *this* build; the offline path was verified on the previous landing build (`738024d8…`) and on the pre-refactor demo build. Same inference code.
- The "Try again" path from the error screen and the model-download UI on a fresh install (model was already installed on the emulator) were not exercised.
- Earlier automated runs reported empty readings during the analysing state and one unexplained return to the idle home screen; both traced to `uiautomator` returning stale UI trees while the dial animates (and two `uiautomator` instances crashing each other). Logcat confirms the app never crashed. Treat those earlier automation readings as invalid, not as app failures.
- Real phones, iOS, accuracy, multilingual robustness: unchanged, unverified.

**Repo repair (staged, not committed):** `mobile` was a gitlink (mode 160000 → the Expo template's local "Initial commit"), so GitHub showed an empty folder; `mobile.zip` (694 MB) was committed locally in `1956e34`. Nested history preserved in `runs/mobile-nested-repo-cdaee38.bundle` and `mobile/.git-nested-backup/`; gitlink replaced by the real tree; `.gitignore` now excludes node_modules, generated `mobile/android` and `mobile/ios`, module build dirs, APKs, zips, model weights and audio. Staged list: `runs/staged_files_20261024.txt`.

## 2026-10-24 (night) — In-app calling with live check (branch `live-calling`)

Source: Devansh Modi's LiveCalling kit (github.com/DevanshModi09/LiveCalling @ `1a01138`): WebView-hosted WebRTC audio call, Node signaling server, 16 kHz remote-PCM bridge. Vendored `server/`, `web/`, `deploy/` into `calling/` and `WebRtcCallView.kt` (package renamed) into our native module. Kit's own 6 server tests pass on the vendored copy.

**Implemented**
- `LiveAnalyser.kt`: rolling 10 s buffer of the *remote* participant's PCM; every 2 s scores the latest 4 s window through the **shared** `DetectorEngine` (now a process singleton) using the adapter's default mode; skips windows while one is in flight; never scores the local mic. `analysePcm` gained a float overload (byte version delegates; inference unchanged).
- `VcdCallView` / `VcdCallingModule`: Expo view in the existing `vcd-detector` Gradle module (`serverUrl`, `liveCheck`, `onCallState`, `onLiveResult`). RECORD_AUDIO/MODIFY_AUDIO_SETTINGS/INTERNET added to the module manifest; R8 keep for the JS bridge.
- `CallScreen.tsx`: server or invitation entry (server remembered in SecureStore), runtime mic permission, pre-loads the model if installed, hosts the call view with a live strip (latest window verdict + uncalibrated score + skipped count). Reached from Home ("start an in-app call"); hardware back returns.
- APK: `mobile/dist/voice-check-livecalling-arm64-release-debugsigned.apk` (sha256 `f3fa1dd0…`), installed on the emulator (same package; model kept).

**Verified on the emulator (`runs/20261024_live_calling_check/`)**: Home shows the call entry; call screen renders; empty and `http://` inputs rejected with a message; mic permission dialog shown and grant works; with `https://unreachable.invalid/` the native view reports "Could not load calling server…" via `onCallState` and the live strip shows its waiting state; End call → setup; hardware back → Home. App process never restarted (the one logcat fatal is UiAutomation, my test tool).

**Not verified**: an actual two-party call and the remote-PCM → live-check path. The kit's WebView only accepts HTTPS, so a test needs either a deployed HTTPS calling server or a temporary tunnel to the local server; starting a tunnel from this session was blocked by the permission policy. Also unverified: audio quality/echo, TURN across networks, real phones, inference latency during a call, and behaviour when the model is not installed (code path: call runs, live check off).

## 2026-10-24 (late) — Live call smoke test: live-check path confirmed end to end; emulator lost to host memory

Setup: local `calling/server` behind a temporary Cloudflare quick tunnel (HTTPS origin, `ALLOWED_ORIGIN` set), room `822523f0d06b4a4c`; emulator rebooted with audio enabled; second participant joined from the Mac (Chrome), not automated (the Claude in Chrome extension is not connected; the built-in browser pane blocks microphone capture). Tunnel and server were shut down afterwards. Evidence: `runs/20261024_live_call_test/`.

**Verified**
- The emulator app joined the room over the WebView bridge; microphone capture worked (`-no-audio` had to be dropped).
- A real WebRTC call connected ("Call connected"), and the remote participant's audio flowed through the JS bridge → `LiveAnalyser` → shared `DetectorEngine` → `onLiveResult` → UI. The live strip reached **window 25** (≥ ~50 s of remote audio), showing "Likely genuine · score 0.0001–0.0002 (uncalibrated)" with per-window inference of **0.7–3.0 s** and 1–2 skipped windows (skip-when-busy working). Screenshots 01/03/04.
- Rejoin after "Call ended" worked ("Waiting for the other participant").

**Not verified / caveats**
- The remote audio was the Mac's microphone (ambient/speech); no synthetic clip was played into the call, so only the genuine side of the live verdict was exercised.
- Calls ended more than once with "Call ended" (local stop) during the automated steps; the cause was not isolated (my scripted taps on a shifting layout are the prime suspect). Needs a manual two-phone run.
- TURN/cross-network, audio quality, echo, and phone performance remain untested.

**Environment incidents (not app bugs)**
1. First attempt: the guest's `system_server` entered a watchdog restart loop (every ~72 s). Contributing: my join script ran `uiautomator dump` once per second (crash-looped it, 12 fatal exceptions at 1.1 s intervals), host swap at ~5.8 GB, and the emulator having been booted with `-no-audio`. Fixed by a forced restart with audio.
2. Second attempt: the emulator exited after its own watchdog saw the QEMU main loop and all CPU threads hang for 15 s, with host swap at 6.3 GB of 7 GB and 1.7 GB disk free. Not restarted: free host memory first (Codex, Android Studio and the Claude app each hold ~1 GB).

## 2026-10-24 (late) — Call reliability fix (branch `live-calling`, c76c104)

Reported: no audio either way, calls dropping within ~10 s. Diagnosis: the signaling server handed clients **no ICE servers** (`/config` → `{"iceServers": []}`), so peers on different networks (or on mobile data) never found a media route; ICE timed out and the kit hung up with "Connection failed". The kit also ended the call whenever the signaling socket dropped.

Changes (server + web page; no app rebuild needed for these, the WebView loads the page from the server):
- `calling/server/server.mjs`: CLI start defaults to public STUN; `TURN_URLS` / `TURN_USERNAME` / `TURN_CREDENTIAL` add TURN; `ICE_SERVERS_JSON` still overrides. `createServer()` default unchanged (kit tests still 6/6). Startup log prints the ICE servers and warns when no TURN is set.
- `calling/web/call.js`: a signaling drop no longer ends a connected call; `disconnected`/`failed` → ICE restart from the initiator (up to 3 tries) instead of an immediate stop; the status line shows the media route once connected (direct / NAT traversal / via TURN relay); failure text names STUN/TURN.
- `VcdCallView.kt`: `keepScreenOn` during a call (a sleeping screen pauses the WebView).
- `calling/deploy/turnserver.conf.example` + README section on STUN/TURN.

Verified locally: `/config` returns STUN by default, TURN entries when env is set, and the override; kit tests pass; Kotlin compiles. **Not verified:** an actual cross-network call with these changes (needs the server redeployed with the new `web/` and `server/`, and TURN for mobile-data callers).
- APK rebuilt with the screen-on fix: `mobile/dist/voice-check-livecalling-arm64-release-debugsigned.apk` (sha256 `d139f2f213e5…`), copied to `testing apk/`.
