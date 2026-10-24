# Screenshots for the hackathon deck

All captured from the real build (Android emulator, Pixel 10a, arm64 release APK) and the live
calling server. Nothing is mocked. Suggested slide use in the right-hand column.

| File | What it shows | Suggested slide |
|---|---|---|
| 01_home.png | Landing screen: "A second check. For every voice.", Select audio, Call a friend | Solution / product |
| 02_home_clip_selected.png | Clip chosen, "Analyse on device" and Call button | Flow step 1 |
| 03_analysing_on_device.png | Analysing state (offline, on-device) | Flow step 2 |
| 04_result_likely_genuine.png | Result screen for the real recording | Flow step 3 / demo |
| 05_result_likely_synthetic.png | Result screen for the Chatterbox TTS clip (MLAAD) | Flow step 3 / demo |
| 06_result_mixed_voices_timeline.png | 20 s clip, 8 s real + 7 s AI + 5 s real → "Mixed voices" with segment timeline | Unique points |
| 07_dev_technical_details.png | Dev screen after a check: raw score, windows, timings | Technical approach |
| 08_dev_diagnostics.png | Dev screen: model status, checksum, memory | Technical approach |
| 09_in_call_waiting.png | In-app call screen with live strip, waiting for the other participant | Live call feature |
| 10_in_call_ai_voice_toggle.png | "Play AI voice into call" demo toggle switched on | Live call demo |
| 11_call_screen.png | Call screen: Start a call / AI demo / join with link | Live call feature |
| 12_ai_caller_demo_room.png | AI caller demo room with "Open AI caller page" | Live call demo |
| 13_web_call_page.png | Web call page on the Lightsail server (invite link target) | Architecture |
| 14_web_ai_caller_page.png | AI caller web page that streams a synthetic voice into the room | Live call demo |

Notes for the presenter
- Scores shown in Dev are uncalibrated model outputs, not probabilities; say "likely", not "x% AI".
- The detector is the open Dhwani checkpoint (ayush2635, Hugging Face, revision ee01871), ported to run on-device; we did not train it.
- The AI voice sample is Chatterbox TTS speech from the MLAAD dataset (CC-BY-NC-4.0, demo only).
