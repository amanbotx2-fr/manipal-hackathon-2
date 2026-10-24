# Voice Check — UI / design handoff (for Devansh)

Everything the "Voice Check" screens need, lifted from the working React Native app. Pure React
Native + TypeScript: **no icon, SVG, font or navigation libraries**. Only `expo-document-picker`
(file picker) and `react-native-safe-area-context` (safe areas) are used.

`screenshots/00_reference_design.png` is the original design; the other screenshots are the real
app on an emulator (home, selected, analysing, result, rejection, Dev).

## Files

| File | What it is |
|---|---|
| `src/theme.ts` | Colours, spacing, type scale (the design tokens) |
| `src/components/Dial.tsx` | The circular dial: 72 radial ticks, 3 orange accent ticks, centred waveform, caption. `active` makes the accent ticks revolve and the waveform pulse (motion only, no fake progress) |
| `src/components/Icons.tsx` | Lock, file, upload, arrow and the sunburst logo, drawn with Views/Text |
| `src/screens/HomeScreen.tsx` | Landing screen with 3 states: **home** (headline, dial, Select audio), **selected** (file row, Change file / Analyse on device), **analysing** (dial animating, spinner, elapsed seconds). Also the model-setup card (download/progress/retry) |
| `src/screens/ResultScreen.tsx` | Dedicated compact result screen (Back, "Analysis result", one verdict, plain-language note, duration/analysis/model-load row, Check another file, disclaimer). Also renders rejections and errors |
| `src/screens/DiagnosticsScreen.tsx` | The "Dev" screen (optional): raw score, windows, timings, model install/load controls |
| `App.tsx` | Screen switching (home / result / dev) with Android hardware-back handling, no router |
| `src/analysis.ts` | The `Analysis` record passed from Home to Result |
| `src/mime.ts` | MIME/extension helper for picked files |
| `src/detector/index.ts`, `src/detector/types.ts` | The `Detector` interface the UI calls, and the result/status types. **Adapt `index.ts` to your inference** (see below) |

## Design tokens (`src/theme.ts`)

| Token | Value | Use |
|---|---|---|
| `bg` | `#F2F1EC` | warm light-grey background |
| `surface` / `surfaceStrong` | `#F8F7F3` / `#FFFFFF` | dial fill / cards |
| `hairline` | `#DAD9D2` | rules, borders, progress track |
| `tick` | `#C8C7BF` | dial ticks |
| `ink` | `#1F2A24` | charcoal-green text, outlines |
| `inkSoft` / `inkMute` | `#5E6962` / `#8A938D` | secondary / fine text |
| `accent` | `#F4592B` | primary button, accent ticks, spinner |
| `danger` / `ok` | `#B42318` / `#1E6B3A` | "Likely synthetic" / "Likely genuine" |

Type: headline 32/38 bold (-0.5 tracking), lede 16/22, body 15/21, small 13/18, fine 12/17,
eyebrow 12 uppercase with 2.4 letter-spacing. System font (Roboto on Android). All text has
`maxFontSizeMultiplier={1.3}` so large accessibility fonts don't break the layout.

Spacing: 6 / 10 / 16 / 24 / 32. Screen gutter 24. Buttons radius 14; Dev pill radius 999.

## Layout rules

- Dial size = `min(width × 0.62, height × 0.30, 256)`. This keeps the primary action on screen on a
  normal phone; the ScrollView only scrolls when content really overflows (small screens, big fonts).
  Nothing is clipped.
- Home bottom block uses `marginTop: 'auto'` so the Select/Analyse action sits at the bottom of the
  viewport, above the privacy note.
- Primary button: full-width orange, icon left + arrow right on "Select audio" (`spread`), plain
  centred label elsewhere. Compact variant (`compact`) for the two-button row.
- Result screen is a **separate screen**; results are never appended under the landing page.

## Copy (use verbatim)

- Eyebrow: `AUDIO AUTHENTICITY` · Headline: `A second check.` / `For every voice.`
- Lede: `Check a voice note for signs of AI-generated speech.`
- Dial captions: `Select an audio clip / to begin` · `Clip ready / tap Analyse on device` ·
  `Loading model / first use only` · `Analysing / on this device`
- Info row: `Voice notes & recordings` · `2–10 sec` · Button: `Select audio`
- Privacy: `Your audio stays on this device. Works offline after model setup.`
- Busy row: `Analysing on this device` / `Runs in the background · N s elapsed`
- Result: eyebrow `CHECKED ON THIS DEVICE`; verdict `Likely synthetic` or `Likely genuine`;
  explanation (synthetic): `The detector found patterns it associates with AI-generated speech. It can be wrong, especially for voices, languages or recordings unlike the ones it was built on.`
  (genuine): `The detector did not find patterns it associates with AI-generated speech. That does not confirm who is speaking or that the request is legitimate.`
- Disclaimer: `Experimental result. Verify the request through another channel.`
- Rejection: `This clip couldn’t be checked` + reason + `Clips need 2–10 seconds of clearly audible speech.`

Never label an output as verified confidence, accuracy, or proof that a caller is safe. The raw
score, threshold and technical values belong in Dev only.

## Wiring it to your inference

`HomeScreen` and `ResultScreen` call only `src/detector/index.ts`. Implement its `Detector`
interface over your model:

```ts
status(): Promise<ModelStatus>          // { installed, installing, loaded, partialBytes, expectedBytes, ... }
install(): Promise<void>                // one-time model download (progress via onInstallProgress)
onInstallProgress(cb): { remove() }
load(): Promise<LoadInfo>               // { loadMs, alreadyLoaded?, ... }
analyseFile(uri, opts?): Promise<LocalResult>
analysePcm(samples, sampleRate, channels, opts?): Promise<LocalResult>   // optional for now
```

`LocalResult` (see `types.ts`) needs at least: `status: 'ok' | 'rejected'`, `prediction`,
`synthetic_score`, `duration_s`, `truncated`, `reject_reason`, `timing_ms: { total, model_load?, ... }`,
`windows: []`, `preprocessing`. If your model is always installed, `status()` can return
`installed: true, loaded: true` and `install()` can resolve immediately; the setup card then never shows.

Then copy `App.tsx`, `src/` into your project, run `npx tsc --noEmit`, and remove `DiagnosticsScreen`
(and the `Dev` button in `HomeScreen`/`App.tsx`) if you don't want it.
