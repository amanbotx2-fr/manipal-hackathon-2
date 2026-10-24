import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { Analysis, PickedFile } from '../analysis';
import { colors, space, type } from '../theme';
import { PrimaryButton, SecondaryButton } from './HomeScreen';

// Dedicated, compact result screen. Shows the prediction once in plain language. The raw score,
// threshold, windows, preprocessing and timings breakdown live in Dev (DiagnosticsScreen).

export type ResultProps =
  | { kind: 'analysis'; analysis: Analysis; onBack: () => void; onAnother: () => void; onOpenDev: () => void }
  | { kind: 'error'; title: string; detail: string; file: PickedFile; onBack: () => void; onAnother: () => void; onRetry: () => void };

export default function ResultScreen(p: ResultProps) {
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Pressable testID="result-back" onPress={p.onBack} hitSlop={12} accessibilityRole="button" accessibilityLabel="Back">
          <Text style={styles.back} maxFontSizeMultiplier={1.3}>‹ Back</Text>
        </Pressable>
        <Text style={styles.headerTitle} maxFontSizeMultiplier={1.3}>Analysis result</Text>
        <View style={{ width: 56 }} />
      </View>
      <View style={styles.rule} />
      {p.kind === 'error' ? <ErrorBody {...p} /> : <AnalysisBody {...p} />}
    </ScrollView>
  );
}

function AnalysisBody({ analysis, onAnother, onOpenDev }: Extract<ResultProps, { kind: 'analysis' }>) {
  const { result: r, loadMs } = analysis;
  if (r.status !== 'ok') {
    return (
      <View style={styles.body}>
        <Text style={[type.eyebrow, styles.eyebrow]}>Clip not analysed</Text>
        <Text style={styles.verdict} testID="result-rejected" maxFontSizeMultiplier={1.3}>This clip couldn’t be checked</Text>
        <Text style={[type.body, styles.explain]}>{plainReason(r.reject_reason)}</Text>
        <Text style={[type.small, styles.explain]}>Clips need 2–10 seconds of clearly audible speech.</Text>
        <PrimaryButton testID="result-another" label="Choose another file" onPress={onAnother} />
      </View>
    );
  }
  const analysisS = r.timing_ms.total / 1000;
  // Timeline: full-clip windows when available (clip > 10 s), else the single-verdict path's own windows.
  const winLen = analysis.timeline?.window_s ?? 4;
  const segs = analysis.timeline?.windows?.map((w) => ({ start: w.start_s, end: w.end_s, score: w.synthetic_score ?? null, status: w.status }))
    ?? r.windows.map((w) => ({ start: w.start_s, end: Math.min(w.start_s + winLen, r.duration_s ?? w.start_s + winLen), score: w.synthetic_score, status: 'ok' as const }));
  const total = analysis.timeline?.duration_s ?? r.duration_s ?? 0;
  const scored = segs.filter((x) => x.status === 'ok' && x.score != null);
  const nSyn = scored.filter((x) => (x.score as number) >= r.decision_threshold).length;
  const mixed = scored.length > 1 && nSyn > 0 && nSyn < scored.length;
  const allSyn = scored.length > 0 && nSyn === scored.length;
  const synSpan = analysis.timeline?.synthetic_span_s ?? mergedSpan(scored.filter((x) => (x.score as number) >= r.decision_threshold));
  const synthetic = analysis.timeline ? allSyn : r.prediction === 'synthetic';
  return (
    <View style={styles.body}>
      <Text style={[type.eyebrow, styles.eyebrow]}>Checked on this device</Text>
      <Text style={[styles.verdict, { color: mixed ? '#A15C00' : synthetic ? colors.danger : colors.ok }]} testID="result-prediction" maxFontSizeMultiplier={1.3}>
        {mixed ? 'Mixed voices' : synthetic ? 'Likely synthetic' : 'Likely genuine'}
      </Text>
      {segs.length > 1 && <Timeline segs={segs} total={total} threshold={r.decision_threshold} />}
      {mixed && (
        <Text style={[type.body, styles.explain]} maxFontSizeMultiplier={1.3} testID="result-mixed">
          Likely synthetic speech in about {synSpan.toFixed(0)} s of this {total.toFixed(0)} s clip (orange segments); the rest reads as genuine. Segments are 4-second windows, so the boundary is approximate.
        </Text>
      )}
      <Text style={[type.body, styles.explain]} maxFontSizeMultiplier={1.3}>
        {synthetic
          ? 'The detector found patterns it associates with AI-generated speech. It can be wrong, especially for voices, languages or recordings unlike the ones it was built on.'
          : 'The detector did not find patterns it associates with AI-generated speech. That does not confirm who is speaking or that the request is legitimate.'}
      </Text>

      <View style={styles.metaRow}>
        <Meta label="Duration" value={`${r.duration_s?.toFixed(1) ?? '–'} s${r.truncated ? '*' : ''}`} testID="result-duration" />
        <View style={styles.metaDivider} />
        <Meta label="Analysis" value={`${analysisS.toFixed(1)} s`} testID="result-time" />
        {loadMs != null && (
          <>
            <View style={styles.metaDivider} />
            <Meta label="Model load" value={`${(loadMs / 1000).toFixed(1)} s`} testID="result-load" />
          </>
        )}
      </View>
      {r.truncated && <Text style={type.fine}>* Only the first 10 seconds were analysed.</Text>}

      <PrimaryButton testID="result-another" label="Check another file" onPress={onAnother} />
      <Pressable testID="result-dev" onPress={onOpenDev} hitSlop={8} style={styles.devLink} accessibilityRole="button">
        <Text style={[type.small, styles.devLinkText]}>Technical details in Dev ›</Text>
      </Pressable>

      <Text style={[type.small, styles.disclaimer]} testID="result-disclaimer" maxFontSizeMultiplier={1.3}>
        Experimental result. Verify the request through another channel.
      </Text>
    </View>
  );
}

function ErrorBody({ title, detail, onAnother, onRetry }: Extract<ResultProps, { kind: 'error' }>) {
  return (
    <View style={styles.body}>
      <Text style={[type.eyebrow, styles.eyebrow]}>Something went wrong</Text>
      <Text style={[styles.verdict, { color: colors.danger }]} testID="result-error" maxFontSizeMultiplier={1.3}>{title}</Text>
      <Text style={[type.body, styles.explain]}>{detail}</Text>
      <View style={styles.actions}>
        <SecondaryButton testID="result-another" label="Choose another file" onPress={onAnother} />
        <PrimaryButton testID="result-retry" label="Try again" onPress={onRetry} compact />
      </View>
    </View>
  );
}

function plainReason(reason: string | null) {
  const r = reason ?? '';
  if (/too short/i.test(r)) return `The clip is too short. ${r}.`;
  if (/mostly silent|silent/i.test(r)) return `The clip is silent or nearly silent. ${r}.`;
  if (/unreadable|empty/i.test(r)) return `The file could not be read as audio. ${r}.`;
  return r || 'The audio was rejected before analysis.';
}

function mergedSpan(segs: { start: number; end: number }[]) {
  let span = 0, cs = -1, ce = -1;
  for (const s of [...segs].sort((a, b) => a.start - b.start)) {
    if (ce < 0 || s.start > ce) { if (ce >= 0) span += ce - cs; cs = s.start; ce = s.end; } else ce = Math.max(ce, s.end);
  }
  return ce >= 0 ? span + (ce - cs) : 0;
}

/** Horizontal bar of the clip: each window drawn from start to end, orange = likely synthetic, green = genuine, grey = not enough speech. */
function Timeline({ segs, total, threshold }: { segs: { start: number; end: number; score: number | null; status: string }[]; total: number; threshold: number }) {
  if (!total) return null;
  return (
    <View style={styles.tlWrap} testID="result-timeline">
      <View style={styles.tlBar}>
        {segs.map((s, i) => {
          const c = s.status !== 'ok' || s.score == null ? '#C8C7BF' : s.score >= threshold ? colors.danger : colors.ok;
          return <View key={i} style={[styles.tlSeg, { left: `${(100 * s.start) / total}%`, width: `${(100 * (s.end - s.start)) / total}%`, backgroundColor: c }]} />;
        })}
      </View>
      <View style={styles.tlAxis}><Text style={type.fine}>0 s</Text><Text style={type.fine}>{total.toFixed(0)} s</Text></View>
      <Text style={type.fine}>Green: likely genuine · Orange: likely synthetic · Grey: not enough speech. 4 s windows, every 2 s.</Text>
    </View>
  );
}

function Meta({ label, value, testID }: { label: string; value: string; testID?: string }) {
  return (
    <View style={styles.meta}>
      <Text style={type.fine} maxFontSizeMultiplier={1.3}>{label}</Text>
      <Text style={styles.metaValue} testID={testID} maxFontSizeMultiplier={1.3}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { flexGrow: 1, paddingHorizontal: space.lg, paddingBottom: space.lg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: space.md },
  back: { fontSize: 17, color: colors.ink, fontWeight: '600', width: 56 },
  headerTitle: { fontSize: 17, fontWeight: '600', color: colors.ink },
  rule: { height: 1, backgroundColor: colors.hairline },
  body: { flex: 1, paddingTop: space.xl },
  eyebrow: { marginBottom: space.sm },
  verdict: { fontSize: 34, lineHeight: 40, fontWeight: '700', color: colors.ink, letterSpacing: -0.5, marginBottom: space.md },
  explain: { marginBottom: space.lg },
  metaRow: { flexDirection: 'row', alignItems: 'center', borderTopWidth: 1, borderBottomWidth: 1, borderColor: colors.hairline, paddingVertical: space.md, marginBottom: space.lg },
  meta: { flex: 1, gap: 2 },
  metaDivider: { width: 1, height: 34, backgroundColor: colors.hairline, marginHorizontal: space.md },
  metaValue: { fontSize: 18, fontWeight: '600', color: colors.ink },
  actions: { flexDirection: 'row', gap: 10, marginTop: space.sm },
  tlWrap: { marginBottom: space.md },
  tlBar: { height: 18, borderRadius: 6, backgroundColor: colors.hairline, overflow: 'hidden', position: 'relative' },
  tlSeg: { position: 'absolute', top: 0, bottom: 0, opacity: 0.85 },
  tlAxis: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 2, marginBottom: 4 },
  devLink: { alignSelf: 'center', paddingVertical: space.md },
  devLinkText: { color: colors.inkSoft, textDecorationLine: 'underline' },
  disclaimer: { marginTop: 'auto', textAlign: 'center', paddingTop: space.lg },
});
