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
  const synthetic = r.prediction === 'synthetic';
  const analysisS = r.timing_ms.total / 1000;
  return (
    <View style={styles.body}>
      <Text style={[type.eyebrow, styles.eyebrow]}>Checked on this device</Text>
      <Text style={[styles.verdict, { color: synthetic ? colors.danger : colors.ok }]} testID="result-prediction" maxFontSizeMultiplier={1.3}>
        {synthetic ? 'Likely synthetic' : 'Likely genuine'}
      </Text>
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
  devLink: { alignSelf: 'center', paddingVertical: space.md },
  devLinkText: { color: colors.inkSoft, textDecorationLine: 'underline' },
  disclaimer: { marginTop: 'auto', textAlign: 'center', paddingTop: space.lg },
});
