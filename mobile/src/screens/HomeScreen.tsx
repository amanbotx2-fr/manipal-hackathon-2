import * as DocumentPicker from 'expo-document-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View, useWindowDimensions } from 'react-native';

import type { Analysis, PickedFile } from '../analysis';
import Dial from '../components/Dial';
import { ArrowRight, FileIcon, LockIcon, Sunburst, UploadIcon } from '../components/Icons';
import detector, { InstallProgress, ModelStatus } from '../detector';
import { resolveMimeType } from '../mime';
import { colors, space, type } from '../theme';

// Landing screen with three compact states: home, selected, analysing. A finished analysis is
// handed to the parent (onResult) which shows the dedicated result screen; nothing is appended here.
// Only the Detector interface is used: real on-device inference, no network, no invented scores.

type Phase = 'home' | 'selected' | 'loading' | 'analysing';
const MB = 1024 * 1024;

type Props = {
  onOpenDev: () => void;
  /** Opens the in-app call screen (live check of the other participant). */
  onOpenCall: () => void;
  onResult: (a: Analysis) => void;
  onError: (title: string, detail: string, file: PickedFile) => void;
  /** Bumped by the parent to clear the selected file ("Check another file"). */
  resetToken: number;
  /** Bumped by the parent to re-run the analysis of the current file ("Try again"). */
  retryToken: number;
};

export default function HomeScreen({ onOpenDev, onOpenCall, onResult, onError, resetToken, retryToken }: Props) {
  const { width, height } = useWindowDimensions();
  // ~62% of width, smaller on short screens, so the primary action stays on screen without clipping
  const dialSize = Math.round(Math.min(width * 0.62, height * 0.3, 256));

  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [statusErr, setStatusErr] = useState<string | null>(null);
  const [install, setInstall] = useState<InstallProgress | null>(null);
  const [file, setFile] = useState<PickedFile | null>(null);
  const [phase, setPhase] = useState<Phase>('home');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const inFlight = useRef(false);

  const refresh = useCallback(() => {
    detector.status().then((s) => { setStatus(s); setStatusErr(null); })
      .catch((e) => setStatusErr(String(e?.message ?? e)));
  }, []);

  useEffect(() => {
    refresh();
    const sub = detector.onInstallProgress((p) => { setInstall(p); if (p.phase === 'done' || p.phase === 'error') refresh(); });
    return () => sub.remove();
  }, [refresh]);

  useEffect(() => {
    if (!status?.installing) return;
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [status?.installing, refresh]);

  // "Check another file": the parent bumps resetToken; adjust state during render (no effect needed).
  const [seenReset, setSeenReset] = useState(resetToken);
  if (resetToken !== seenReset) { setSeenReset(resetToken); setFile(null); setPhase('home'); }

  // Elapsed-seconds ticker while native work runs: shows real waiting time (not progress) and
  // only advances if the JS thread is responsive. Derived from a start timestamp + ticking clock.
  const busy = phase === 'loading' || phase === 'analysing';
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [busy]);
  const elapsed = busy && startedAt ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0;

  const installed = !!status?.installed;
  const installing = !!status?.installing;

  const onInstall = () => {
    setInstall(null);
    detector.install().then(() => setTimeout(refresh, 300)).catch((e) => setInstall({ phase: 'error', error: String(e?.message ?? e) }));
  };

  const onPick = async () => {
    if (busy) return;
    const res = await DocumentPicker.getDocumentAsync({ type: ['audio/*', 'application/ogg', 'video/mp4'], copyToCacheDirectory: true });
    if (res.canceled || !res.assets?.length) return;
    const a = res.assets[0];
    setFile({ uri: a.uri, name: a.name, size: a.size, mimeType: resolveMimeType(a.name, a.mimeType) });
    setPhase('selected');
  };

  const onAnalyse = useCallback(async () => {
    if (!file || inFlight.current) return;
    inFlight.current = true;
    let loadMs: number | null = null;
    try {
      const s = await detector.status();
      if (!s.installed) throw Object.assign(new Error('Set up the detector model first, then try again.'), { code: 'ERR_NOT_INSTALLED' });
      setStartedAt(Date.now()); setNow(Date.now());
      if (!s.loaded) {
        setPhase('loading');
        const info = await detector.load();           // measured natively (OnnxWindowSession.open)
        loadMs = info.alreadyLoaded ? null : info.loadMs;
      }
      setPhase('analysing');
      const t0 = Date.now();
      const r = await detector.analyseFile(file.uri);  // adapter default mode; no compare/dump
      onResult({ result: r, file, loadMs, wallMs: Date.now() - t0, finishedAt: Date.now() });
      setPhase('selected');
    } catch (e: any) {
      const { title, detail } = readable(e);
      setPhase('selected');
      onError(title, detail, file);
    } finally {
      inFlight.current = false; refresh();
    }
  }, [file, onResult, onError, refresh]);

  const retrySeen = useRef(retryToken);
  useEffect(() => {
    if (retryToken !== retrySeen.current) { retrySeen.current = retryToken; onAnalyse(); }
  }, [retryToken, onAnalyse]);

  const dial = phase === 'home' ? { title: 'Select an audio clip', subtitle: 'to begin' }
    : phase === 'selected' ? { title: 'Clip ready', subtitle: 'tap Analyse on device' }
    : phase === 'loading' ? { title: 'Loading model', subtitle: 'first use only' }
    : { title: 'Analysing', subtitle: 'on this device' };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <View style={styles.brand}>
          <Sunburst size={36} />
          <Text style={styles.brandName} maxFontSizeMultiplier={1.3}>Voice Check</Text>
        </View>
        <Pressable testID="home-dev" onPress={onOpenDev} disabled={busy} style={({ pressed }) => [styles.devBtn, pressed && styles.pressed, busy && styles.disabled]}
                   accessibilityRole="button" accessibilityLabel="Open diagnostics" accessibilityState={{ disabled: busy }} hitSlop={6}>
          <Text style={styles.devCode}>{'</>'}</Text>
          <Text style={styles.devText}>Dev</Text>
        </Pressable>
      </View>
      <View style={styles.rule} />

      <Text style={[type.eyebrow, styles.eyebrow]} maxFontSizeMultiplier={1.3}>Audio authenticity</Text>
      <Text style={[type.headline, styles.headline]} maxFontSizeMultiplier={1.3}>A second check.{'\n'}For every voice.</Text>
      <Text style={[type.lede, styles.lede]} maxFontSizeMultiplier={1.3}>Check a voice note for signs of AI-generated speech.</Text>

      <View style={styles.dialWrap}>
        <Dial size={dialSize} active={busy} title={dial.title} subtitle={dial.subtitle} />
      </View>
      <View style={styles.privacy}>
        <LockIcon />
        <Text style={[type.eyebrow, styles.privacyText]} maxFontSizeMultiplier={1.3}>On-device analysis</Text>
      </View>

      <ModelLine status={status} statusErr={statusErr} install={install} installing={installing} busy={busy}
                 onInstall={onInstall} onCancel={() => detector.cancelInstall()} onRetryStatus={refresh} />

      <View style={styles.bottom}>
        {phase === 'home' && (
          <>
            <View style={styles.rule} />
            <View style={styles.infoRow}>
              <FileIcon />
              <Text style={[type.body, styles.infoText]} maxFontSizeMultiplier={1.3}>Voice notes & recordings</Text>
              <View style={styles.infoDivider} />
              <Text style={[type.body, styles.infoMeta]} maxFontSizeMultiplier={1.3}>2–10 sec</Text>
            </View>
            <View style={styles.rule} />
            <PrimaryButton testID="home-select" label="Select audio" onPress={onPick} left={<UploadIcon />} right={<ArrowRight />} spread />
            <View style={styles.callRow}>
              <SecondaryButton testID="home-call" label="Call a friend · live check" onPress={onOpenCall} />
            </View>
          </>
        )}

        {phase === 'selected' && file && (
          <>
            <View style={styles.rule} />
            <View style={styles.fileRow} testID="home-file-row">
              <FileIcon />
              <Text style={[type.body, styles.fileName]} numberOfLines={1} ellipsizeMode="middle" testID="home-file-name">{file.name}</Text>
              <Text style={[type.small, styles.fileMeta]} maxFontSizeMultiplier={1.3}>{fileMeta(file)}</Text>
            </View>
            <View style={styles.rule} />
            <View style={styles.actions}>
              <SecondaryButton testID="home-change" label="Change file" onPress={onPick} />
              <PrimaryButton testID="home-analyse" label="Analyse on device" onPress={onAnalyse} disabled={!installed} compact />
            </View>
            {!installed && <Text style={[type.small, styles.hint]}>Set up the model above to analyse this clip.</Text>}
          </>
        )}

        {busy && file && (
          <>
            <View style={styles.rule} />
            <View style={styles.busyRow}>
              <ActivityIndicator color={colors.accent} testID="home-busy-spinner" />
              <View style={{ flex: 1 }}>
                <Text style={[type.body, styles.busyTitle]} testID="home-busy-text">
                  {phase === 'loading' ? 'Loading the detector model' : 'Analysing on this device'}
                </Text>
                <Text style={type.small} testID="home-elapsed">
                  {phase === 'loading' ? 'First use only · ' : 'Runs in the background · '}{elapsed} s elapsed
                </Text>
              </View>
            </View>
            <View style={styles.rule} />
          </>
        )}

        <Text style={[type.small, styles.privacyNote]} maxFontSizeMultiplier={1.3}>
          Your audio stays on this device. Works offline after model setup.
        </Text>
      </View>
    </ScrollView>
  );
}

function fileMeta(f: PickedFile) {
  if (f.size == null) return '';
  return f.size >= MB ? `${(f.size / MB).toFixed(1)} MB` : `${Math.max(1, Math.round(f.size / 1024))} KB`;
}

function readable(e: any): { title: string; detail: string } {
  const code: string = e?.code ?? '';
  const msg: string = e?.message ?? String(e);
  if (code === 'ERR_NOT_INSTALLED') return { title: 'Model not set up', detail: msg };
  if (code === 'ERR_OUT_OF_MEMORY' || /OutOfMemory/i.test(msg)) return { title: 'Not enough memory', detail: 'This device could not hold the 1.26 GB model in memory. Close other apps and try again.' };
  if (/not installed/i.test(msg)) return { title: 'Model not set up', detail: 'Download the detector model first, then try again.' };
  if (/sha256|checksum/i.test(msg)) return { title: 'Model file failed verification', detail: msg };
  return { title: 'Analysis failed', detail: msg.replace(/^[A-Za-z]+Exception:\s*/, '') };
}

function ModelLine(p: { status: ModelStatus | null; statusErr: string | null; install: InstallProgress | null; installing: boolean;
                        busy: boolean; onInstall: () => void; onCancel: () => void; onRetryStatus: () => void }) {
  const { status, statusErr, install, installing } = p;
  if (statusErr) {
    return (
      <View style={[styles.card, styles.errorCard]} testID="home-model-status">
        <Text style={[type.body, styles.errorTitle]}>Could not read model status</Text>
        <Text style={[type.small, styles.errorDetail]}>{statusErr}</Text>
        <SecondaryButton label="Retry" onPress={p.onRetryStatus} />
      </View>
    );
  }
  if (!status) return <Text style={[type.small, styles.modelLine]} testID="home-model-status">Checking detector model…</Text>;
  if (status.installed) {
    return <Text style={[type.small, styles.modelLine]} testID="home-model-status">{status.loaded ? 'Detector model ready' : 'Detector model installed · loads on first check'}</Text>;
  }
  const failed = install?.phase === 'error';
  const done = install && install.phase !== 'error' && install.phase !== 'done' ? install.doneBytes : status.partialBytes;
  const total = status.expectedBytes;
  const verifying = install?.phase === 'verifying' || (installing && status.partialBytes >= total);
  return (
    <View style={styles.card} testID="home-model-status">
      <Text style={[type.body, styles.cardTitle]}>One-time model setup</Text>
      <Text style={type.small}>
        Downloads the detector model (1.26 GB) into this app’s private storage and verifies it. Needs internet once.
        {status.freeBytes != null ? ` Free space: ${(status.freeBytes / 1e9).toFixed(1)} GB.` : ''}
      </Text>
      {installing ? (
        <>
          <View style={styles.progressTrack}><View style={[styles.progressFill, { width: `${Math.min(100, (100 * done) / total)}%` }]} /></View>
          <Text style={type.small} testID="home-install-progress">{verifying ? 'Verifying checksum' : 'Downloading'} · {(done / MB).toFixed(0)} / {(total / MB).toFixed(0)} MB</Text>
          <SecondaryButton label="Cancel" onPress={p.onCancel} />
        </>
      ) : (
        <>
          {failed && <Text style={[type.small, styles.errorDetail]}>Setup failed: {(install as any).error}</Text>}
          <PrimaryButton testID="home-install" label={status.partialBytes > 0 ? 'Resume download' : failed ? 'Retry setup' : 'Download model (1.26 GB)'}
                         onPress={p.onInstall} disabled={p.busy} compact />
        </>
      )}
    </View>
  );
}

export function PrimaryButton({ label, onPress, disabled, left, right, compact, spread, testID }:
  { label: string; onPress: () => void; disabled?: boolean; left?: React.ReactNode; right?: React.ReactNode; compact?: boolean; spread?: boolean; testID?: string }) {
  return (
    <Pressable testID={testID} onPress={onPress} disabled={disabled} accessibilityRole="button" accessibilityLabel={label}
               accessibilityState={{ disabled: !!disabled }}
               style={({ pressed }) => [styles.primary, spread && styles.primarySpread, compact && styles.primaryCompact, disabled && styles.disabled, pressed && styles.pressed]}>
      {left ? <View style={styles.primaryEdge}>{left}</View> : null}
      <Text style={[styles.primaryText, compact && styles.primaryTextCompact]} maxFontSizeMultiplier={1.3} numberOfLines={1}>{label}</Text>
      {right ? <View style={styles.primaryEdge}>{right}</View> : null}
    </Pressable>
  );
}

export function SecondaryButton({ label, onPress, testID }: { label: string; onPress: () => void; testID?: string }) {
  return (
    <Pressable testID={testID} onPress={onPress} accessibilityRole="button" accessibilityLabel={label}
               style={({ pressed }) => [styles.secondary, pressed && styles.pressed]}>
      <Text style={styles.secondaryText} maxFontSizeMultiplier={1.3} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { flexGrow: 1, paddingHorizontal: space.lg, paddingBottom: space.md },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: space.sm },
  brand: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  brandName: { fontSize: 22, fontWeight: '700', color: colors.ink, letterSpacing: -0.3 },
  devBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, borderWidth: 1, borderColor: colors.ink, borderRadius: 999, paddingHorizontal: 14, paddingVertical: 8 },
  devCode: { fontSize: 13, color: colors.ink, fontWeight: '600', letterSpacing: 1 },
  devText: { fontSize: 14, color: colors.ink, fontWeight: '500' },
  rule: { height: 1, backgroundColor: colors.hairline },
  eyebrow: { textAlign: 'center', marginTop: space.md },
  headline: { textAlign: 'center', marginTop: 4, fontSize: 32, lineHeight: 38 },
  lede: { textAlign: 'center', marginTop: 6, marginBottom: space.md, fontSize: 16, lineHeight: 22, paddingHorizontal: space.md },
  dialWrap: { alignItems: 'center' },
  privacy: { alignItems: 'center', gap: 4, marginTop: space.md, marginBottom: 6 },
  privacyText: { fontSize: 11 },
  modelLine: { textAlign: 'center', marginBottom: 4 },
  bottom: { marginTop: 'auto', paddingTop: space.sm },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12 },
  infoText: { flex: 1, fontSize: 15 },
  infoDivider: { width: 1, height: 20, backgroundColor: colors.hairline, marginRight: 8 },
  infoMeta: { color: colors.inkSoft, fontSize: 15 },
  fileRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12 },
  fileName: { flex: 1, fontWeight: '600' },
  fileMeta: { color: colors.inkSoft },
  actions: { flexDirection: 'row', gap: 10, alignItems: 'stretch', marginTop: space.sm },
  hint: { marginTop: space.sm, textAlign: 'center' },
  busyRow: { flexDirection: 'row', alignItems: 'center', gap: 14, paddingVertical: 14 },
  busyTitle: { fontWeight: '600' },
  primary: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 12, backgroundColor: colors.accent, borderRadius: 14, paddingVertical: 17, paddingHorizontal: 18, marginTop: space.sm },
  primarySpread: { justifyContent: 'space-between' },
  primaryCompact: { paddingVertical: 14, paddingHorizontal: 12, marginTop: 0, flex: 1.6 },
  primaryEdge: { width: 26, alignItems: 'center' },
  primaryText: { color: colors.accentInk, fontSize: 18, fontWeight: '700', flexShrink: 1, textAlign: 'center' },
  primaryTextCompact: { fontSize: 16 },
  secondary: { borderWidth: 1, borderColor: colors.ink, borderRadius: 14, paddingVertical: 14, paddingHorizontal: 14, alignItems: 'center', justifyContent: 'center', flex: 1 },
  secondaryText: { color: colors.ink, fontSize: 15, fontWeight: '600' },
  disabled: { opacity: 0.4 },
  pressed: { opacity: 0.8 },
  card: { backgroundColor: colors.surfaceStrong, borderRadius: 14, borderWidth: 1, borderColor: colors.hairline, padding: space.md, marginTop: space.sm },
  cardTitle: { fontWeight: '600', marginBottom: 4 },
  progressTrack: { height: 6, borderRadius: 3, backgroundColor: colors.hairline, marginTop: space.md, overflow: 'hidden' },
  progressFill: { height: 6, backgroundColor: colors.accent },
  errorCard: { backgroundColor: colors.dangerBg, borderColor: '#F0C2BA' },
  errorTitle: { fontWeight: '700', color: colors.danger },
  errorDetail: { color: '#7A271A', marginTop: 4 },
  privacyNote: { textAlign: 'center', marginTop: space.sm, fontSize: 13, lineHeight: 19 },
  callRow: { flexDirection: 'row', marginTop: space.sm },
});
