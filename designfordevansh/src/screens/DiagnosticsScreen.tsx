import * as DocumentPicker from 'expo-document-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

import type { Analysis } from '../analysis';
import detector, {
  InstallProgress, LoadInfo, LocalResult, ModelStatus, ReferenceRun,
} from '../detector';

// Diagnostics screen ("Dev"): model install/load, Colab reference checks, per-mode analysis,
// timings and memory. Unchanged behaviour from the verified offline demo; now reached from the
// landing screen via the Dev button. The server-upload version is kept in reference/server-upload-v1/.

type Picked = { uri: string; name: string; size?: number };

type DiagProps = { onBack: () => void; active: boolean; lastAnalysis: Analysis | null };

export default function DiagnosticsScreen(p: DiagProps) {
  return <OfflineDetector {...p} />;
}

function OfflineDetector({ onBack, active, lastAnalysis }: DiagProps) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [install, setInstall] = useState<InstallProgress | null>(null);
  const [load, setLoad] = useState<LoadInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // label of the running native task
  const [error, setError] = useState<string | null>(null);
  const [ref, setRef] = useState<ReferenceRun | null>(null);
  const [mode, setMode] = useState<string | null>(null); // null -> adapter default
  const [compare, setCompare] = useState(true);
  const [file, setFile] = useState<Picked | null>(null);
  const [result, setResult] = useState<LocalResult | null>(null);
  const inFlight = useRef(false);

  const refresh = useCallback(() => {
    detector.status().then(setStatus).catch((e) => setError(String(e?.message ?? e)));
  }, []);

  useEffect(() => {
    refresh();
    const sub = detector.onInstallProgress((p: InstallProgress) => {
      setInstall(p);
      if (p.phase === 'done' || p.phase === 'error') refresh();
      if (p.phase === 'error') setError(`Model install failed: ${p.error}`);
    });
    return () => sub.remove();
  }, [refresh]);

  // The native engine (model + session) is shared with the landing screen; re-read its state on show.
  useEffect(() => { if (active) refresh(); }, [active, refresh]);

  // Serialises native tasks from the UI side; the native engine also runs them on one thread.
  const run = useCallback(async <T,>(label: string, fn: () => Promise<T>): Promise<T | undefined> => {
    if (inFlight.current) return undefined;
    inFlight.current = true;
    setBusy(label);
    setError(null);
    try {
      return await fn();
    } catch (e: any) {
      setError(`${e?.code ? `${e.code}: ` : ''}${e?.message ?? String(e)}`);
      return undefined;
    } finally {
      inFlight.current = false;
      setBusy(null);
      refresh();
    }
  }, [refresh]);

  const onInstall = () => {
    setError(null); setInstall(null);
    detector.install().then(() => setTimeout(refresh, 300)).catch((e) => setError(String(e?.message ?? e)));
  };
  const onLoad = () => run('Loading model', async () => setLoad(await detector.load()));
  const onRef = () => run('Running reference tensors', async () => setRef(await detector.runReferenceTensors(1000)));
  const onPick = async () => {
    if (inFlight.current) return;
    const res = await DocumentPicker.getDocumentAsync({ type: ['audio/*', 'application/ogg', 'video/mp4'], copyToCacheDirectory: true });
    if (res.canceled || !res.assets?.length) return;
    const a = res.assets[0];
    setFile({ uri: a.uri, name: a.name, size: a.size });
    setResult(null);
  };
  const onAnalyse = () => file && activeMode && run('Analysing on device', async () => {
    setResult(null);
    setResult(await detector.analyseFile(file.uri, { mode: activeMode, compare, dump: compare }));
  });

  // Poll native status while installing: progress comes from the .part file size, so the
  // progress display does not depend on event delivery alone.
  useEffect(() => {
    if (!status?.installing) return;
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, [status?.installing, refresh]);

  const installed = !!status?.installed;
  const modes = status?.adapter?.preprocessingModes ?? [];
  const activeMode = mode ?? status?.adapter?.defaultMode;
  const installing = !!status?.installing || install?.phase === 'downloading' || install?.phase === 'verifying';

  return (
    <ScrollView contentContainerStyle={styles.scroll}>
      <View style={styles.topRow}>
        <Pressable testID="dev-back" onPress={onBack} hitSlop={12} accessibilityRole="button" accessibilityLabel="Back">
          <Text style={styles.back}>‹ Back</Text>
        </Pressable>
        <Text style={styles.title}>Diagnostics</Text>
      </View>
      {lastAnalysis && <LastAnalysis a={lastAnalysis} attribution={status?.adapter?.attribution} />}
      <Text style={styles.subtitle}>Offline, on-device · third-party Dhwani model (MIT) · scores are uncalibrated and unverified</Text>

      <Section title="Model">
        <Text style={styles.muted} testID="model-status">
          {status == null ? 'Checking…' : installed
            ? `Installed · ${(status.modelBytes / 1e9).toFixed(2)} GB · revision ${status.revision.slice(0, 7)}${status.loaded ? ' · loaded' : ''}`
            : `Not installed${status.partialBytes ? ` · ${(status.partialBytes / 1e6).toFixed(0)} MB downloaded so far` : ''}`}
        </Text>
        {status?.freeBytes != null && <Text style={styles.muted}>Free device storage: {(status.freeBytes / 1e9).toFixed(2)} GB</Text>}
        {!installed && (
          <>
            <Text style={styles.note}>One-time download of the pinned 1.26 GB checkpoint into app-private storage, verified by SHA-256. Needs internet once; analysis afterwards is offline.</Text>
            <Button testID="btn-install" label={installing ? 'Installing…' : status?.partialBytes ? 'Resume download' : 'Download model (1.26 GB)'}
                    onPress={onInstall} disabled={installing || !!busy} />
            {installing && <Button testID="btn-cancel-install" label="Cancel download" kind="secondary" onPress={() => detector.cancelInstall()} />}
          </>
        )}
        {status?.installing && !(install && (install.phase === 'downloading' || install.phase === 'verifying')) && (
          <View style={styles.progressWrap} testID="install-progress">
            <View style={[styles.progressBar, { width: `${(100 * status.partialBytes) / status.expectedBytes}%` }]} />
            <Text style={styles.muted}>
              {status.partialBytes < status.expectedBytes ? 'Downloading' : 'Verifying SHA-256'} · {(status.partialBytes / 1e6).toFixed(0)} / {(status.expectedBytes / 1e6).toFixed(0)} MB
            </Text>
          </View>
        )}
        {install && install.phase !== 'error' && install.phase !== 'done' && (
          <View style={styles.progressWrap} testID="install-progress">
            <View style={[styles.progressBar, { width: `${(100 * install.doneBytes) / install.totalBytes}%` }]} />
            <Text style={styles.muted}>
              {install.phase === 'downloading' ? 'Downloading' : 'Verifying SHA-256'} · {(install.doneBytes / 1e6).toFixed(0)} / {(install.totalBytes / 1e6).toFixed(0)} MB
            </Text>
          </View>
        )}
        {installed && (
          <Button testID="btn-load" label={status?.loaded ? 'Model loaded' : 'Load model'} onPress={onLoad}
                  disabled={!!busy || !!status?.loaded} kind="secondary" />
        )}
        {(load ?? status?.lastLoad) && <LoadSummary info={(load ?? status?.lastLoad)!} />}
        {status && <Text style={styles.tiny}>{status.attribution}</Text>}
      </Section>

      {installed && status?.referenceAvailable && (
        <Section title="Reference check (Colab export)">
          <Text style={styles.note}>Runs the exact preprocessed windows exported from Colab through this device’s runtime and compares logits.</Text>
          <Button testID="btn-ref" label="Run reference tensors" onPress={onRef} disabled={!!busy} kind="secondary" />
          {ref && <RefSummary r={ref} />}
        </Section>
      )}

      <Section title="Analyse audio (offline)">
        <Text style={styles.label}>Preprocessing (default {status?.adapter?.defaultMode ?? '–'} is provisional)</Text>
        <View style={styles.chips}>
          {modes.map((m) => (
            <Pressable key={m} testID={`mode-${m}`} onPress={() => setMode(m)} disabled={!!busy}
                       style={[styles.chip, activeMode === m && styles.chipOn]}>
              <Text style={[styles.chipText, activeMode === m && styles.chipTextOn]}>{m}</Text>
            </Pressable>
          ))}
        </View>
        <View style={styles.switchRow}>
          <Text style={styles.label}>Compare with Colab reference (if available)</Text>
          <Switch testID="switch-compare" value={compare} onValueChange={setCompare} disabled={!!busy} />
        </View>
        <Button testID="btn-pick" label={file ? 'Choose a different file' : 'Choose audio file'} onPress={onPick} disabled={!!busy} kind="secondary" />
        {file && <Text testID="picked-file" style={styles.fileLine}>{file.name}{file.size != null ? ` · ${(file.size / 1024).toFixed(0)} KB` : ''}</Text>}
        <Button testID="btn-analyse" label="Analyse on this device" onPress={onAnalyse} disabled={!file || !installed || !!busy} />
        {!installed && <Text style={styles.muted}>Install the model first.</Text>}
      </Section>

      {busy && (
        <View style={styles.busy} testID="busy">
          <ActivityIndicator />
          <Text style={styles.muted}>  {busy}… (the UI stays usable; this can take a while on a phone CPU)</Text>
        </View>
      )}
      {error && <View style={styles.errorBox} testID="error"><Text style={styles.errorText}>{error}</Text></View>}
      {result && <ResultCard r={result} />}
    </ScrollView>
  );
}

const f1 = (v?: number | null, d = 1) => (v == null ? '–' : v.toFixed(d));

// Technical detail of the most recent check from the landing screen: raw values from the detector.
function LastAnalysis({ a, attribution }: { a: Analysis; attribution?: string }) {
  const r = a.result; const t = r.timing_ms;
  const ms = (v?: number | null) => (v == null ? '–' : `${Math.round(v)} ms`);
  return (
    <Section title="Last check (from Voice Check)">
      <Row label="File" value={a.file.name} testID="dev-last-file" />
      {r.source?.sha256 && <Row label="SHA-256" value={r.source.sha256.slice(0, 16) + '…'} />}
      <Row label="Status" value={r.status} />
      {r.status === 'ok' ? (
        <>
          <Row label="Prediction" value={r.prediction ?? '–'} />
          <Row label="Synthetic score (uncalibrated)" value={r.synthetic_score?.toFixed(6) ?? '–'} testID="dev-last-score" />
          <Row label="Highest window score" value={r.synthetic_score_max?.toFixed(6) ?? '–'} />
          <Text style={styles.note}>Mean softmax[synthetic] over 4 s windows; labelled at the untuned {r.decision_threshold} threshold. Not a probability, confidence or accuracy figure.</Text>
          <Row label="Preprocessing" value={`${r.preprocessing} (provisional)`} />
          <Row label="Duration" value={`${r.duration_s?.toFixed(3)} s${r.truncated ? ' (first 10 s)' : ''}`} />
          <Row label="Model load (this check)" value={a.loadMs == null ? 'already loaded' : `${(a.loadMs / 1000).toFixed(2)} s`} testID="dev-last-load" />
          <Row label="Decode / resample / prep" value={`${ms(t.decode)} / ${ms(t.resample)} / ${ms(t.preprocess)}`} />
          <Row label="Inference (all windows)" value={`${((t.model_total ?? 0) / 1000).toFixed(2)} s`} testID="dev-last-inference" />
          <Row label="Pipeline total (native)" value={`${(t.total / 1000).toFixed(2)} s`} />
          <Row label="Wall time (JS call)" value={`${(a.wallMs / 1000).toFixed(2)} s`} />
          {r.windows.map((w, i) => (
            <Text key={i} style={styles.tiny} testID={i === 0 ? 'dev-last-windows' : undefined}>
              window {i} @ {w.start_s.toFixed(2)} s: score {w.synthetic_score.toFixed(6)} · logits [{w.logits.map((v) => v.toFixed(4)).join(', ')}] · {Math.round(w.ms)} ms
            </Text>
          ))}
          {r.memory && <Row label="Peak RSS" value={`${r.memory.peak_rss_mb?.toFixed(0) ?? '–'} MB`} />}
        </>
      ) : (
        <Row label="Reject reason" value={r.reject_reason ?? '–'} testID="dev-last-reject" />
      )}
      {attribution && <Text style={styles.tiny}>{attribution}</Text>}
    </Section>
  );
}

function LoadSummary({ info }: { info: LoadInfo }) {
  return (
    <View testID="load-summary">
      <Row label="Cold model load" value={`${f1(info.loadMs / 1000, 2)} s`} />
      <Row label="Peak RSS after load" value={`${f1(info.memoryAfter.peak_rss_mb, 0)} MB`} />
      <Row label="RSS after load" value={`${f1(info.memoryAfter.rss_mb, 0)} MB`} />
      <Row label="Threads" value={String(info.settings.intraOpThreads ?? '–')} />
    </View>
  );
}

function RefSummary({ r }: { r: ReferenceRun }) {
  return (
    <View testID="ref-summary">
      <Row label="Windows run" value={String(r.windows_run)} />
      <Row label="Max |logit diff| vs Colab" value={r.max_abs_logit_diff.toExponential(2)} testID="ref-maxdiff" />
      <Row label="Mean ms / window" value={f1(r.rows.reduce((a, x) => a + x.ms, 0) / Math.max(1, r.rows.length), 0)} />
      {r.rows.slice(0, 6).map((x) => (
        <Text key={`${x.clip}-${x.mode}-${x.window}`} style={styles.tiny}>
          {x.clip}/{x.mode}/w{x.window}: device {x.device_score.toFixed(6)} · colab {x.colab_score.toFixed(6)} · Δlogit {x.max_abs_logit_diff.toExponential(1)}
        </Text>
      ))}
    </View>
  );
}

function ResultCard({ r }: { r: LocalResult }) {
  if (r.status === 'rejected') {
    return (
      <Section title="Audio rejected">
        <Text testID="result-rejected" style={styles.errorText}>{r.reject_reason}</Text>
      </Section>
    );
  }
  const synthetic = r.prediction === 'synthetic';
  const c = r.comparison;
  return (
    <Section title="Result (computed on this device)">
      <Text testID="result-prediction" style={[styles.prediction, synthetic ? styles.predSynthetic : styles.predGenuine]}>
        {synthetic ? 'Likely synthetic' : 'Likely genuine'}
      </Text>
      <Text style={styles.muted}>Model output at an untuned 0.5 threshold. A “genuine” result is not proof that a caller is safe.</Text>
      <Row label="Synthetic score (uncalibrated)" value={r.synthetic_score?.toFixed(6) ?? '–'} testID="result-score" />
      <Text style={styles.note}>Not a probability, confidence or accuracy figure.</Text>
      <Row label="Highest window score" value={r.synthetic_score_max?.toFixed(6) ?? '–'} />
      <Row label="Duration" value={`${f1(r.duration_s, 3)} s${r.truncated ? ' (first 10 s analysed)' : ''}`} />
      <Row label="Preprocessing" value={r.preprocessing} />
      <Row label="Windows" value={String(r.windows.length)} />
      <Row label="Decode / resample / prep" value={`${f1(r.timing_ms.decode, 0)} / ${f1(r.timing_ms.resample, 0)} / ${f1(r.timing_ms.preprocess, 0)} ms`} />
      {r.timing_ms.model_load != null && <Row label="Model load (this run)" value={`${f1(r.timing_ms.model_load! / 1000, 2)} s`} />}
      <Row label="Inference total" value={`${f1((r.timing_ms.model_total ?? 0) / 1000, 2)} s`} testID="result-model-time" />
      <Row label="Per window (mean)" value={`${f1(r.timing_ms.per_window_mean, 0)} ms`} />
      <Row label="Total processing" value={`${f1(r.timing_ms.total / 1000, 2)} s`} testID="result-time" />
      <Row label="Peak RSS" value={`${f1(r.memory?.peak_rss_mb, 0)} MB`} />
      {c && (c.available ? (
        <View testID="comparison">
          <Text style={styles.label}>Compared with Colab reference ({c.clip}, {c.mode})</Text>
          <Row label="Native rate device / colab" value={`${c.native_rate.device} / ${c.native_rate.colab} Hz`} />
          <Row label="Decoded len device / colab" value={`${c.native_mono.device_len} / ${c.native_mono.colab_len}`} />
          <Row label="Decoded max |Δ| (lag 0)" value={c.native_mono.max_abs_diff.toExponential(2)} />
          <Row label={`Decoded best lag ${c.native_mono_best_lag.lag}`} value={c.native_mono_best_lag.max_abs_diff.toExponential(2)} />
          <Row label="16 kHz max |Δ| · SNR" value={`${c.x16.max_abs_diff.toExponential(2)} · ${f1(c.x16.snr_db, 1)} dB`} />
          {c.windows.map((w, i) => (
            <Row key={i} label={`Window ${i} max |Δ|`} value={w.max_abs_diff.toExponential(2)} />
          ))}
          <Row label="Clip score device / colab" value={`${c.clip_score.device.toFixed(6)} / ${c.clip_score.colab.toFixed(6)}`} testID="cmp-score" />
        </View>
      ) : <Text style={styles.tiny}>Reference comparison: {c.reason}</Text>)}
    </Section>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{title}</Text>
      {children}
    </View>
  );
}

function Row({ label, value, testID }: { label: string; value: string; testID?: string }) {
  return (
    <View style={styles.kv}>
      <Text style={styles.kvLabel}>{label}</Text>
      <Text style={styles.kvValue} testID={testID}>{value}</Text>
    </View>
  );
}

function Button({ label, onPress, disabled, kind = 'primary', testID }:
  { label: string; onPress: () => void; disabled?: boolean; kind?: 'primary' | 'secondary'; testID?: string }) {
  return (
    <Pressable
      testID={testID} accessibilityRole="button" accessibilityLabel={label} accessibilityState={{ disabled: !!disabled }}
      onPress={onPress} disabled={disabled}
      style={({ pressed }) => [styles.btn, kind === 'secondary' && styles.btnSecondary, disabled && styles.btnDisabled, pressed && styles.btnPressed]}
    >
      <Text style={[styles.btnText, kind === 'secondary' && styles.btnTextSecondary]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4F5F7' },
  scroll: { padding: 16, paddingBottom: 48 },
  topRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  back: { fontSize: 17, color: '#2457C5', fontWeight: '600' },
  title: { fontSize: 24, fontWeight: '700', color: '#111' },
  subtitle: { fontSize: 13, color: '#5B6170', marginTop: 4, marginBottom: 16 },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 16, marginBottom: 14, borderWidth: 1, borderColor: '#E3E5EA' },
  cardTitle: { fontSize: 16, fontWeight: '600', color: '#111', marginBottom: 8 },
  label: { fontSize: 13, color: '#3A3F4B', marginTop: 8, marginBottom: 4, flexShrink: 1 },
  muted: { fontSize: 13, color: '#5B6170', marginTop: 6, flexShrink: 1 },
  note: { fontSize: 12, color: '#5B6170', marginTop: 6 },
  tiny: { fontSize: 11, color: '#6B7180', marginTop: 6 },
  fileLine: { marginTop: 10, fontSize: 14, color: '#111' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { borderWidth: 1, borderColor: '#CDD1DA', borderRadius: 16, paddingHorizontal: 12, paddingVertical: 6 },
  chipOn: { backgroundColor: '#2457C5', borderColor: '#2457C5' },
  chipText: { color: '#3A3F4B', fontSize: 13 },
  chipTextOn: { color: '#fff', fontWeight: '600' },
  switchRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 6 },
  progressWrap: { marginTop: 10 },
  progressBar: { height: 6, backgroundColor: '#2457C5', borderRadius: 3 },
  busy: { flexDirection: 'row', alignItems: 'center', marginBottom: 12, paddingHorizontal: 4 },
  btn: { backgroundColor: '#2457C5', borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 12 },
  btnSecondary: { backgroundColor: '#fff', borderWidth: 1, borderColor: '#2457C5' },
  btnDisabled: { opacity: 0.4 },
  btnPressed: { opacity: 0.75 },
  btnText: { color: '#fff', fontWeight: '600', fontSize: 15 },
  btnTextSecondary: { color: '#2457C5' },
  prediction: { fontSize: 22, fontWeight: '700', marginBottom: 2 },
  predSynthetic: { color: '#B42318' },
  predGenuine: { color: '#1E7B34' },
  kv: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#E3E5EA' },
  kvLabel: { color: '#3A3F4B', fontSize: 14, flexShrink: 1, paddingRight: 8 },
  kvValue: { color: '#111', fontSize: 14, fontWeight: '600' },
  errorBox: { backgroundColor: '#FEF3F2', borderColor: '#FDA29B', borderWidth: 1, borderRadius: 8, padding: 12, marginBottom: 12 },
  errorText: { color: '#B42318' },
});
