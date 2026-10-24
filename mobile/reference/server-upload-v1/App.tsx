import { StatusBar } from 'expo-status-bar';
import * as DocumentPicker from 'expo-document-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import { ApiError, checkHealth, detect, DetectResult, Health, MAX_UPLOAD_BYTES, PickedFile } from './src/api';
import { ensureExtension, resolveMimeType } from './src/mime';
import { loadSettings, saveSettings } from './src/storage';

type Phase = 'idle' | 'analysing' | 'done' | 'error';

export default function App() {
  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        <StatusBar style="dark" />
        <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <DetectorScreen />
        </KeyboardAvoidingView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

function DetectorScreen() {
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [settingsMsg, setSettingsMsg] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(true);

  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState<ApiError | null>(null);
  const [checking, setChecking] = useState(false);

  const [file, setFile] = useState<PickedFile | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [result, setResult] = useState<DetectResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const inFlight = useRef(false); // guards against double taps before React re-renders
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  useEffect(() => {
    loadSettings()
      .then((s) => {
        setBaseUrl(s.baseUrl);
        setApiKey(s.apiKey);
        if (s.baseUrl && s.apiKey) setShowSettings(false);
      })
      .catch(() => setSettingsMsg('Could not read saved settings.'));
  }, []);

  const onSave = useCallback(async () => {
    try {
      await saveSettings({ baseUrl, apiKey });
      setSettingsMsg('Saved securely on this device.');
      setHealth(null);
      setHealthErr(null);
    } catch {
      setSettingsMsg('Saving failed.');
    }
  }, [baseUrl, apiKey]);

  const onCheck = useCallback(async () => {
    setChecking(true);
    setHealthErr(null);
    setHealth(null);
    try {
      setHealth(await checkHealth(baseUrl));
    } catch (e) {
      setHealthErr(toApiError(e));
    } finally {
      setChecking(false);
    }
  }, [baseUrl]);

  const onPick = useCallback(async () => {
    if (inFlight.current) return;
    const res = await DocumentPicker.getDocumentAsync({
      type: ['audio/*', 'application/ogg', 'video/mp4', 'application/octet-stream'],
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (res.canceled || !res.assets?.length) return;
    const a = res.assets[0];
    const mimeType = resolveMimeType(a.name, a.mimeType);
    setFile({ uri: a.uri, name: ensureExtension(a.name, mimeType), mimeType, size: a.size });
    setResult(null);
    setError(null);
    setPhase('idle');
  }, []);

  const onAnalyse = useCallback(async () => {
    if (!file || inFlight.current) return;
    inFlight.current = true;
    setPhase('analysing');
    setResult(null);
    setError(null);
    const t0 = Date.now();
    try {
      const r = await detect(baseUrl, apiKey, file, (xhr) => { xhrRef.current = xhr; });
      setResult(r);
      setRoundTripMs(Date.now() - t0);
      setPhase('done');
    } catch (e) {
      setError(toApiError(e));
      setPhase('error');
    } finally {
      xhrRef.current = null;
      inFlight.current = false;
    }
  }, [baseUrl, apiKey, file]);

  const onCancel = useCallback(() => xhrRef.current?.abort(), []);

  const onReset = useCallback(() => {
    setFile(null);
    setResult(null);
    setError(null);
    setPhase('idle');
  }, []);

  const busy = phase === 'analysing';

  return (
    <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
      <Text style={styles.title}>Voice-clone check</Text>
      <Text style={styles.subtitle}>Demo · third-party Dhwani baseline model · results are not verified</Text>

      {/* Settings */}
      <Section title="Backend" right={
        <Pressable onPress={() => setShowSettings((v) => !v)} hitSlop={8} testID="toggle-settings">
          <Text style={styles.link}>{showSettings ? 'Hide' : 'Edit'}</Text>
        </Pressable>
      }>
        {showSettings ? (
          <>
            <Text style={styles.label}>Backend URL</Text>
            <TextInput
              testID="input-url" style={styles.input} value={baseUrl} onChangeText={setBaseUrl}
              placeholder="https://….trycloudflare.com" autoCapitalize="none" autoCorrect={false}
              keyboardType="url" editable={!busy}
            />
            <Text style={styles.label}>API key</Text>
            <TextInput
              testID="input-key" style={styles.input} value={apiKey} onChangeText={setApiKey}
              placeholder="X-API-Key" autoCapitalize="none" autoCorrect={false} secureTextEntry editable={!busy}
            />
            <Button testID="btn-save" label="Save" onPress={onSave} disabled={busy} />
            {settingsMsg && <Text style={styles.muted}>{settingsMsg}</Text>}
          </>
        ) : (
          <Text style={styles.muted} numberOfLines={1}>{baseUrl || 'No URL set'} · key {apiKey ? 'saved' : 'missing'}</Text>
        )}
        <Button testID="btn-health" label={checking ? 'Checking…' : 'Check connection'} onPress={onCheck}
                disabled={checking || busy || !baseUrl} kind="secondary" />
        {health && (
          <Text testID="health-status" style={health.model_ready ? styles.ok : styles.warn}>
            {health.model_ready ? '● Connected · model ready' : `● Connected · model NOT ready${health.load_error ? `: ${health.load_error}` : ''}`}
            {health.busy ? ' · busy' : ''}
          </Text>
        )}
        {healthErr && <ErrorBox err={healthErr} testID="health-error" />}
      </Section>

      {/* Upload */}
      <Section title="Audio">
        <Button testID="btn-pick" label={file ? 'Choose a different file' : 'Choose audio file'} onPress={onPick}
                disabled={busy} kind="secondary" />
        {file && (
          <Text testID="picked-file" style={styles.fileLine}>
            {file.name} · {file.size != null ? `${(file.size / 1024).toFixed(0)} KB` : 'size unknown'} · {file.mimeType}
          </Text>
        )}
        {file?.size != null && file.size > MAX_UPLOAD_BYTES && (
          <Text style={styles.warn}>Larger than the 8 MB server limit.</Text>
        )}
        <Button testID="btn-analyse" label={busy ? 'Analysing…' : 'Analyse'} onPress={onAnalyse}
                disabled={!file || busy || !baseUrl || !apiKey} />
        {busy && (
          <View style={styles.row} testID="analysing">
            <ActivityIndicator />
            <Text style={styles.muted}>  Uploading and analysing. Can take up to a minute on the Colab CPU.</Text>
          </View>
        )}
        {busy && <Button testID="btn-cancel" label="Cancel" onPress={onCancel} kind="secondary" />}
      </Section>

      {phase === 'done' && result && <ResultCard r={result} roundTripMs={roundTripMs} onReset={onReset} />}
      {phase === 'error' && error && (
        <Section title="Could not analyse">
          <ErrorBox err={error} testID="detect-error" />
          <Button testID="btn-retry" label="Try again" onPress={onAnalyse} disabled={!file} />
          <Button testID="btn-another" label="Choose another file" onPress={onReset} kind="secondary" />
        </Section>
      )}
    </ScrollView>
  );
}

function ResultCard({ r, roundTripMs, onReset }: { r: DetectResult; roundTripMs: number | null; onReset: () => void }) {
  const synthetic = r.prediction === 'synthetic';
  const fmt = (ms?: number | null) => (ms == null ? '–' : `${(ms / 1000).toFixed(1)} s`);
  return (
    <Section title="Result">
      <Text testID="result-prediction" style={[styles.prediction, synthetic ? styles.predSynthetic : styles.predGenuine]}>
        {synthetic ? 'Likely synthetic' : 'Likely genuine'}
      </Text>
      <Text style={styles.muted}>Model prediction at an untuned 0.5 threshold. Not a verified verdict.</Text>
      <Row label="Synthetic score (uncalibrated)" value={r.synthetic_score?.toFixed(4) ?? '–'} testID="result-score" />
      <Text style={styles.note}>An uncalibrated model score between 0 and 1. It is not a probability, confidence or accuracy figure.</Text>
      <Row label="Highest window score" value={r.synthetic_score_max?.toFixed(4) ?? '–'} />
      <Row label="Audio duration" value={r.duration_s != null ? `${r.duration_s.toFixed(2)} s${r.truncated ? ' (first 10 s analysed)' : ''}` : '–'} />
      <Row label="Server processing time" value={fmt(r.timing_ms.total)} testID="result-time" />
      <Row label="Total incl. upload" value={fmt(roundTripMs)} />
      <Row label="Preprocessing" value={`${r.preprocessing} (provisional)`} />
      <Row label="Windows analysed" value={String(r.windows.length)} />
      {r.request_id && <Row label="Request ID" value={r.request_id} />}
      <Button testID="btn-another" label="Analyse another file" onPress={onReset} />
    </Section>
  );
}

function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e;
  return new ApiError('server', 'Unexpected error', e instanceof Error ? e.message : String(e));
}

function ErrorBox({ err, testID }: { err: ApiError; testID?: string }) {
  return (
    <View style={styles.errorBox} testID={testID}>
      <Text style={styles.errorTitle}>{err.title}{err.httpStatus ? ` (HTTP ${err.httpStatus})` : ''}</Text>
      <Text style={styles.errorText}>{err.message}</Text>
    </View>
  );
}

function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>{title}</Text>
        {right}
      </View>
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
  title: { fontSize: 24, fontWeight: '700', color: '#111' },
  subtitle: { fontSize: 13, color: '#5B6170', marginTop: 4, marginBottom: 16 },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 16, marginBottom: 14, borderWidth: 1, borderColor: '#E3E5EA' },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  cardTitle: { fontSize: 16, fontWeight: '600', color: '#111' },
  link: { color: '#2457C5', fontWeight: '600' },
  label: { fontSize: 13, color: '#3A3F4B', marginTop: 8, marginBottom: 4 },
  input: { borderWidth: 1, borderColor: '#CDD1DA', borderRadius: 8, paddingHorizontal: 12, paddingVertical: 10, fontSize: 15, color: '#111', backgroundColor: '#FAFBFC' },
  muted: { fontSize: 13, color: '#5B6170', marginTop: 8, flexShrink: 1 },
  note: { fontSize: 12, color: '#5B6170', marginBottom: 6 },
  ok: { color: '#1E7B34', marginTop: 10, fontWeight: '600' },
  warn: { color: '#A15C00', marginTop: 10, fontWeight: '600' },
  fileLine: { marginTop: 10, fontSize: 14, color: '#111' },
  row: { flexDirection: 'row', alignItems: 'center', marginTop: 12 },
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
  errorBox: { backgroundColor: '#FEF3F2', borderColor: '#FDA29B', borderWidth: 1, borderRadius: 8, padding: 12, marginTop: 10 },
  errorTitle: { color: '#B42318', fontWeight: '700', marginBottom: 2 },
  errorText: { color: '#7A271A' },
});
