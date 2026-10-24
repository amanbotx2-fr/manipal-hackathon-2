import * as SecureStore from 'expo-secure-store';
import { useCallback, useEffect, useRef, useState } from 'react';
import { PermissionsAndroid, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { VcdCallView, type LiveResult } from '../../modules/vcd-detector/src/calling';
import detector, { type ModelStatus } from '../detector';
import { colors, space, type } from '../theme';
import { PrimaryButton } from './HomeScreen';

// In-app audio call with a live voice check of the OTHER participant. The call itself is the
// vendored LiveCalling kit (WebView WebRTC); the live check runs natively on the shared detector.
// Scores are uncalibrated model outputs and are labelled as such. Nothing here scores the local mic.

const SERVER_KEY = 'vcd.callServer';
const isHttps = (s: string) => /^https:\/\/[^\s/]+/i.test(s.trim());

export default function CallScreen({ onBack }: { onBack: () => void }) {
  const [server, setServer] = useState('');
  const [invite, setInvite] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [preparing, setPreparing] = useState<string | null>(null);
  const [callUrl, setCallUrl] = useState<string | null>(null);
  const [callState, setCallState] = useState('');
  const [latest, setLatest] = useState<LiveResult | null>(null);
  const [history, setHistory] = useState<LiveResult[]>([]);
  const starting = useRef(false);

  useEffect(() => {
    SecureStore.getItemAsync(SERVER_KEY).then((v) => { if (v) setServer(v); }).catch(() => {});
    detector.status().then(setStatus).catch(() => {});
  }, []);

  const start = useCallback(async () => {
    if (starting.current) return;
    starting.current = true; setMsg(null);
    try {
      const inv = invite.trim();
      const base = server.trim().replace(/\/+$/, '') + '/';
      const url = inv ? (isHttps(inv) ? inv : null) : (isHttps(base) ? base : null);
      if (!url) { setMsg('Enter the HTTPS address of your calling server, or paste a room invitation link.'); return; }
      if (Platform.OS !== 'android') { setMsg('In-app calls are Android-only in this build.'); return; }
      const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, {
        title: 'Microphone', message: 'Needed so the other person can hear you during the call.', buttonPositive: 'Allow',
      });
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) { setMsg('Microphone permission is needed to call.'); return; }
      await SecureStore.setItemAsync(SERVER_KEY, server.trim()).catch(() => {});
      const s = await detector.status(); setStatus(s);
      if (s.installed && !s.loaded) {
        setPreparing('Loading the detector model so the live check is ready…');
        await detector.load().catch((e) => setMsg(`Live check unavailable: ${e?.message ?? e}`));
        setPreparing(null);
      }
      setLatest(null); setHistory([]); setCallState('Opening calling room');
      setCallUrl(url);
    } finally { starting.current = false; }
  }, [server, invite]);

  const end = useCallback(() => { setCallUrl(null); setCallState(''); }, []);
  const onLive = useCallback((e: { nativeEvent: LiveResult }) => {
    const r = e.nativeEvent; setLatest(r); setHistory((h) => [r, ...h].slice(0, 6));
  }, []);

  const liveAvailable = !!status?.installed;

  if (callUrl) {
    return (
      <View style={styles.screen}>
        <View style={styles.header}>
          <Pressable testID="call-end" onPress={end} hitSlop={12} accessibilityRole="button" accessibilityLabel="End call and go back">
            <Text style={styles.back}>‹ End call</Text>
          </Pressable>
          <Text style={styles.headerTitle}>In-app call</Text>
          <View style={{ width: 80 }} />
        </View>
        <Text style={[type.small, styles.stateLine]} testID="call-state" numberOfLines={2}>{callState || '…'}</Text>
        <LiveStrip latest={latest} history={history} available={liveAvailable} />
        <VcdCallView style={styles.callView} serverUrl={callUrl} liveCheck={liveAvailable}
                     onCallState={(e) => setCallState(e.nativeEvent.state)} onLiveResult={onLive} />
        <Text style={[type.fine, styles.foot]}>Experimental live check of the other participant’s audio. Verify requests through another channel.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      <View style={styles.header}>
        <Pressable testID="call-back" onPress={onBack} hitSlop={12} accessibilityRole="button" accessibilityLabel="Back">
          <Text style={styles.back}>‹ Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>In-app call</Text>
        <View style={{ width: 56 }} />
      </View>
      <View style={styles.rule} />
      <Text style={[type.eyebrow, styles.eyebrow]}>Live voice check during a call</Text>
      <Text style={[type.body, styles.explain]}>
        Two people on this app talk over an internet audio call. While you talk, the other person’s audio is checked on this device in 4-second windows. Your own microphone is never scored.
      </Text>
      <Text style={[type.eyebrow, styles.label]}>Your calling server</Text>
      <TextInput testID="call-server" style={styles.input} value={server} onChangeText={setServer} placeholder="https://call.yourdomain.com/"
                 autoCapitalize="none" autoCorrect={false} keyboardType="url" placeholderTextColor={colors.inkMute} />
      <Text style={[type.eyebrow, styles.label]}>Or paste a room invitation</Text>
      <TextInput testID="call-invite" style={styles.input} value={invite} onChangeText={setInvite} placeholder="https://…/?room=…"
                 autoCapitalize="none" autoCorrect={false} keyboardType="url" placeholderTextColor={colors.inkMute} />
      <Text style={[type.small, styles.hint]}>
        {status == null ? 'Checking detector model…' : liveAvailable
          ? (status.loaded ? 'Detector model ready · live check on' : 'Detector model installed · loads when the call starts')
          : 'Detector model not installed · the call works, but without a live check'}
      </Text>
      {preparing && <Text style={[type.small, styles.hint]} testID="call-preparing">{preparing}</Text>}
      {msg && <Text style={[type.small, styles.error]} testID="call-msg">{msg}</Text>}
      <PrimaryButton testID="call-open" label="Open calling room" onPress={start} />
      <Text style={[type.fine, styles.foot]}>
        HTTPS calling server required. Both people join the same room. Keep this screen open during the call. No ringing or background calls in this version.
      </Text>
    </ScrollView>
  );
}

function LiveStrip({ latest, history, available }: { latest: LiveResult | null; history: LiveResult[]; available: boolean }) {
  if (!available) return <Text style={[type.small, styles.strip]} testID="live-strip">Live check off: detector model not installed.</Text>;
  if (!latest) return <Text style={[type.small, styles.strip]} testID="live-strip">Live check waiting for the other participant’s audio…</Text>;
  const ok = latest.kind === 'window' && latest.status === 'ok';
  const synthetic = latest.prediction === 'synthetic';
  const verdict = ok ? (synthetic ? 'Likely synthetic' : 'Likely genuine') : latest.status === 'rejected' ? 'Not enough speech' : 'Check failed';
  const detail = ok ? `score ${latest.synthetic_score?.toFixed(4)} (uncalibrated)` : (latest.reject_reason ?? latest.message ?? '');
  const n = history.filter((h) => h.kind === 'window').length;
  return (
    <View style={styles.stripBox} testID="live-strip">
      <Text style={[styles.stripVerdict, { color: ok ? (synthetic ? colors.danger : colors.ok) : colors.inkSoft }]} testID="live-verdict">{verdict}</Text>
      <Text style={type.small} testID="live-detail">{detail} · window {latest.window_index ?? '–'} · {latest.timing_ms ? `${(latest.timing_ms.total / 1000).toFixed(1)} s` : ''}{latest.dropped ? ` · ${latest.dropped} skipped` : ''}</Text>
      <Text style={type.fine}>Latest 4 s window of the other participant · {n} recent · model output at an untuned 0.5 threshold, not a verified verdict</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { paddingHorizontal: space.lg, paddingBottom: space.lg },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: space.md, paddingHorizontal: space.lg },
  back: { fontSize: 17, color: colors.ink, fontWeight: '600', width: 80 },
  headerTitle: { fontSize: 17, fontWeight: '600', color: colors.ink },
  rule: { height: 1, backgroundColor: colors.hairline },
  eyebrow: { marginTop: space.lg, marginBottom: space.sm },
  explain: { marginBottom: space.md },
  label: { marginTop: space.md, marginBottom: 6, fontSize: 11 },
  input: { borderWidth: 1, borderColor: colors.hairline, borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, color: colors.ink, backgroundColor: colors.surfaceStrong },
  hint: { marginTop: space.sm, textAlign: 'center' },
  error: { marginTop: space.sm, color: colors.danger, textAlign: 'center' },
  foot: { textAlign: 'center', marginTop: space.md, paddingHorizontal: space.lg },
  stateLine: { paddingHorizontal: space.lg, marginBottom: 6 },
  strip: { paddingHorizontal: space.lg, paddingVertical: 8 },
  stripBox: { marginHorizontal: space.lg, marginBottom: space.sm, padding: 12, borderRadius: 12, borderWidth: 1, borderColor: colors.hairline, backgroundColor: colors.surfaceStrong },
  stripVerdict: { fontSize: 18, fontWeight: '700' },
  callView: { flex: 1, marginHorizontal: space.md, borderRadius: 12, overflow: 'hidden', backgroundColor: colors.surface },
});
