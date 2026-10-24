import * as SecureStore from 'expo-secure-store';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Linking, PermissionsAndroid, Platform, Pressable, ScrollView, Share, StyleSheet, Text, TextInput, View } from 'react-native';

import { VcdCallView, type LiveResult } from '../../modules/vcd-detector/src/calling';
import detector, { type ModelStatus } from '../detector';
import { colors, space, type } from '../theme';
import { PrimaryButton, SecondaryButton } from './HomeScreen';

// App-to-app audio call with a live voice check of the OTHER participant.
// Flow: Start a call -> a room is created and an invite link shared -> the friend taps the link ->
// Android App Links open this app (App.tsx passes the URL in as `incomingUrl`) -> auto-join.
// The call itself is the vendored LiveCalling kit (WebView WebRTC); the live check runs natively on
// the shared detector. Scores are uncalibrated model outputs. Nothing here scores the local mic.

export const DEFAULT_SERVER = 'https://call.13-234-179-79.sslip.io/';
const SERVER_KEY = 'vcd.callServer';
const isHttps = (s: string) => /^https:\/\/[^\s/]+/i.test(s.trim());
const normaliseServer = (s: string) => s.trim().replace(/\/+$/, '') + '/';
const makeRoom = () => Array.from({ length: 16 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('');

/** Parse an invitation: https://<host>/?room=CODE or voicecheck://call?room=CODE[&server=https://...]. */
export function parseInvite(url: string, fallbackServer: string): { server: string; room: string } | null {
  try {
    const u = new URL(url);
    const room = u.searchParams.get('room') ?? '';
    if (!/^[a-zA-Z0-9_-]{8,64}$/.test(room)) return null;
    if (u.protocol === 'https:') return { server: `${u.origin}/`, room };
    const server = u.searchParams.get('server');
    return { server: server && isHttps(server) ? normaliseServer(server) : fallbackServer, room };
  } catch { return null; }
}

type Props = { onBack: () => void; incomingUrl?: string | null; onConsumedIncoming?: () => void };

export default function CallScreen({ onBack, incomingUrl, onConsumedIncoming }: Props) {
  const [server, setServer] = useState(DEFAULT_SERVER);
  const [invite, setInvite] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [preparing, setPreparing] = useState<string | null>(null);
  const [call, setCall] = useState<{ server: string; room: string; aiDemo?: boolean } | null>(null);
  const [callState, setCallState] = useState('');
  const [latest, setLatest] = useState<LiveResult | null>(null);
  const [history, setHistory] = useState<LiveResult[]>([]);
  const starting = useRef(false);

  useEffect(() => {
    SecureStore.getItemAsync(SERVER_KEY).then((v) => { if (v && isHttps(v)) setServer(normaliseServer(v)); }).catch(() => {});
    detector.status().then(setStatus).catch(() => {});
  }, []);

  const begin = useCallback(async (target: { server: string; room: string; aiDemo?: boolean }) => {
    if (starting.current) return;
    starting.current = true; setMsg(null);
    try {
      if (Platform.OS !== 'android') { setMsg('In-app calls are Android-only in this build.'); return; }
      const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO, {
        title: 'Microphone', message: 'Needed so the other person can hear you during the call.', buttonPositive: 'Allow',
      });
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) { setMsg('Microphone permission is needed to call.'); return; }
      await SecureStore.setItemAsync(SERVER_KEY, target.server).catch(() => {});
      const s = await detector.status(); setStatus(s);
      if (s.installed && !s.loaded) {
        setPreparing('Loading the detector model so the live check is ready…');
        await detector.load().catch((e) => setMsg(`Live check unavailable: ${e?.message ?? e}`));
        setPreparing(null);
      }
      setLatest(null); setHistory([]); setCallState('Opening calling room');
      setCall(target);
    } finally { starting.current = false; }
  }, []);

  // Friend tapped an invite link: App Links opened the app and handed us the URL.
  useEffect(() => {
    if (!incomingUrl) return;
    // Handle on the next tick: this reacts to an external navigation event, so state updates are
    // deferred rather than made synchronously inside the effect.
    const t = setTimeout(() => {
      const parsed = parseInvite(incomingUrl, server);
      onConsumedIncoming?.();
      if (!parsed) { setMsg('That link is not a valid call invitation.'); return; }
      setServer(parsed.server); setInvite('');
      begin(parsed);
    }, 0);
    return () => clearTimeout(t);
  }, [incomingUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const startCall = () => {
    const srv = normaliseServer(server);
    if (!isHttps(srv)) { setMsg('Enter the HTTPS address of the calling server.'); return; }
    begin({ server: srv, room: makeRoom() });
  };
  // Demo: the other participant is a web page that streams a synthetic voice (no microphone).
  const startAiDemo = () => {
    const srv = normaliseServer(server);
    if (!isHttps(srv)) { setMsg('Enter the HTTPS address of the calling server.'); return; }
    begin({ server: srv, room: makeRoom(), aiDemo: true });
  };
  const botLink = call ? `${call.server}bot.html?room=${call.room}` : '';
  const openBot = () => Linking.openURL(botLink).catch(() => setMsg('Could not open a browser for the AI caller page.'));
  const joinFromLink = () => {
    const parsed = parseInvite(invite.trim(), normaliseServer(server));
    if (!parsed) { setMsg('Paste a valid invitation link (…/?room=CODE).'); return; }
    begin(parsed);
  };
  const inviteLink = call ? `${call.server}?room=${call.room}` : '';
  const shareInvite = () => Share.share({ message: `Join my Voice Check call: ${inviteLink}`, url: inviteLink }).catch(() => {});
  const end = useCallback(() => { setCall(null); setCallState(''); }, []);
  const onLive = useCallback((e: { nativeEvent: LiveResult }) => {
    const r = e.nativeEvent; setLatest(r); setHistory((h) => [r, ...h].slice(0, 6));
  }, []);
  const liveAvailable = !!status?.installed;

  if (call) {
    return (
      <View style={styles.screen}>
        <View style={styles.header}>
          <Pressable testID="call-end" onPress={end} hitSlop={12} accessibilityRole="button" accessibilityLabel="End call and go back">
            <Text style={styles.back}>‹ End call</Text>
          </Pressable>
          <Text style={styles.headerTitle}>In-app call</Text>
          <Pressable testID="call-share" onPress={shareInvite} hitSlop={12} accessibilityRole="button" accessibilityLabel="Share invite link">
            <Text style={styles.share}>Share invite</Text>
          </Pressable>
        </View>
        <Text style={[type.small, styles.stateLine]} testID="call-state" numberOfLines={2}>{callState || '…'}</Text>
        <Text style={[type.fine, styles.stateLine]} testID="call-room">Room {call.room} · {call.aiDemo ? 'AI caller demo: open the AI caller page (on this phone or any device) and tap Start' : 'the other person opens your invite link and the app joins automatically'}</Text>
        {call.aiDemo && (
          <View style={styles.aiRow}>
            <PrimaryButton testID="call-open-bot" label="Open AI caller page" onPress={openBot} compact />
            <SecondaryButton testID="call-share-bot" label="Share AI link" onPress={() => Share.share({ message: botLink, url: botLink }).catch(() => {})} />
          </View>
        )}
        <LiveStrip latest={latest} history={history} available={liveAvailable} />
        <VcdCallView style={styles.callView} serverUrl={`${call.server}?room=${call.room}&autojoin=1`} liveCheck={liveAvailable}
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
        <View style={{ width: 80 }} />
      </View>
      <View style={styles.rule} />
      <Text style={[type.eyebrow, styles.eyebrow]}>Call a friend with a live check</Text>
      <Text style={[type.body, styles.explain]} maxFontSizeMultiplier={1.3}>
        Start a call, share the invite link. When your friend opens it, this app opens on their phone and joins. While you talk, the other person’s audio is checked on this device in 4-second windows. Your own microphone is never scored.
      </Text>
      <PrimaryButton testID="call-start" label="Start a call" onPress={startCall} />
      <View style={{ flexDirection: 'row', marginTop: space.sm }}><SecondaryButton testID="call-ai" label="Demo: call an AI voice" onPress={startAiDemo} /></View>
      {preparing && <Text style={[type.small, styles.hint]} testID="call-preparing">{preparing}</Text>}
      {msg && <Text style={[type.small, styles.error]} testID="call-msg">{msg}</Text>}
      <Text style={[type.small, styles.hint]}>
        {status == null ? 'Checking detector model…' : liveAvailable
          ? (status.loaded ? 'Detector model ready · live check on' : 'Detector model installed · loads when the call starts')
          : 'Detector model not installed · the call works, but without a live check'}
      </Text>

      <View style={styles.rule} />
      <Text style={[type.eyebrow, styles.label]}>Got an invite link?</Text>
      <TextInput testID="call-invite" style={styles.input} value={invite} onChangeText={setInvite} placeholder="Paste it here: https://…/?room=…"
                 autoCapitalize="none" autoCorrect={false} keyboardType="url" placeholderTextColor={colors.inkMute} />
      <View style={{ flexDirection: 'row' }}><SecondaryButton testID="call-join" label="Join with link" onPress={joinFromLink} /></View>

      <Text style={[type.eyebrow, styles.label]}>Calling server (advanced)</Text>
      <TextInput testID="call-server" style={styles.input} value={server} onChangeText={setServer} placeholder={DEFAULT_SERVER}
                 autoCapitalize="none" autoCorrect={false} keyboardType="url" placeholderTextColor={colors.inkMute} />
      <Text style={[type.fine, styles.foot]}>
        Both phones need internet. Keep this screen open during the call. No ringing or background calls in this version.
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
  share: { fontSize: 15, color: colors.accent, fontWeight: '700', width: 100, textAlign: 'right' },
  headerTitle: { fontSize: 17, fontWeight: '600', color: colors.ink },
  rule: { height: 1, backgroundColor: colors.hairline, marginVertical: space.md },
  eyebrow: { marginTop: space.md, marginBottom: space.sm },
  explain: { marginBottom: space.sm },
  label: { marginTop: space.sm, marginBottom: 6, fontSize: 11 },
  input: { borderWidth: 1, borderColor: colors.hairline, borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, color: colors.ink, backgroundColor: colors.surfaceStrong, marginBottom: space.sm },
  hint: { marginTop: space.sm, textAlign: 'center' },
  error: { marginTop: space.sm, color: colors.danger, textAlign: 'center' },
  foot: { textAlign: 'center', marginTop: space.md, paddingHorizontal: space.lg },
  stateLine: { paddingHorizontal: space.lg, marginBottom: 6 },
  strip: { paddingHorizontal: space.lg, paddingVertical: 8 },
  stripBox: { marginHorizontal: space.lg, marginBottom: space.sm, padding: 12, borderRadius: 12, borderWidth: 1, borderColor: colors.hairline, backgroundColor: colors.surfaceStrong },
  stripVerdict: { fontSize: 18, fontWeight: '700' },
  callView: { flex: 1, marginHorizontal: space.md, borderRadius: 12, overflow: 'hidden', backgroundColor: colors.surface },
  aiRow: { flexDirection: 'row', gap: 10, marginHorizontal: space.lg, marginBottom: space.sm },
});
