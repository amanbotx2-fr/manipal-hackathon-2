const $ = id => document.getElementById(id);
let pc, socket, mic, context, source, tap, remoteStream;
let candidates = [], stopped = true, joining = false, generation = 0;
let initiator = false, restarts = 0, restartTimer = null, routeTimer = null;
const status = message => {
  $('status').textContent = message;
  window.VoiceCheckAndroid?.onCallState(message);
};
function controls(active) {
  $('join').disabled = active; $('new').disabled = active; $('room').disabled = active;
  $('mute').disabled = !active; $('hangup').disabled = !active;
  if (!active) {
    document.body.classList.remove('in-call');
    $('dial-title').textContent = 'Ready to connect';
    $('dial-subtitle').textContent = 'Your next conversation';
  }
}
function send(message) {
  if (socket?.readyState !== WebSocket.OPEN) throw new Error('Signaling connection unavailable');
  socket.send(JSON.stringify(message));
}
async function stop(message = 'Call ended') {
  stopped = true; joining = false; generation++;
  const ws = socket; socket = null; if (ws) { ws.onclose = null; ws.close(); }
  if (pc) { pc.onconnectionstatechange = null; pc.close(); pc = null; }
  mic?.getTracks().forEach(t => t.stop()); mic = null;
  source?.disconnect(); tap?.disconnect(); source = tap = null;
  const ctx = context; context = null; if (ctx) ctx.close().catch(() => {});
  $('remote').srcObject = null; remoteStream = null; candidates = []; restarts = 0; initiator = false;
  clearTimeout(restartTimer); clearInterval(routeTimer); restartTimer = routeTimer = null;
  $('mute').textContent = 'Mute microphone'; controls(false); status(message);
}
async function remoteAudio(track) {
  remoteStream = new MediaStream([track]); $('remote').srcObject = remoteStream;
  $('remote').play().catch(() => status('Connected — tap the audio player to hear the caller'));
  // Detection is optional: calling still works if this browser cannot create a 16 kHz context.
  try {
    const token = generation;
    const ctx = new AudioContext({ sampleRate: 16000 });
    context = ctx;
    if (ctx.sampleRate !== 16000) throw new Error('16 kHz context unsupported');
    await ctx.audioWorklet.addModule('/pcm-worklet.js');
    if (stopped || token !== generation) return;
    source = context.createMediaStreamSource(remoteStream);
    tap = new AudioWorkletNode(context, 'remote-pcm');
    tap.port.onmessage = ({ data }) => {
      if (stopped) return;
      // Subscribe in browser integrations. Do inference off the audio/UI thread.
      window.dispatchEvent(new CustomEvent('voicecheck-remote-pcm', { detail: { samples: data, sampleRate: 16000 } }));
      if (window.VoiceCheckAndroid) {
        const bytes = new Uint8Array(data.length * 2); const view = new DataView(bytes.buffer);
        for (let i = 0; i < data.length; i++) view.setInt16(i * 2, Math.round(Math.max(-1, Math.min(1, data[i])) * 32767), true);
        window.VoiceCheckAndroid.onRemotePcm(btoa(String.fromCharCode(...bytes)));
      }
    };
    source.connect(tap); tap.connect(context.destination); await context.resume();
  } catch (e) { status(`Call connected; detector audio unavailable: ${e.message}`); }
}
async function signal(msg) {
  if (stopped || !pc) return;
  if (msg.type === 'joined') { status(msg.waiting ? 'Waiting for the other participant' : 'Connecting'); return; }
  if (msg.type === 'ready') initiator = !!msg.initiator;
  if (msg.type === 'ready' && msg.initiator) {
    const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
    send({ type: 'offer', sdp: pc.localDescription.sdp });
  } else if (msg.type === 'offer' || msg.type === 'answer') {
    await pc.setRemoteDescription({ type: msg.type, sdp: msg.sdp });
    for (const candidate of candidates) await pc.addIceCandidate(candidate);
    candidates = [];
    if (msg.type === 'offer') {
      const answer = await pc.createAnswer(); await pc.setLocalDescription(answer);
      send({ type: 'answer', sdp: pc.localDescription.sdp });
    }
  } else if (msg.type === 'candidate') {
    if (pc.remoteDescription) await pc.addIceCandidate(msg.candidate); else candidates.push(msg.candidate);
  } else if (msg.type === 'verdict') showPeerVerdict(msg);
  else if (msg.type === 'peer-left') await stop('Other participant left');
  else if (msg.type === 'error') await stop(msg.message);
}

// --- live verdict relay ----------------------------------------------------------------------
// The Voice Check app computes a verdict about the remote voice on the phone and pushes it here;
// we forward it so the other side can see what the phone concluded about THEIR voice.
window.voiceCheckVerdict = v => {
  try { if (socket?.readyState === WebSocket.OPEN) send({ type: 'verdict', verdict: v }); } catch {}
};
function showPeerVerdict(m) {
  const el = $('verdict'); if (!el) return;
  const ok = m.status === 'ok' && m.prediction;
  el.textContent = ok ? `Other phone's Voice Check on your voice: ${m.prediction === 'synthetic' ? 'Likely synthetic' : 'Likely genuine'} (uncalibrated score ${m.score?.toFixed(4)}, window ${m.window})`
    : `Other phone's Voice Check: ${m.reason || 'not enough speech yet'}`;
  el.className = 'model ' + (ok ? (m.prediction === 'synthetic' ? 'bad' : 'good') : '');
}
// --- resilience helpers ---------------------------------------------------------------
async function restartIce() {
  if (!pc || stopped || !initiator || socket?.readyState !== WebSocket.OPEN) return;
  restarts++;
  try {
    const offer = await pc.createOffer({ iceRestart: true });
    await pc.setLocalDescription(offer);
    send({ type: 'offer', sdp: pc.localDescription.sdp });
    status(`Reconnecting (attempt ${restarts})`);
  } catch (e) { status(`Reconnect failed: ${e.message}`); }
}
function scheduleRestart(delay) {
  clearTimeout(restartTimer);
  restartTimer = setTimeout(() => {
    if (!pc || stopped) return;
    if (pc.connectionState === 'connected') return;
    if (initiator) restartIce();
  }, delay);
}
// Show whether media flows directly or through TURN; useful when "no audio" is reported.
async function reportRoute() {
  if (!pc) return;
  try {
    const stats = await pc.getStats();
    let pair = null; const cands = new Map();
    stats.forEach(r => { if (r.type === 'local-candidate' || r.type === 'remote-candidate') cands.set(r.id, r); });
    stats.forEach(r => { if (r.type === 'candidate-pair' && r.state === 'succeeded' && (r.selected || r.nominated)) pair = r; });
    if (!pair) return;
    const l = cands.get(pair.localCandidateId), r = cands.get(pair.remoteCandidateId);
    const via = (l?.candidateType === 'relay' || r?.candidateType === 'relay') ? 'via TURN relay' : (l?.candidateType === 'host' && r?.candidateType === 'host') ? 'direct (same network)' : 'direct (NAT traversal)';
    status(`Call connected · ${via}`);
  } catch { /* stats unavailable */ }
}
const validRoom = room => /^[a-zA-Z0-9_-]{8,64}$/.test(room);
function updateRoomLink() {
  const room = $('room').value.trim();
  $('share').disabled = !validRoom(room);
  if (!validRoom(room)) { $('room-link').value = ''; return; }
  const url = new URL('/', location.href);
  url.searchParams.set('room', room);
  $('room-link').value = url.href;
  history.replaceState(null, '', url);
}
const invitedRoom = new URL(location.href).searchParams.get('room');
// Read before updateRoomLink() rewrites the URL to the clean ?room= form.
const autoJoinRequested = new URL(location.href).searchParams.get('autojoin') === '1';
if (invitedRoom !== null) {
  if (validRoom(invitedRoom)) { $('room').value = invitedRoom; status('Room invitation loaded — tap Join call'); }
  else status('Invalid room invitation — generate a new room code');
}
updateRoomLink();
if (invitedRoom === null) status('Create a room or open an invitation');
// Invitation opened in a browser on Android: offer to continue in the app. The intent: URL opens the
// installed app directly (no App Links verification needed); if it is not installed, Chrome stays here.
if (invitedRoom !== null && validRoom(invitedRoom) && /Android/i.test(navigator.userAgent) && !window.VoiceCheckAndroid) {
  const a = $('openapp');
  a.href = `intent://call?room=${encodeURIComponent(invitedRoom)}&server=${encodeURIComponent(new URL('/', location.href).href)}#Intent;scheme=voicecheck;package=com.hypenex.voiceclonecheck;S.browser_fallback_url=${encodeURIComponent(location.href)};end`;
  a.hidden = false; a.style.display = 'block';
}
// Opened by the Voice Check app with ?autojoin=1: join without a tap (the app already holds the mic permission).
if (invitedRoom !== null && validRoom(invitedRoom) && autoJoinRequested) {
  setTimeout(() => { if (stopped && !joining) $('join').onclick(); }, 300);
}
$('room').addEventListener('input', updateRoomLink);
$('new').onclick = () => {
  $('room').value = crypto.randomUUID().replaceAll('-', '').slice(0, 16);
  updateRoomLink(); status('Room ready — share the link, then join');
};
$('share').onclick = async () => {
  updateRoomLink();
  const url = $('room-link').value;
  if (!url) return;
  try {
    if (window.VoiceCheckAndroid?.shareRoom) {
      window.VoiceCheckAndroid.shareRoom($('room').value.trim());
    } else if (navigator.share) {
      await navigator.share({ title: 'Voice Check call', text: 'Join my audio call', url });
    } else if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(url); status('Room link copied');
    } else {
      $('room-link').focus(); $('room-link').select(); status('Copy the selected room link');
    }
  } catch (e) {
    if (e.name === 'AbortError') return;
    $('room-link').focus(); $('room-link').select(); status('Share unavailable — copy the selected link');
  }
};
$('join').onclick = async () => {
  if (joining || !stopped) return;
  const room = $('room').value.trim();
  if (!/^[a-zA-Z0-9_-]{8,64}$/.test(room)) return status('Use an 8–64 character room code');
  const token = ++generation;
  stopped = false; joining = true; controls(true); status('Requesting microphone');
  try {
    if (!window.isSecureContext) throw new Error('Open this page over HTTPS (localhost also works)');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
    if (stopped || token !== generation) { stream.getTracks().forEach(t => t.stop()); return; }
    mic = stream;
    const config = await fetch('/config').then(r => { if (!r.ok) throw new Error('Could not load ICE configuration'); return r.json(); });
    if (stopped || token !== generation) return;
    pc = new RTCPeerConnection(config);
    for (const track of mic.getAudioTracks()) pc.addTrack(track, mic);
    pc.onicecandidate = ({ candidate }) => { if (candidate && !stopped) { try { send({ type: 'candidate', candidate: candidate.toJSON() }); } catch (e) { stop(e.message); } } };
    pc.ontrack = ({ track }) => { if (track.kind === 'audio') remoteAudio(track); };
    pc.onconnectionstatechange = () => {
      if (!pc) return;
      if (pc.connectionState === 'connected') {
        clearTimeout(restartTimer); restartTimer = null; restarts = 0;
        status('Call connected'); document.body.classList.add('in-call');
        $('dial-title').textContent = 'You’re connected';
        $('dial-subtitle').textContent = 'Audio call in progress';
        reportRoute();
      }
      else if (pc.connectionState === 'disconnected') { status('Connection interrupted — reconnecting'); scheduleRestart(4000); }
      else if (pc.connectionState === 'failed') {
        if (initiator && restarts < 3 && socket?.readyState === WebSocket.OPEN) { status('Connection failed — retrying'); scheduleRestart(500); }
        else if (!initiator && restarts < 3) { restarts++; status('Connection failed — waiting for the other side to retry'); restartTimer = setTimeout(() => { if (pc?.connectionState === 'failed') stop('Could not connect. Both sides need STUN/TURN on this network; see the server configuration.'); }, 15000); }
        else stop('Could not connect. Both sides need STUN/TURN on this network; see the server configuration.');
      }
    };
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`);
    socket.onopen = () => { joining = false; send({ type: 'join', room }); };
    // Serialize SDP and ICE operations to avoid races while setRemoteDescription resolves.
    let queue = Promise.resolve();
    socket.onmessage = event => { queue = queue.then(() => signal(JSON.parse(event.data))).catch(e => stop(e.message)); };
    socket.onclose = () => {
      if (stopped) return;
      if (pc?.connectionState === 'connected') { socket = null; status('Call connected (signaling lost — reconnect unavailable until you rejoin)'); }
      else stop('Signaling disconnected — rejoin the room');
    };
    socket.onerror = () => { if (!stopped) stop('Could not reach signaling server'); };
  } catch (e) { if (token === generation) await stop(e.message); }
};
$('mute').onclick = () => {
  const track = mic?.getAudioTracks()[0]; if (!track) return;
  track.enabled = !track.enabled; $('mute').textContent = track.enabled ? 'Mute microphone' : 'Unmute microphone';
};
$('hangup').onclick = () => stop();
window.addEventListener('pagehide', () => { stop(); });
window.voiceCheckHangup = () => stop();
