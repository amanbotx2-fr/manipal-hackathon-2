const $ = id => document.getElementById(id);
let pc, socket, mic, context, source, tap, remoteStream;
let candidates = [], stopped = true, joining = false, generation = 0;
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
  $('remote').srcObject = null; remoteStream = null; candidates = [];
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
  } else if (msg.type === 'peer-left') await stop('Other participant left');
  else if (msg.type === 'error') await stop(msg.message);
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
if (invitedRoom !== null) {
  if (validRoom(invitedRoom)) { $('room').value = invitedRoom; status('Room invitation loaded — tap Join call'); }
  else status('Invalid room invitation — generate a new room code');
}
updateRoomLink();
if (invitedRoom === null) status('Create a room or open an invitation');
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
        status('Call connected'); document.body.classList.add('in-call');
        $('dial-title').textContent = 'You’re connected';
        $('dial-subtitle').textContent = 'Audio call in progress';
      }
      else if (pc.connectionState === 'failed') stop('Connection failed — check TURN configuration');
      else if (pc.connectionState === 'disconnected') status('Connection interrupted');
    };
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`);
    socket.onopen = () => { joining = false; send({ type: 'join', room }); };
    // Serialize SDP and ICE operations to avoid races while setRemoteDescription resolves.
    let queue = Promise.resolve();
    socket.onmessage = event => { queue = queue.then(() => signal(JSON.parse(event.data))).catch(e => stop(e.message)); };
    socket.onclose = () => { if (!stopped) stop('Signaling disconnected — rejoin the room'); };
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
