// AI caller demo: joins a room with NO microphone and streams a synthetic-speech sample on a loop as
// its audio track, so the other participant's Voice Check app has an AI voice to score.
// Same signaling as call.js (join / ready / offer / answer / candidate / verdict).
const $ = id => document.getElementById(id);
const room = new URL(location.href).searchParams.get('room') || '';
const valid = /^[a-zA-Z0-9_-]{8,64}$/.test(room);
let pc, socket, ctx, stopped = true, candidates = [];
const status = m => { $('status').textContent = m; };
const send = m => { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(m)); };

function showVerdict(m) {
  const ok = m.status === 'ok' && m.prediction;
  $('verdict').textContent = ok
    ? `The phone's Voice Check says the AI voice is: ${m.prediction === 'synthetic' ? 'LIKELY SYNTHETIC' : 'likely genuine'} (uncalibrated score ${m.score?.toFixed(4)}, window ${m.window})`
    : `Phone's Voice Check: ${m.reason || 'waiting for enough speech'}`;
  $('verdict').className = 'model ' + (ok ? (m.prediction === 'synthetic' ? 'bad' : 'good') : '');
}

async function stop(message = 'AI caller stopped') {
  stopped = true;
  const ws = socket; socket = null; if (ws) { ws.onclose = null; ws.close(); }
  if (pc) { pc.onconnectionstatechange = null; pc.close(); pc = null; }
  const c = ctx; ctx = null; if (c) c.close().catch(() => {});
  candidates = []; $('start').disabled = false; $('stop').disabled = true; status(message);
}

async function signal(msg) {
  if (stopped || !pc) return;
  if (msg.type === 'joined') return status(msg.waiting ? 'Waiting for the phone to join this room' : 'Connecting to the phone');
  if (msg.type === 'ready' && msg.initiator) {
    const offer = await pc.createOffer(); await pc.setLocalDescription(offer); send({ type: 'offer', sdp: pc.localDescription.sdp });
  } else if (msg.type === 'offer' || msg.type === 'answer') {
    await pc.setRemoteDescription({ type: msg.type, sdp: msg.sdp });
    for (const c of candidates) await pc.addIceCandidate(c); candidates = [];
    if (msg.type === 'offer') { const a = await pc.createAnswer(); await pc.setLocalDescription(a); send({ type: 'answer', sdp: pc.localDescription.sdp }); }
  } else if (msg.type === 'candidate') {
    if (pc.remoteDescription) await pc.addIceCandidate(msg.candidate); else candidates.push(msg.candidate);
  } else if (msg.type === 'verdict') showVerdict(msg);
  else if (msg.type === 'peer-left') await stop('The phone left the call');
  else if (msg.type === 'error') await stop(msg.message);
}

$('start').onclick = async () => {
  if (!valid) return status('Open this page from an invite link (…/bot.html?room=CODE)');
  stopped = false; $('start').disabled = true; $('stop').disabled = false;
  try {
    status('Loading the AI voice sample');
    ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await (await fetch('/samples/ai-voice-sample.wav')).arrayBuffer());
    const dest = ctx.createMediaStreamDestination();
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true; src.connect(dest); src.start();
    await ctx.resume();
    const config = await fetch('/config').then(r => r.json());
    pc = new RTCPeerConnection(config);
    for (const t of dest.stream.getAudioTracks()) pc.addTrack(t, dest.stream);
    pc.onicecandidate = ({ candidate }) => { if (candidate && !stopped) send({ type: 'candidate', candidate: candidate.toJSON() }); };
    pc.onconnectionstatechange = () => {
      if (!pc) return;
      if (pc.connectionState === 'connected') status('Connected: the AI voice is now speaking into the call');
      else if (pc.connectionState === 'failed') stop('Connection failed (STUN/TURN)');
    };
    socket = new WebSocket(`wss://${location.host}`);
    socket.onopen = () => send({ type: 'join', room });
    let queue = Promise.resolve();
    socket.onmessage = e => { queue = queue.then(() => signal(JSON.parse(e.data))).catch(err => stop(err.message)); };
    socket.onclose = () => { if (!stopped && pc?.connectionState !== 'connected') stop('Signaling disconnected'); };
  } catch (e) { stop(e.message); }
};
$('stop').onclick = () => stop();
$('room').textContent = valid ? room : '(no room in link)';
status(valid ? 'Ready. Tap Start to put the AI voice into the call.' : 'Invalid or missing room code in this link.');
