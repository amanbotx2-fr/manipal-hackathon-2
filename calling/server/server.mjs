import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, WebSocket } from 'ws';

export function createServer({ allowedOrigin, iceServers = [] } = {}) {
  const assets = new Map([['/', ['index.html', 'text/html']], ['/call.js', ['call.js', 'text/javascript']], ['/pcm-worklet.js', ['pcm-worklet.js', 'text/javascript']]]);
  // Android App Links: lets https://<host>/?room=... open the Voice Check app directly. The statement
  // lists the app package and its signing certificate fingerprint(s) (ASSETLINKS_JSON env).
  const assetlinks = process.env.ASSETLINKS_JSON || '';
  const server = http.createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (pathname === '/.well-known/assetlinks.json') {
      if (!assetlinks) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'max-age=300' });
      return res.end(assetlinks);
    }
    if (pathname === '/config') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      return res.end(JSON.stringify({ iceServers }));
    }
    const asset = assets.get(pathname);
    if (!asset) { res.writeHead(404); return res.end(); }
    try {
      const data = await readFile(new URL(`../web/${asset[0]}`, import.meta.url));
      res.writeHead(200, { 'Content-Type': asset[1], 'Cache-Control': 'no-store', 'Permissions-Policy': 'microphone=(self)' });
      res.end(data);
    } catch { res.writeHead(500); res.end('Asset unavailable'); }
  });
  const wss = new WebSocketServer({ server, maxPayload: 64 * 1024 });
  const rooms = new Map();
  const send = (ws, data) => { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data)); };
  function leave(ws) {
    const room = rooms.get(ws.room);
    if (!room) return;
    room.delete(ws);
    for (const peer of room) send(peer, { type: 'peer-left' });
    if (!room.size) rooms.delete(ws.room);
    ws.room = undefined;
  }
  wss.on('connection', (ws, req) => {
    if (allowedOrigin && req.headers.origin !== allowedOrigin) return ws.close(1008, 'Origin not allowed');
    ws.alive = true;
    ws.on('pong', () => { ws.alive = true; });
    ws.on('message', raw => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch { return ws.close(1008, 'Invalid JSON'); }
      if (msg.type === 'join') {
        if (ws.room || typeof msg.room !== 'string' || !/^[a-zA-Z0-9_-]{8,64}$/.test(msg.room)) return ws.close(1008, 'Invalid room');
        const room = rooms.get(msg.room) ?? new Set();
        if (room.size >= 2) { send(ws, { type: 'error', message: 'Room is full' }); return ws.close(1008, 'Room full'); }
        ws.room = msg.room; room.add(ws); rooms.set(msg.room, room);
        send(ws, { type: 'joined', waiting: room.size === 1 });
        if (room.size === 2) {
          const [caller, callee] = [...room];
          send(callee, { type: 'ready', initiator: false });
          send(caller, { type: 'ready', initiator: true });
        }
      } else if (['offer', 'answer', 'candidate'].includes(msg.type)) {
        if (!ws.room) return ws.close(1008, 'Join first');
        if (msg.type === 'candidate' ? !(msg.candidate && typeof msg.candidate.candidate === 'string') : typeof msg.sdp !== 'string') return ws.close(1008, 'Invalid signaling');
        for (const peer of rooms.get(ws.room) ?? []) if (peer !== ws) send(peer, msg.type === 'candidate' ? { type: msg.type, candidate: msg.candidate } : { type: msg.type, sdp: msg.sdp });
      } else ws.close(1008, 'Unknown message');
    });
    ws.on('close', () => leave(ws));
    ws.on('error', () => leave(ws));
  });
  const heartbeat = setInterval(() => {
    for (const ws of wss.clients) {
      if (!ws.alive) { ws.terminate(); continue; }
      ws.alive = false; ws.ping();
    }
  }, 15000);
  server.on('close', () => clearInterval(heartbeat));
  return { server, wss };
}
// ICE servers for a deployment. Without STUN, peers on different networks (or on mobile data)
// never find a route: no audio, then "Connection failed" after ~10 s. ICE_SERVERS_JSON overrides
// everything; otherwise public STUN is used, plus TURN when TURN_URLS/TURN_USERNAME/TURN_CREDENTIAL
// are set (needed when both sides sit behind strict NAT/CGNAT, common on mobile data).
export function iceServersFromEnv(env = process.env) {
  if (env.ICE_SERVERS_JSON) return JSON.parse(env.ICE_SERVERS_JSON);
  const servers = [{ urls: ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302'] }];
  if (env.TURN_URLS && env.TURN_USERNAME && env.TURN_CREDENTIAL) {
    servers.push({ urls: env.TURN_URLS.split(',').map(u => u.trim()).filter(Boolean), username: env.TURN_USERNAME, credential: env.TURN_CREDENTIAL });
  }
  return servers;
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const iceServers = iceServersFromEnv();
  console.log('ICE servers:', iceServers.map(s => s.urls).flat().join(', '), iceServers.some(s => s.username) ? '(TURN configured)' : '(no TURN: cross-network calls may fail)');
  const { server } = createServer({ allowedOrigin: process.env.ALLOWED_ORIGIN, iceServers });
  server.listen(Number(process.env.PORT || 8788), '0.0.0.0', () => console.log('WebRTC demo on port', server.address().port));
}
