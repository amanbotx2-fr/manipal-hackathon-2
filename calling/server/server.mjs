import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { WebSocketServer, WebSocket } from 'ws';

export function createServer({ allowedOrigin, iceServers = [] } = {}) {
  const assets = new Map([['/', ['index.html', 'text/html']], ['/call.js', ['call.js', 'text/javascript']], ['/pcm-worklet.js', ['pcm-worklet.js', 'text/javascript']]]);
  const server = http.createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
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
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { server } = createServer({ allowedOrigin: process.env.ALLOWED_ORIGIN, iceServers: JSON.parse(process.env.ICE_SERVERS_JSON || '[]') });
  server.listen(Number(process.env.PORT || 8788), '0.0.0.0', () => console.log('WebRTC demo on port', server.address().port));
}
