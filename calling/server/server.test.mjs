import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import WebSocket from 'ws';
import { createServer } from './server.mjs';

function inbox(ws) {
  const messages = [], waiters = [];
  ws.on('message', raw => { const value = JSON.parse(raw); if (waiters.length) waiters.shift()(value); else messages.push(value); });
  return () => messages.length ? Promise.resolve(messages.shift()) : new Promise(resolve => waiters.push(resolve));
}
test('two peers exchange SDP and ICE; third is rejected; leaving notifies peer', { timeout: 10000 }, async () => {
  const { server, wss } = createServer(); server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const url = `ws://127.0.0.1:${server.address().port}`;
  const clients = [];
  try {
    const a = new WebSocket(url), readA = inbox(a); clients.push(a); await once(a, 'open');
    a.send(JSON.stringify({ type: 'join', room: 'room12345' })); assert.equal((await readA()).waiting, true);
    const b = new WebSocket(url), readB = inbox(b); clients.push(b); await once(b, 'open');
    b.send(JSON.stringify({ type: 'join', room: 'room12345' })); assert.equal((await readB()).type, 'joined');
    assert.equal((await readB()).initiator, false); assert.equal((await readA()).initiator, true);
    a.send(JSON.stringify({ type: 'offer', sdp: 'offer-sdp' })); assert.deepEqual(await readB(), { type: 'offer', sdp: 'offer-sdp' });
    b.send(JSON.stringify({ type: 'answer', sdp: 'answer-sdp' })); assert.equal((await readA()).sdp, 'answer-sdp');
    a.send(JSON.stringify({ type: 'candidate', candidate: { candidate: 'candidate:example' } })); assert.equal((await readB()).type, 'candidate');
    const c = new WebSocket(url), readC = inbox(c); clients.push(c); await once(c, 'open');
    c.send(JSON.stringify({ type: 'join', room: 'room12345' })); assert.equal((await readC()).message, 'Room is full');
    a.close(); assert.equal((await readB()).type, 'peer-left');
    const invitation = await fetch(`http://127.0.0.1:${server.address().port}/?room=room12345`);
    assert.equal(invitation.status, 200); assert.match(await invitation.text(), /Share room link/);
    const response = await fetch(`http://127.0.0.1:${server.address().port}/config`); assert.deepEqual(await response.json(), { iceServers: [] });
  } finally { for (const ws of clients) ws.terminate(); wss.close(); await new Promise(resolve => server.close(resolve)); }
});
test('rejects untrusted origins', { timeout: 5000 }, async () => {
  const { server, wss } = createServer({ allowedOrigin: 'https://calls.example.com' });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const ws = new WebSocket(`ws://127.0.0.1:${server.address().port}`, { origin: 'https://other.example' });
  try { const [code] = await once(ws, 'close'); assert.equal(code, 1008); }
  finally { ws.terminate(); wss.close(); await new Promise(resolve => server.close(resolve)); }
});
