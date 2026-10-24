import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';

async function load(url, navigator = {}, bridge) {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, { value: '', disabled: false, textContent: '', addEventListener() {}, focus() {}, select() {} });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: get }, location: new URL(url), URL, crypto: webcrypto,
    history: { replaceState() {} }, navigator,
    window: { VoiceCheckAndroid: bridge, addEventListener() {} },
  });
  vm.runInContext(await readFile(new URL('../web/call.js', import.meta.url), 'utf8'), context);
  return get;
}
test('invitation prefills room without starting mic; clipboard gets the invitation', async () => {
  let copied;
  const get = await load('https://calls.example/?room=room12345', { clipboard: { writeText: async value => { copied = value; } } });
  assert.equal(get('room').value, 'room12345');
  assert.match(get('status').textContent, /tap Join call/);
  await get('share').onclick(); assert.equal(copied, 'https://calls.example/?room=room12345');
});
test('invalid invitation is rejected; generation enables sharing', async () => {
  const get = await load('https://calls.example/?room=bad');
  assert.equal(get('room').value, ''); assert.equal(get('share').disabled, true);
  get('new').onclick(); assert.equal(get('room').value.length, 16);
  assert.equal(get('share').disabled, false); assert.match(get('room-link').value, /\?room=/);
});
test('Android uses native share bridge', async () => {
  let shared;
  const get = await load('https://calls.example/?room=room12345', {}, { onCallState() {}, shareRoom: room => { shared = room; } });
  await get('share').onclick(); assert.equal(shared, 'room12345');
});
