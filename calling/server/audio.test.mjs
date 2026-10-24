import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

test('remote tap emits continuous mono frames without duplicate playback', async () => {
  let Processor; const frames = [];
  const scope = vm.createContext({
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: samples => frames.push([...samples]) }; } },
    registerProcessor: (_, type) => { Processor = type; }, Float32Array,
  });
  vm.runInContext(await readFile(new URL('../web/pcm-worklet.js', import.meta.url), 'utf8'), scope);
  const tap = new Processor();
  for (let block = 0; block < 5; block++) {
    const left = Float32Array.from({ length: 128 }, (_, i) => (block * 128 + i) / 1000);
    const output = new Float32Array(128).fill(1);
    assert.equal(tap.process([[left, new Float32Array(128)]], [[output]]), true);
    assert.ok(output.every(x => x === 0));
  }
  assert.equal(frames.length, 2); assert.equal(frames[0].length, 320);
  const samples = frames.flat();
  for (let i = 0; i < 640; i++) assert.ok(Math.abs(samples[i] - i / 2000) < 1e-6);
});
