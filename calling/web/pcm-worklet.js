// AudioContext requests 16 kHz, so this receives the browser's resampled audio.
// Only the received track connects to this node. Never connect the local mic.
class RemotePcm extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(320); this.offset = 0; }
  process(inputs, outputs) {
    for (const channel of outputs[0] || []) channel.fill(0); // detector branch stays inaudible
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const c of channels) value += c[i];
      this.buffer[this.offset++] = value / channels.length;
      if (this.offset === 320) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(320); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('remote-pcm', RemotePcm);
