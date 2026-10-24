// Model-independent detector interface used by the UI (and, later, the in-app calling code).
// Nothing here knows about Dhwani: the active model adapter (native DetectorAdapter) declares
// its checksum, sample rate, windows, preprocessing, output mapping and threshold in AdapterSpec.
import Native from '../../modules/vcd-detector/src';
import type {
  AdapterSpec, InstallProgress, LoadInfo, LocalResult, Memory, ModelStatus, ReferenceRun,
} from '../../modules/vcd-detector/src';

export type { AdapterSpec, InstallProgress, LoadInfo, LocalResult, Memory, ModelStatus, ReferenceRun };

export type AnalyseOptions = {
  /** One of spec.preprocessingModes; defaults to spec.defaultMode. */
  mode?: string;
  /** Compare against an on-device reference export, if present (diagnostics). */
  compare?: boolean;
  /** Write device tensors for offline comparison (diagnostics). */
  dump?: boolean;
};

export interface Detector {
  spec(): Promise<AdapterSpec>;
  status(): Promise<ModelStatus>;
  install(): Promise<void>;
  cancelInstall(): Promise<void>;
  onInstallProgress(listener: (p: InstallProgress) => void): { remove(): void };
  load(): Promise<LoadInfo>;
  /** Picked/recorded file (content:// or file:// URI). Runs on-device; no upload. */
  analyseFile(uri: string, opts?: AnalyseOptions): Promise<LocalResult>;
  /**
   * In-memory PCM, float32 samples in [-1, 1], interleaved channels, any sample rate (resampled to
   * spec.sampleRate natively). Intended for remote-participant audio windows from in-app calls later.
   */
  analysePcm(samples: Float32Array, sampleRate: number, channels: number, opts?: AnalyseOptions): Promise<LocalResult>;
  /** Diagnostics: run exported reference windows through the device runtime. */
  runReferenceTensors(maxWindows?: number): Promise<ReferenceRun>;
  memory(): Promise<Memory>;
}

async function resolveMode(opts?: AnalyseOptions): Promise<string> {
  if (opts?.mode) return opts.mode;
  return (await Native.getAdapterSpec()).defaultMode;
}

/** The on-device ONNX detector backed by the native vcd-detector module and its default adapter. */
export const onDeviceDetector: Detector = {
  spec: () => Native.getAdapterSpec(),
  status: () => Native.getStatus(),
  install: async () => { await Native.installModel(); },
  cancelInstall: () => Native.cancelInstall(),
  onInstallProgress: (listener) => Native.addListener('onInstallProgress', listener),
  load: () => Native.loadModel(),
  analyseFile: async (uri, opts) =>
    Native.analyseFile(uri, (await resolveMode(opts)) as any, !!opts?.compare, !!opts?.dump),
  analysePcm: async (samples, sampleRate, channels, opts) => {
    // Little-endian float32 bytes (all supported Android/iOS CPUs are little-endian).
    const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    return Native.analysePcm(bytes, sampleRate, channels, (await resolveMode(opts)) as any);
  },
  runReferenceTensors: (maxWindows = 1000) => Native.runReferenceTensors(maxWindows),
  memory: () => Native.getMemory(),
};

export default onDeviceDetector;
