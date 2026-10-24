import { NativeModule, requireNativeModule } from 'expo';

import type {
  AdapterSpec, LoadInfo, LocalResult, Memory, Mode, ModelStatus, ReferenceRun, TimelineResult, VcdDetectorModuleEvents,
} from './VcdDetector.types';

declare class VcdDetectorModule extends NativeModule<VcdDetectorModuleEvents> {
  getStatus(): Promise<ModelStatus>;
  getAdapterSpec(): Promise<AdapterSpec>;
  listAdapters(): Promise<AdapterSpec[]>;
  /** Starts the one-time pinned download (progress via onInstallProgress). Needs internet. */
  installModel(): Promise<boolean>;
  cancelInstall(): Promise<void>;
  deleteModel(): Promise<boolean>;
  loadModel(): Promise<LoadInfo>;
  unloadModel(): Promise<boolean>;
  /** Decode, resample, preprocess and score a picked file on-device. No network. */
  analyseFile(uri: string, mode: Mode, compare: boolean, dump: boolean): Promise<LocalResult>;
  /** Segment timeline over the whole clip (per-window verdicts, ~2 s resolution). */
  analyseTimeline(uri: string, mode: Mode, maxSeconds: number): Promise<TimelineResult>;
  /** In-memory little-endian float32 PCM (interleaved) - for future in-app call audio. */
  analysePcm(pcm: Uint8Array, sampleRate: number, channels: number, mode: Mode): Promise<LocalResult>;
  /** Milestone 1: run exported Colab windows through the on-device runtime and compare logits. */
  runReferenceTensors(maxWindows: number): Promise<ReferenceRun>;
  getMemory(): Promise<Memory>;
}

export default requireNativeModule<VcdDetectorModule>('VcdDetector');
