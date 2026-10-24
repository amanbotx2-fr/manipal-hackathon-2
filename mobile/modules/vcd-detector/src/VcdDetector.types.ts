export type InstallProgress =
  | { phase: 'downloading' | 'verifying' | 'done'; doneBytes: number; totalBytes: number }
  | { phase: 'error'; error: string };

export type VcdDetectorModuleEvents = {
  onInstallProgress: (p: InstallProgress) => void;
};

export type Memory = {
  rss_mb: number | null;
  peak_rss_mb: number | null;
  pss_mb: number | null;
  native_heap_mb: number;
  device_total_mb: number;
  device_avail_mb: number;
  low_memory: boolean;
};

export type LoadInfo = {
  loadMs: number;
  settings: Record<string, unknown>;
  memoryBefore: Memory;
  memoryAfter: Memory;
  alreadyLoaded?: boolean;
};

export type ModelStatus = {
  installed: boolean;
  installing: boolean;
  modelPath: string;
  modelBytes: number;
  partialBytes: number;
  expectedBytes: number;
  sha256: string;
  revision: string;
  freeBytes: number | null;
  loaded: boolean;
  lastLoad: LoadInfo | null;
  referenceAvailable: boolean;
  attribution: string;
  adapter: AdapterSpec;
};

/** Dhwani's preprocessing modes; other adapters declare their own in AdapterSpec.preprocessingModes. */
export type Mode = 'train_peak' | 'train_zscore' | 'api_replica';

/** Declared contract of a model adapter (mirrors AdapterSpec.toMap() in DetectorAdapter.kt). */
export type AdapterSpec = {
  id: string;
  displayName: string;
  version: string;
  model: { repo: string; revision: string; file: string; sha256: string; bytes: number; licence: string };
  sampleRate: number;
  windowSamples: number;
  hopSamples: number;
  minDurationS: number;
  maxDurationS: number;
  batchSize: number;
  preprocessingModes: string[];
  defaultMode: string;
  outputLabels: string[];
  positiveIndex: number;
  aggregation: string;
  decisionThreshold: number;
  thresholdTuned: boolean;
  scoreCalibrated: boolean;
  attribution: string;
};

export type ArrayDiff = {
  device_len: number; colab_len: number; overlap: number; lag: number;
  max_abs_diff: number; rms_diff: number | null; snr_db: number | null;
};

export type Comparison =
  | { available: false; reason: string }
  | {
      available: true; clip: string; mode: Mode;
      native_rate: { device: number; colab: number };
      native_mono: ArrayDiff; native_mono_best_lag: ArrayDiff; x16: ArrayDiff;
      window_count: { device: number; colab: number };
      windows: (ArrayDiff & { device_logits: number[]; colab_logits: number[] })[];
      clip_score: { device: number; colab: number; abs_diff: number };
    };

export type LocalResult = {
  status: 'ok' | 'rejected';
  prediction: 'genuine' | 'synthetic' | null;
  synthetic_score: number | null;
  synthetic_score_max: number | null;
  score_is_calibrated: boolean;
  decision_threshold: number;
  threshold_is_tuned?: boolean;
  adapter?: { id: string; version: string; model_revision: string };
  reject_reason: string | null;
  duration_s?: number;
  analysed_duration_s?: number;
  truncated?: boolean;
  audio: Record<string, unknown> | null;
  windows: { start_s: number; logits: number[]; synthetic_score: number; ms: number }[];
  preprocessing: Mode;
  timing_ms: {
    decode?: number; resample?: number; preprocess?: number; model_load?: number | null;
    model_total?: number; per_window_mean?: number; total: number;
  };
  memory?: Memory;
  session?: Record<string, unknown>;
  source: { sha256?: string; in_memory?: boolean };
  comparison?: Comparison;
  dump_dir?: string;
  on_device: true;
};

export type ReferenceRun = {
  windows_run: number;
  max_abs_logit_diff: number;
  model_load_ms: number | null;
  rows: {
    clip: string; mode: Mode; window: number; device_logits: number[]; colab_logits: number[];
    max_abs_logit_diff: number; device_score: number; colab_score: number; ms: number;
  }[];
  memory: Memory;
  session: Record<string, unknown>;
};

export type TimelineWindow = {
  start_s: number; end_s: number; status: 'ok' | 'rejected';
  synthetic_score?: number; prediction?: 'genuine' | 'synthetic'; logits?: number[]; reject_reason?: string;
};
export type TimelineResult = {
  status: 'ok' | 'rejected';
  reject_reason?: string;
  duration_s?: number; truncated?: boolean; window_s?: number; hop_s?: number;
  windows: TimelineWindow[];
  windows_ok?: number; windows_synthetic?: number; synthetic_span_s?: number;
  preprocessing?: string; score_is_calibrated?: boolean; decision_threshold?: number;
  timing_ms?: { model_load?: number | null; model_total?: number; total: number };
  note?: string;
};
