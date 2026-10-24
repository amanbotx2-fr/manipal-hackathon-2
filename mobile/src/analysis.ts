import type { LocalResult, TimelineResult } from './detector';

// One completed on-device analysis, shared between the result screen (plain-language summary)
// and the Dev screen (raw score, windows, timings, attribution). Values are the detector's own.
export type PickedFile = { uri: string; name: string; size?: number; mimeType: string };

export type Analysis = {
  result: LocalResult;
  file: PickedFile;
  /** Native cold-load time if the model was loaded as part of this check, else null (already loaded). */
  loadMs: number | null;
  /** JS wall-clock time of the analyseFile call (native pipeline + bridge). */
  wallMs: number;
  finishedAt: number;
  /** Full-clip per-window timeline; present when the clip was longer than the single-verdict path (10 s). */
  timeline?: TimelineResult;
};
