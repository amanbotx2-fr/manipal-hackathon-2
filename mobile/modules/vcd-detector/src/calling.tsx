import { requireNativeViewManager } from 'expo-modules-core';
import type { ComponentType } from 'react';
import type { ViewProps } from 'react-native';

// In-app call view (WebView WebRTC from the vendored LiveCalling kit) with live on-device checks
// of the remote participant's audio. Android only for now.

export type CallStateEvent = { nativeEvent: { state: string } };

export type LiveResult = {
  kind: 'window' | 'error';
  status: 'ok' | 'rejected' | 'error';
  prediction?: 'genuine' | 'synthetic' | null;
  synthetic_score?: number | null;
  reject_reason?: string | null;
  message?: string;
  window_index?: number;
  dropped?: number;
  audio_end_s: number;
  window_s?: number;
  preprocessing?: string;
  timing_ms?: { total: number; model_total?: number; model_load?: number | null };
  decision_threshold?: number;
};
export type LiveResultEvent = { nativeEvent: LiveResult };

export type VcdCallViewProps = ViewProps & {
  /** HTTPS origin of the calling server, optionally with ?room=… Set only after RECORD_AUDIO is granted. */
  serverUrl: string;
  /** Score the remote participant's audio on-device (default true). Never scores the local mic. */
  liveCheck?: boolean;
  onCallState?: (e: CallStateEvent) => void;
  onLiveResult?: (e: LiveResultEvent) => void;
};

export const VcdCallView: ComponentType<VcdCallViewProps> = requireNativeViewManager('VcdCalling');
