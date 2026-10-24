// Client for the Colab detector API (src/vcd/vcd_api.py + vcd_service.py).
// Types mirror the JSON those files actually return.

export const REQUEST_TIMEOUT_MS = 180_000;
export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024; // server default VCD_MAX_UPLOAD_BYTES

export type Health = {
  status: 'ready' | 'not_ready';
  model_ready: boolean;
  load_error: string | null;
  default_preprocessing: string;
  preprocessing_modes: string[];
  busy: boolean;
  model: { model: string; revision: string; licence: string; authors: string };
  load_info?: Record<string, unknown>;
};

export type DetectResult = {
  status: 'ok' | 'rejected';
  prediction: 'genuine' | 'synthetic' | null;
  synthetic_score: number | null;
  synthetic_score_max: number | null;
  score_is_calibrated: boolean;
  decision_threshold: number;
  threshold_is_tuned: boolean;
  score_note: string;
  reject_reason: string | null;
  duration_s: number | null;
  analysed_duration_s: number | null;
  truncated: boolean | null;
  windows: { start_s: number; synthetic_score: number }[];
  preprocessing: string;
  preprocessing_note: string;
  timing_ms: { model: number | null; pipeline: number | null; queue_wait?: number; total?: number };
  model: { repo: string; revision: string };
  request_id?: string;
  upload_bytes?: number;
};

export type ErrorKind =
  | 'auth' | 'invalid_audio' | 'too_large' | 'unsupported_type' | 'busy' | 'not_ready'
  | 'bad_request' | 'tunnel' | 'server' | 'network' | 'timeout' | 'config' | 'aborted';

export class ApiError extends Error {
  constructor(public kind: ErrorKind, public title: string, message: string, public httpStatus?: number) {
    super(message);
  }
}

export type PickedFile = { uri: string; name: string; mimeType: string; size?: number };

export function normaliseBaseUrl(raw: string): string {
  const url = raw.trim().replace(/\/+$/, '');
  if (!/^https?:\/\/[^\s/]+/i.test(url)) {
    throw new ApiError('config', 'Invalid backend URL', 'Enter the full URL, e.g. https://example.trycloudflare.com');
  }
  return url;
}

type RawResponse = { status: number; body: string };

// XMLHttpRequest: native networking (off the UI thread), real timeout events,
// and FormData file parts streamed from the file URI with a native boundary.
function request(method: 'GET' | 'POST', url: string, headers: Record<string, string>,
                 body?: FormData, onXhr?: (xhr: XMLHttpRequest) => void): Promise<RawResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.timeout = REQUEST_TIMEOUT_MS;
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    xhr.onload = () => resolve({ status: xhr.status, body: xhr.responseText ?? '' });
    xhr.ontimeout = () => reject(new ApiError('timeout', 'Request timed out',
      `No response within ${REQUEST_TIMEOUT_MS / 1000} s. The server may be overloaded or the tunnel stalled.`));
    xhr.onerror = () => reject(new ApiError('network', 'Connection failed',
      'Could not reach the backend. Check the URL, your internet connection, and that the Colab tunnel is still running.'));
    xhr.onabort = () => reject(new ApiError('aborted', 'Cancelled', 'The request was cancelled.'));
    onXhr?.(xhr);
    xhr.send(body ?? null);
  });
}

function parseJson(raw: RawResponse): any {
  try {
    return JSON.parse(raw.body);
  } catch {
    return null;
  }
}

function httpError(raw: RawResponse, json: any): ApiError {
  const s = raw.status;
  const msg: string = json?.message ?? json?.reject_reason ?? '';
  if (s === 401) return new ApiError('auth', 'Authentication failed', 'The API key was missing or wrong. Update it in Settings.', s);
  if (s === 413) return new ApiError('too_large', 'File too large', msg || 'The server accepts uploads up to 8 MB.', s);
  if (s === 415) return new ApiError('unsupported_type', 'Unsupported file type', msg || 'Choose an audio file such as .m4a, .wav, .mp3 or .ogg.', s);
  if (s === 422) return new ApiError('invalid_audio', 'Audio rejected', json?.reject_reason || msg || 'The server could not analyse this audio.', s);
  if (s === 400) return new ApiError('bad_request', 'Bad request', msg || 'The server rejected the request.', s);
  if (s === 503 && json?.error === 'busy') return new ApiError('busy', 'Server busy', 'Another analysis is running. Try again in a few seconds.', s);
  if (s === 503) return new ApiError('not_ready', 'Model not ready', msg || 'The detector model is not loaded on the server.', s);
  if (s === 502 || s === 504 || s === 530 || s === 1033) {
    return new ApiError('tunnel', 'Backend unreachable', `The tunnel answered HTTP ${s}. The Colab server or tunnel may have stopped, or the URL changed.`, s);
  }
  const snippet = raw.body.slice(0, 160);
  return new ApiError('server', `Server error (HTTP ${s})`, msg || snippet || 'Unexpected response.', s);
}

export async function checkHealth(baseUrl: string): Promise<Health> {
  const raw = await request('GET', `${normaliseBaseUrl(baseUrl)}/health`, { Accept: 'application/json' });
  const json = parseJson(raw);
  if (raw.status !== 200 || !json || typeof json.model_ready !== 'boolean') throw httpError(raw, json);
  return json as Health;
}

export async function detect(baseUrl: string, apiKey: string, file: PickedFile,
                             onXhr?: (xhr: XMLHttpRequest) => void): Promise<DetectResult> {
  if (!apiKey.trim()) throw new ApiError('config', 'API key missing', 'Enter the API key in Settings.');
  if (file.size != null && file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError('too_large', 'File too large', `This file is ${(file.size / 1048576).toFixed(1)} MB; the server accepts up to 8 MB.`);
  }
  const form = new FormData();
  // React Native file part: native code reads `uri` and sets the part's filename + Content-Type.
  form.append('file', { uri: file.uri, name: file.name, type: file.mimeType } as any);
  // No Content-Type header: the networking layer adds multipart/form-data with its own boundary.
  const raw = await request('POST', `${normaliseBaseUrl(baseUrl)}/detect`,
    { 'X-API-Key': apiKey.trim(), Accept: 'application/json' }, form, onXhr);
  const json = parseJson(raw);
  if (raw.status === 200 && json?.status === 'ok') return json as DetectResult;
  throw httpError(raw, json);
}
