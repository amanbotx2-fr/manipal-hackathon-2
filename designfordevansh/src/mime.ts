// Pickers sometimes report no MIME type or application/octet-stream for audio
// (e.g. .m4a from Downloads). Fall back to the extension so the server and any
// proxy see a correct Content-Type. The server itself validates by extension.
const BY_EXT: Record<string, string> = {
  wav: 'audio/wav',
  m4a: 'audio/mp4',
  mp4: 'audio/mp4',
  aac: 'audio/aac',
  mp3: 'audio/mpeg',
  ogg: 'audio/ogg',
  opus: 'audio/ogg',
  flac: 'audio/flac',
  '3gp': 'audio/3gpp',
  amr: 'audio/amr',
  webm: 'audio/webm',
};

export const SUPPORTED_EXTENSIONS = Object.keys(BY_EXT);

export function extensionOf(name: string): string {
  const m = /\.([a-z0-9]+)$/i.exec(name);
  return m ? m[1].toLowerCase() : '';
}

export function resolveMimeType(name: string, reported?: string | null): string {
  if (reported && reported !== 'application/octet-stream' && reported.includes('/')) return reported;
  return BY_EXT[extensionOf(name)] ?? 'application/octet-stream';
}

const EXT_BY_MIME: Record<string, string> = {
  'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/wave': 'wav',
  'audio/mp4': 'm4a', 'audio/x-m4a': 'm4a', 'audio/m4a': 'm4a',
  'audio/aac': 'aac', 'audio/mpeg': 'mp3', 'audio/ogg': 'ogg', 'audio/opus': 'opus',
  'audio/flac': 'flac', 'audio/3gpp': '3gp', 'audio/amr': 'amr', 'audio/webm': 'webm',
};

// Some document providers return display names without an extension; the
// server checks the extension, so add one from the MIME type when missing.
export function ensureExtension(name: string, mimeType: string): string {
  if (SUPPORTED_EXTENSIONS.includes(extensionOf(name))) return name;
  const ext = EXT_BY_MIME[mimeType];
  return ext ? `${name}.${ext}` : name;
}
