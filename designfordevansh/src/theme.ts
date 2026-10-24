// Design tokens for the landing flow (matches the reference: warm grey, charcoal-green type,
// restrained orange accent). The diagnostics screen keeps its own utilitarian styling.
export const colors = {
  bg: '#F2F1EC',
  surface: '#F8F7F3',
  surfaceStrong: '#FFFFFF',
  hairline: '#DAD9D2',
  tick: '#C8C7BF',
  ink: '#1F2A24',
  inkSoft: '#5E6962',
  inkMute: '#8A938D',
  accent: '#F4592B',
  accentInk: '#FFFFFF',
  danger: '#B42318',
  dangerBg: '#FBEFEC',
  ok: '#1E6B3A',
};

export const space = { xs: 6, sm: 10, md: 16, lg: 24, xl: 32 };

export const type = {
  eyebrow: { fontSize: 12, letterSpacing: 2.4, color: colors.inkSoft, textTransform: 'uppercase' as const },
  headline: { fontSize: 36, lineHeight: 42, fontWeight: '700' as const, color: colors.ink, letterSpacing: -0.5 },
  lede: { fontSize: 17, lineHeight: 24, color: colors.inkSoft },
  body: { fontSize: 15, lineHeight: 21, color: colors.ink },
  small: { fontSize: 13, lineHeight: 18, color: colors.inkSoft },
  fine: { fontSize: 12, lineHeight: 17, color: colors.inkMute },
};
