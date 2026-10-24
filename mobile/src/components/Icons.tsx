import { StyleSheet, Text, View } from 'react-native';

import { colors } from '../theme';

// Small vector-ish icons built from Views/Text so no icon or SVG package is needed.

export function LockIcon({ color = colors.inkSoft }: { color?: string }) {
  return (
    <View style={icon.lockWrap}>
      <View style={[icon.lockShackle, { borderColor: color }]} />
      <View style={[icon.lockBody, { borderColor: color }]} />
    </View>
  );
}

export function FileIcon({ color = colors.ink }: { color?: string }) {
  return (
    <View style={[icon.file, { borderColor: color }]}>
      <View style={[icon.fileFold, { borderColor: color, backgroundColor: colors.bg }]} />
    </View>
  );
}

export function UploadIcon({ color = colors.accentInk }: { color?: string }) {
  return (
    <View style={icon.upWrap}>
      <View style={[icon.upArrowHead, { borderColor: color }]} />
      <View style={[icon.upStem, { backgroundColor: color }]} />
      <View style={[icon.upTray, { borderColor: color }]} />
    </View>
  );
}

export function ArrowRight({ color = colors.accentInk }: { color?: string }) {
  return <Text style={[icon.arrow, { color }]} maxFontSizeMultiplier={1.2}>→</Text>;
}

export function Sunburst({ size = 40 }: { size?: number }) {
  const n = 24;
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }} accessible={false}>
      {Array.from({ length: n }, (_, i) => (
        <View key={i} style={[icon.ray, { left: size / 2 - 0.75, top: size / 2 - size * 0.14, height: size * 0.28,
          backgroundColor: i >= 20 ? colors.accent : colors.ink,
          transform: [{ rotate: `${(360 / n) * i}deg` }, { translateY: -size * 0.32 }] }]} />
      ))}
    </View>
  );
}

const icon = StyleSheet.create({
  lockWrap: { width: 18, height: 22, alignItems: 'center' },
  lockShackle: { width: 11, height: 10, borderWidth: 1.6, borderBottomWidth: 0, borderTopLeftRadius: 6, borderTopRightRadius: 6, marginBottom: -1 },
  lockBody: { width: 16, height: 12, borderWidth: 1.6, borderRadius: 3 },
  file: { width: 18, height: 22, borderWidth: 1.6, borderRadius: 3, overflow: 'hidden' },
  fileFold: { position: 'absolute', right: -6, top: -6, width: 12, height: 12, borderWidth: 1.6, transform: [{ rotate: '45deg' }] },
  upWrap: { width: 20, height: 22, alignItems: 'center' },
  upArrowHead: { width: 9, height: 9, borderLeftWidth: 1.8, borderTopWidth: 1.8, transform: [{ rotate: '45deg' }], marginTop: 1 },
  upStem: { width: 1.8, height: 11, marginTop: -6 },
  upTray: { width: 18, height: 7, borderWidth: 1.8, borderTopWidth: 0, borderBottomLeftRadius: 3, borderBottomRightRadius: 3, marginTop: 1 },
  arrow: { fontSize: 22, lineHeight: 24, fontWeight: '600' },
  ray: { position: 'absolute', width: 1.5, borderRadius: 1 },
});
