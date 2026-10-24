import { useEffect, useMemo, useState } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';

import { colors } from '../theme';

// Circular dial from the reference: fine radial ticks, three orange ticks near the top-right,
// a centred waveform glyph and a caption. Pure Views (no SVG dependency). While `active` the
// orange tick group revolves; it is a motion cue only and encodes no progress or score.

const TICKS = 72;
const WAVE = [10, 18, 26, 18, 34, 46, 66, 86, 108, 86, 66, 46, 34, 18, 26, 18, 10];

type Props = {
  size: number;
  active?: boolean;
  title: string;
  subtitle?: string;
  tone?: 'neutral' | 'ok' | 'danger';
};

export default function Dial({ size, active = false, title, subtitle, tone = 'neutral' }: Props) {
  // useState initialisers: created once, never read as a ref during render (lint: react-hooks/refs)
  const [spin] = useState(() => new Animated.Value(0));
  const [pulse] = useState(() => new Animated.Value(1));

  useEffect(() => {
    if (!active) { spin.stopAnimation(); spin.setValue(0); pulse.stopAnimation(); pulse.setValue(1); return; }
    const a = Animated.loop(Animated.timing(spin, { toValue: 1, duration: 2600, easing: Easing.linear, useNativeDriver: true }));
    const b = Animated.loop(Animated.sequence([
      Animated.timing(pulse, { toValue: 0.86, duration: 700, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
      Animated.timing(pulse, { toValue: 1, duration: 700, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
    ]));
    a.start(); b.start();
    return () => { a.stop(); b.stop(); };
  }, [active, spin, pulse]);

  const radius = size / 2;
  const tickLen = Math.max(10, size * 0.045);
  const ticks = useMemo(() => Array.from({ length: TICKS }, (_, i) => (360 / TICKS) * i), []);
  const accentTicks = [14, 21, 28]; // degrees clockwise from 12 o'clock (top-right, as in the reference)
  const rotate = spin.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });
  const inkColor = tone === 'ok' ? colors.ok : tone === 'danger' ? colors.danger : colors.ink;

  return (
    <View style={[styles.wrap, { width: size, height: size }]} accessibilityRole="image" accessibilityLabel={`${title}${subtitle ? `, ${subtitle}` : ''}`}>
      <View style={[styles.ring, { width: size, height: size, borderRadius: radius }]} />
      {ticks.map((deg) => (
        <View key={deg} pointerEvents="none" style={[styles.tickHolder, { left: radius - 1, top: radius - tickLen / 2, height: tickLen },
          { transform: [{ rotate: `${deg}deg` }, { translateY: -(radius - tickLen / 2 - 8) }] }]}>
          <View style={[styles.tick, { height: tickLen }]} />
        </View>
      ))}
      <Animated.View pointerEvents="none" style={[StyleSheet.absoluteFill, { transform: [{ rotate }] }]}>
        {accentTicks.map((deg) => (
          <View key={deg} style={[styles.tickHolder, { left: radius - 1, top: radius - tickLen / 2, height: tickLen },
            { transform: [{ rotate: `${deg}deg` }, { translateY: -(radius - tickLen / 2 - 8) }] }]}>
            <View style={[styles.tick, styles.tickAccent, { height: tickLen }]} />
          </View>
        ))}
      </Animated.View>
      <View style={styles.centre} pointerEvents="none">
        <Animated.View style={[styles.wave, { transform: [{ scaleY: pulse }] }]}>
          {WAVE.map((h, i) => (
            <View key={i} style={[styles.bar, { height: (h / 108) * size * 0.26, backgroundColor: inkColor }]} />
          ))}
        </Animated.View>
        <Text style={[styles.title, { color: inkColor }]} maxFontSizeMultiplier={1.3}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle} maxFontSizeMultiplier={1.3}>{subtitle}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { alignItems: 'center', justifyContent: 'center', alignSelf: 'center' },
  ring: { position: 'absolute', borderWidth: 1, borderColor: colors.hairline, backgroundColor: colors.surface },
  tickHolder: { position: 'absolute', width: 2, alignItems: 'center' },
  tick: { width: 1.5, borderRadius: 1, backgroundColor: colors.tick },
  tickAccent: { width: 2.5, backgroundColor: colors.accent },
  centre: { alignItems: 'center', paddingHorizontal: 24 },
  wave: { flexDirection: 'row', alignItems: 'center', gap: 4, height: 100, marginBottom: 18 },
  bar: { width: 4, borderRadius: 2 },
  title: { fontSize: 20, fontWeight: '600', textAlign: 'center' },
  subtitle: { fontSize: 15, color: colors.inkSoft, marginTop: 2, textAlign: 'center' },
});
