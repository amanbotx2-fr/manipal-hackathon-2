import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useState } from 'react';
import { BackHandler, StyleSheet, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import type { Analysis, PickedFile } from './src/analysis';
import DiagnosticsScreen from './src/screens/DiagnosticsScreen';
import HomeScreen from './src/screens/HomeScreen';
import ResultScreen from './src/screens/ResultScreen';
import { colors } from './src/theme';

// Screens: home (landing / selected / analysing), result (dedicated, compact), dev (diagnostics).
// Home and Dev stay mounted so their state survives navigation; the detector is a native singleton
// (one model, one session) shared by all screens.

type Screen = 'home' | 'result' | 'dev';
type Outcome = { kind: 'analysis'; analysis: Analysis } | { kind: 'error'; title: string; detail: string; file: PickedFile };

export default function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [devFrom, setDevFrom] = useState<Screen>('home');
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [lastAnalysis, setLastAnalysis] = useState<Analysis | null>(null);
  const [resetToken, setResetToken] = useState(0);
  const [retryToken, setRetryToken] = useState(0);

  const goHome = useCallback(() => setScreen('home'), []);
  const openDev = useCallback((from: Screen) => { setDevFrom(from); setScreen('dev'); }, []);
  const another = useCallback(() => { setResetToken((n) => n + 1); setScreen('home'); }, []);

  useEffect(() => {
    if (screen === 'home') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      setScreen(screen === 'dev' ? devFrom : 'home');
      return true;
    });
    return () => sub.remove();
  }, [screen, devFrom]);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={[styles.safe, { backgroundColor: screen === 'dev' ? '#F4F5F7' : colors.bg }]} edges={['top', 'bottom']}>
        <StatusBar style="dark" />
        <View style={[styles.fill, screen !== 'home' && styles.hidden]}>
          <HomeScreen
            onOpenDev={() => openDev('home')}
            onResult={(a) => { setLastAnalysis(a); setOutcome({ kind: 'analysis', analysis: a }); setScreen('result'); }}
            onError={(title, detail, file) => { setOutcome({ kind: 'error', title, detail, file }); setScreen('result'); }}
            resetToken={resetToken}
            retryToken={retryToken}
          />
        </View>
        {screen === 'result' && outcome && (
          <View style={styles.fill}>
            {outcome.kind === 'analysis'
              ? <ResultScreen kind="analysis" analysis={outcome.analysis} onBack={goHome} onAnother={another} onOpenDev={() => openDev('result')} />
              : <ResultScreen kind="error" title={outcome.title} detail={outcome.detail} file={outcome.file} onBack={goHome} onAnother={another}
                              onRetry={() => { setScreen('home'); setRetryToken((n) => n + 1); }} />}
          </View>
        )}
        <View style={[styles.fill, screen !== 'dev' && styles.hidden]}>
          <DiagnosticsScreen onBack={() => setScreen(devFrom)} active={screen === 'dev'} lastAnalysis={lastAnalysis} />
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  fill: { flex: 1 },
  hidden: { display: 'none' },
});
