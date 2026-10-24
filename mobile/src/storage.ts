// Settings persistence. Both values live in platform secure storage
// (Android Keystore-encrypted prefs / iOS Keychain). Never log the key.
import * as SecureStore from 'expo-secure-store';

const URL_KEY = 'vcd.backendUrl';
const API_KEY = 'vcd.apiKey';

export type Settings = { baseUrl: string; apiKey: string };

export async function loadSettings(): Promise<Settings> {
  const [baseUrl, apiKey] = await Promise.all([
    SecureStore.getItemAsync(URL_KEY),
    SecureStore.getItemAsync(API_KEY),
  ]);
  return { baseUrl: baseUrl ?? '', apiKey: apiKey ?? '' };
}

export async function saveSettings(s: Settings): Promise<void> {
  await SecureStore.setItemAsync(URL_KEY, s.baseUrl.trim());
  if (s.apiKey.trim()) await SecureStore.setItemAsync(API_KEY, s.apiKey.trim());
  else await SecureStore.deleteItemAsync(API_KEY);
}
