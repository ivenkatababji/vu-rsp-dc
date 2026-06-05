import type { InputMode } from '../features/game/schemas';
import { Platform } from 'react-native';

const AUTH_HEADER_KEY = 'rps.authHeader';
const USERNAME_KEY = 'rps.username';
const SESSION_ID_KEY = 'rps.sessionId';
const SESSION_MAX_ROUNDS_KEY = 'rps.sessionMaxRounds';
const INPUT_MODE_KEY = 'rps.inputMode';
const API_BASE_URL_KEY = 'rps.apiBaseUrl';

type SecureStoreModule = {
  getItemAsync: (key: string) => Promise<string | null>;
  setItemAsync: (key: string, value: string) => Promise<void>;
  deleteItemAsync: (key: string) => Promise<void>;
};

let secureStore: SecureStoreModule | null = null;

try {
  if (Platform.OS !== 'web') {
    secureStore = require('expo-secure-store') as SecureStoreModule;
  }
} catch {
  secureStore = null;
}

const memoryStore = new Map<string, string>();

async function getItem(key: string): Promise<string | null> {
  if (secureStore) {
    return secureStore.getItemAsync(key);
  }

  if (Platform.OS === 'web' && typeof globalThis.localStorage !== 'undefined') {
    return globalThis.localStorage.getItem(key);
  }

  return memoryStore.get(key) ?? null;
}

async function setItem(key: string, value: string): Promise<void> {
  if (secureStore) {
    await secureStore.setItemAsync(key, value);
    return;
  }

  if (Platform.OS === 'web' && typeof globalThis.localStorage !== 'undefined') {
    globalThis.localStorage.setItem(key, value);
    return;
  }

  memoryStore.set(key, value);
}

async function deleteItem(key: string): Promise<void> {
  if (secureStore) {
    await secureStore.deleteItemAsync(key);
    return;
  }

  if (Platform.OS === 'web' && typeof globalThis.localStorage !== 'undefined') {
    globalThis.localStorage.removeItem(key);
    return;
  }

  memoryStore.delete(key);
}

export async function saveAuthSession(authHeader: string, username: string): Promise<void> {
  await Promise.all([setItem(AUTH_HEADER_KEY, authHeader), setItem(USERNAME_KEY, username)]);
}

export async function loadAuthSession(): Promise<{ authHeader: string; username: string } | null> {
  const [authHeader, username] = await Promise.all([getItem(AUTH_HEADER_KEY), getItem(USERNAME_KEY)]);
  if (!authHeader || !username) {
    return null;
  }

  return { authHeader, username };
}

export async function clearAuthSession(): Promise<void> {
  await Promise.all([
    deleteItem(AUTH_HEADER_KEY),
    deleteItem(USERNAME_KEY),
    deleteItem(SESSION_ID_KEY),
    deleteItem(SESSION_MAX_ROUNDS_KEY),
  ]);
}

export async function clearSavedSession(): Promise<void> {
  await Promise.all([deleteItem(SESSION_ID_KEY), deleteItem(SESSION_MAX_ROUNDS_KEY)]);
}

export async function saveSession(sessionId: string, maxRounds: number): Promise<void> {
  await Promise.all([setItem(SESSION_ID_KEY, sessionId), setItem(SESSION_MAX_ROUNDS_KEY, String(maxRounds))]);
}

export async function loadSession(): Promise<{ sessionId: string; maxRounds: number } | null> {
  const [sessionId, maxRoundsRaw] = await Promise.all([
    getItem(SESSION_ID_KEY),
    getItem(SESSION_MAX_ROUNDS_KEY),
  ]);
  const maxRounds = Number(maxRoundsRaw);
  if (!sessionId || !Number.isFinite(maxRounds) || maxRounds <= 0) {
    return null;
  }

  return { sessionId, maxRounds };
}

export async function saveInputMode(mode: InputMode): Promise<void> {
  await setItem(INPUT_MODE_KEY, mode);
}

export async function loadInputMode(): Promise<InputMode | null> {
  const value = await getItem(INPUT_MODE_KEY);
  if (value === 'buttons' || value === 'vision' || value === 'audio') {
    return value;
  }

  return null;
}

export async function saveApiBaseUrl(baseUrl: string): Promise<void> {
  await setItem(API_BASE_URL_KEY, baseUrl);
}

export async function loadApiBaseUrl(): Promise<string | null> {
  return getItem(API_BASE_URL_KEY);
}

export async function clearSavedApiBaseUrl(): Promise<void> {
  await deleteItem(API_BASE_URL_KEY);
}
