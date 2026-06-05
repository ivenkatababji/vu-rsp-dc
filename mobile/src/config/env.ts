import { Platform } from 'react-native';
import Constants from 'expo-constants';

function resolveExpoHostIp(): string | null {
  const hostUri =
    Constants.expoConfig?.hostUri ??
    (Constants as any).expoGoConfig?.debuggerHost ??
    (Constants as any).manifest2?.extra?.expoGo?.debuggerHost;

  if (!hostUri || typeof hostUri !== 'string') {
    return null;
  }

  const [host] = hostUri.split(':');
  return host || null;
}

const expoHostIp = resolveExpoHostIp();
const configuredApiBaseUrl = (process.env.EXPO_PUBLIC_API_BASE_URL ?? '').trim();
const isDevRuntime = Boolean((globalThis as { __DEV__?: boolean }).__DEV__);

const defaultBaseUrl = Platform.select({
  ios: 'http://localhost:8000',
  android: expoHostIp ? `http://${expoHostIp}:8000` : 'http://10.0.2.2:8000',
  web: 'http://127.0.0.1:8000',
  default: 'http://localhost:8000',
});

export function normalizeApiBaseUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error('API origin is required');
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error('API origin must include http:// or https://');
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('API origin must use http:// or https://');
  }

  const path = url.pathname.replace(/\/+$/, '');
  return `${url.origin}${path === '/' ? '' : path}`;
}

export const DEFAULT_API_BASE_URL = normalizeApiBaseUrl(
  configuredApiBaseUrl || defaultBaseUrl || 'http://localhost:8000',
);

export const MANUAL_API_ORIGIN_ENABLED = isDevRuntime || !configuredApiBaseUrl;

let apiBaseUrlOverride: string | null = null;

export function getApiBaseUrl(): string {
  return apiBaseUrlOverride ?? DEFAULT_API_BASE_URL;
}

export function setApiBaseUrlOverride(value: string): string {
  apiBaseUrlOverride = normalizeApiBaseUrl(value);
  return apiBaseUrlOverride;
}

export function clearApiBaseUrlOverride(): void {
  apiBaseUrlOverride = null;
}

export const API_BASE_URL = DEFAULT_API_BASE_URL;

export function getApiFallbackBaseUrls(): string[] {
  const baseUrl = getApiBaseUrl();
  if (!MANUAL_API_ORIGIN_ENABLED) {
    return [baseUrl];
  }
  return Platform.OS === 'web'
    ? Array.from(new Set([baseUrl, 'http://127.0.0.1:8000', 'http://localhost:8000']))
    : [baseUrl];
}
