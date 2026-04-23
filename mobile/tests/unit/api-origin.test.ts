vi.mock('react-native', () => ({
  Platform: {
    OS: 'android',
    select: (values: Record<string, string>) => values.android ?? values.default,
  },
}));

vi.mock('expo-constants', () => ({
  default: {
    expoConfig: null,
  },
}));

import {
  clearApiBaseUrlOverride,
  getApiBaseUrl,
  getApiFallbackBaseUrls,
  MANUAL_API_ORIGIN_ENABLED,
  normalizeApiBaseUrl,
  setApiBaseUrlOverride,
} from '../../src/config/env';
import { requestJson } from '../../src/lib/http';

describe('API origin configuration', () => {
  beforeEach(() => {
    clearApiBaseUrlOverride();
    vi.restoreAllMocks();
  });

  it('normalizes API origins for stable request building', () => {
    expect(normalizeApiBaseUrl(' https://api.example.com/rps/// ')).toBe('https://api.example.com/rps');
    expect(() => normalizeApiBaseUrl('api.example.com')).toThrow('http:// or https://');
  });

  it('uses the runtime API origin override for relative API calls', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ session_id: 'abc', max_rounds: 5 }),
    }));
    vi.stubGlobal('fetch', fetchMock);

    setApiBaseUrlOverride('https://backend.example.com/game-api/');

    const data = await requestJson('/sessions', { method: 'POST', body: {} });

    expect(getApiBaseUrl()).toBe('https://backend.example.com/game-api');
    expect(data).toEqual({ session_id: 'abc', max_rounds: 5 });
    expect(fetchMock).toHaveBeenCalledWith(
      'https://backend.example.com/game-api/sessions',
      expect.objectContaining({
        method: 'POST',
      }),
    );
  });

  it('allows manual API origin overrides when no production API base is configured', () => {
    expect(MANUAL_API_ORIGIN_ENABLED).toBe(true);
    expect(getApiFallbackBaseUrls()).toEqual(['http://10.0.2.2:8000']);
  });

  it('locks the API origin to the configured production backend when provided', async () => {
    const previousBaseUrl = process.env.EXPO_PUBLIC_API_BASE_URL;

    vi.resetModules();
    process.env.EXPO_PUBLIC_API_BASE_URL = 'https://render.example.com/api/';

    const envModule = await import('../../src/config/env');

    expect(envModule.DEFAULT_API_BASE_URL).toBe('https://render.example.com/api');
    expect(envModule.MANUAL_API_ORIGIN_ENABLED).toBe(false);
    expect(envModule.getApiFallbackBaseUrls()).toEqual(['https://render.example.com/api']);

    if (previousBaseUrl === undefined) {
      delete process.env.EXPO_PUBLIC_API_BASE_URL;
    } else {
      process.env.EXPO_PUBLIC_API_BASE_URL = previousBaseUrl;
    }
  });
});
