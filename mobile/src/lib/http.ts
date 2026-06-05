import { getApiBaseUrl, getApiFallbackBaseUrls } from '../config/env';
import { Buffer } from 'buffer';

type RequestOptions = {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  authHeader?: string;
  body?: unknown;
  timeoutMs?: number;
  absoluteUrl?: boolean;
};

export class HttpError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string) {
    super(`${status}: ${body}`);
    this.name = 'HttpError';
    this.status = status;
    this.body = body;
  }
}

function buildUrl(path: string, absoluteUrl = false, baseUrl = getApiBaseUrl()): string {
  if (absoluteUrl || /^https?:\/\//i.test(path)) {
    return path;
  }

  return `${baseUrl}${path}`;
}

function buildCandidateUrls(path: string, absoluteUrl = false): string[] {
  if (absoluteUrl || /^https?:\/\//i.test(path)) {
    return [buildUrl(path, absoluteUrl)];
  }

  return getApiFallbackBaseUrls().map((baseUrl) => buildUrl(path, false, baseUrl));
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 15000): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function requestHeaders(authHeader?: string): Record<string, string> {
  return {
    Accept: 'application/json',
    ...(authHeader ? { Authorization: authHeader } : {}),
  };
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const urls = buildCandidateUrls(path, options.absoluteUrl);
  let lastError: unknown = null;

  for (const [index, url] of urls.entries()) {
    try {
      const response = await fetchWithTimeout(url, {
        method: options.method ?? 'GET',
        headers: {
          ...requestHeaders(options.authHeader),
          'Content-Type': 'application/json',
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
      }, options.timeoutMs);

      if (!response.ok) {
        const text = await response.text();
        const detail = text || response.statusText;
        const shouldTryFallback =
          index < urls.length - 1 &&
          (response.status === 404 || response.status === 405) &&
          /<!doctype html|<html|not found|cannot (get|post)/i.test(detail);

        if (shouldTryFallback) {
          lastError = new HttpError(response.status, detail);
          continue;
        }

        throw new HttpError(response.status, detail);
      }

      if (response.status === 204) {
        return undefined as T;
      }

      return (await response.json()) as T;
    } catch (error) {
      lastError = error;
      if (error instanceof HttpError || index === urls.length - 1) {
        throw error;
      }
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Unable to reach API');
}

export async function requestBinary(path: string, options: RequestOptions = {}): Promise<ArrayBuffer> {
  const urls = buildCandidateUrls(path, options.absoluteUrl);
  let lastError: unknown = null;

  for (const [index, url] of urls.entries()) {
    try {
      const response = await fetchWithTimeout(url, {
        method: options.method ?? 'GET',
        headers: requestHeaders(options.authHeader),
      }, options.timeoutMs);

      if (!response.ok) {
        const text = await response.text();
        const detail = text || response.statusText;
        const shouldTryFallback =
          index < urls.length - 1 &&
          (response.status === 404 || response.status === 405) &&
          /<!doctype html|<html|not found|cannot (get|post)/i.test(detail);

        if (shouldTryFallback) {
          lastError = new HttpError(response.status, detail);
          continue;
        }

        throw new HttpError(response.status, detail);
      }

      return response.arrayBuffer();
    } catch (error) {
      lastError = error;
      if (error instanceof HttpError || index === urls.length - 1) {
        throw error;
      }
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Unable to reach API');
}

export function toBasicAuthHeader(username: string, password: string): string {
  return `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`;
}
