import { Buffer } from 'buffer';
import * as FileSystem from 'expo-file-system/legacy';
import { getApiBaseUrl } from '../../config/env';
import { requestBinary } from '../../lib/http';
import type { Manifest } from '../game/schemas';
import { buildVisionCacheKey, resolveModelUrl } from './modelIdentity';

type CachedVisionModel = {
  cacheKey: string;
  uri: string;
  bytes: ArrayBuffer;
};

type NativeCrypto = typeof import('crypto');

const CACHE_DIR = `${FileSystem.cacheDirectory ?? ''}rps-ml/vision/`;

function cacheFileUri(cacheKey: string): string {
  const safeName = cacheKey.replace(/[^a-zA-Z0-9._-]/g, '_');
  return `${CACHE_DIR}${safeName}.onnx`;
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const digest = await subtle.digest('SHA-256', bytes);
    return Buffer.from(digest).toString('hex');
  }

  try {
    const nodeCrypto = require('crypto') as NativeCrypto;
    return nodeCrypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex');
  } catch {
    throw new Error('SHA-256 verification is unavailable in this runtime');
  }
}

async function ensureCacheDir(): Promise<void> {
  if (!CACHE_DIR) {
    throw new Error('Model cache directory is unavailable');
  }

  const info = await FileSystem.getInfoAsync(CACHE_DIR);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(CACHE_DIR, { intermediates: true });
  }
}

async function readCachedBytes(uri: string): Promise<ArrayBuffer | null> {
  const info = await FileSystem.getInfoAsync(uri);
  if (!info.exists) {
    return null;
  }

  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: 'base64',
  });
  const buffer = Buffer.from(base64, 'base64');
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

async function writeCachedBytes(uri: string, bytes: ArrayBuffer): Promise<void> {
  await FileSystem.writeAsStringAsync(uri, Buffer.from(bytes).toString('base64'), {
    encoding: 'base64',
  });
}

export async function getCachedVisionModel(
  manifest: Manifest,
  authHeader: string,
): Promise<CachedVisionModel> {
  const vision = manifest.vision;
  if (!vision?.available || !vision.model_url) {
    throw new Error('Vision model not available');
  }

  const cacheKey = buildVisionCacheKey(vision);
  const uri = cacheFileUri(cacheKey);
  await ensureCacheDir();

  const cached = await readCachedBytes(uri);
  if (cached) {
    return { cacheKey, uri, bytes: cached };
  }

  const bytes = await requestBinary(resolveModelUrl(getApiBaseUrl(), vision.model_url), {
    authHeader,
    absoluteUrl: true,
    timeoutMs: 30000,
  });

  if (vision.sha256) {
    const actual = await sha256Hex(bytes);
    if (actual !== vision.sha256) {
      throw new Error('Downloaded vision model failed SHA-256 verification');
    }
  }

  await writeCachedBytes(uri, bytes);
  return { cacheKey, uri, bytes };
}
