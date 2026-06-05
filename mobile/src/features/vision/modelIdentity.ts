type VisionModelIdentity = {
  version?: string | null;
  sha256?: string | null;
};

export function buildVisionCacheKey(model: VisionModelIdentity): string {
  return `vision:${model.version ?? 'none'}:${model.sha256 ?? 'nohash'}`;
}

export function resolveModelUrl(apiBaseUrl: string, modelUrl: string): string {
  if (/^https?:\/\//i.test(modelUrl)) {
    return modelUrl;
  }

  const base = apiBaseUrl.replace(/\/$/, '');
  const path = modelUrl.startsWith('/') ? modelUrl : `/${modelUrl}`;
  return `${base}${path}`;
}
