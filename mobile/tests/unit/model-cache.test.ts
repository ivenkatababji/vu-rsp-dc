import { buildVisionCacheKey, resolveModelUrl } from '../../src/features/vision/modelIdentity';

describe('vision model cache helpers', () => {
  it('builds the same model cache identity as the web client', () => {
    expect(buildVisionCacheKey({ version: 'v1', sha256: 'abc123' })).toBe('vision:v1:abc123');
  });

  it('resolves manifest model paths against the configured API base', () => {
    expect(resolveModelUrl('http://localhost:8000/api', '/me/ml/models/vision')).toBe(
      'http://localhost:8000/api/me/ml/models/vision',
    );
  });
});
