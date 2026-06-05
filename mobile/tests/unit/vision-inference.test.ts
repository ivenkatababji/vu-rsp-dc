import {
  __resetVisionInferenceCacheForTests,
  inferMoveFromImage,
} from '../../src/features/vision/inference';
import type { Manifest } from '../../src/features/game/schemas';
import { getCachedVisionModel } from '../../src/features/vision/modelCache';

const ortMocks = vi.hoisted(() => {
  const run = vi.fn();
  const create = vi.fn();

  class Tensor {
    type: string;
    data: Float32Array;
    dims: number[];

    constructor(type: string, data: Float32Array, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }

  return { run, create, Tensor };
});

const jpegDecodeMock = vi.hoisted(() => vi.fn());

vi.mock('../../src/features/vision/modelCache', () => ({
  getCachedVisionModel: vi.fn(async () => ({
    cacheKey: 'vision:v1:abc',
    uri: 'file:///cache/model.onnx',
    bytes: new Uint8Array([1, 2, 3, 4]).buffer,
  })),
}));

vi.mock('expo-image-manipulator', () => ({
  manipulateAsync: vi.fn(async () => ({
    base64: 'ZmFrZS1pbWFnZS1ieXRlcw==',
  })),
  SaveFormat: {
    JPEG: 'jpeg',
  },
}));

vi.mock('jpeg-js', () => ({
  decode: jpegDecodeMock,
}));

vi.mock('onnxruntime-react-native', () => ({
  InferenceSession: {
    create: ortMocks.create,
  },
  Tensor: ortMocks.Tensor,
}));

const manifest: Manifest = {
  input_modes: ['buttons', 'vision'],
  vision: {
    available: true,
    version: 'v1',
    sha256: 'abc',
    model_url: '/me/ml/models/vision',
    labels: ['rock', 'paper', 'scissors', 'none'],
    input: {
      name: 'input',
      width: 224,
      height: 224,
      layout: 'NCHW',
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
    output: {
      name: 'logits',
    },
  },
};

describe('vision inference adapter', () => {
  beforeEach(() => {
    __resetVisionInferenceCacheForTests();

    vi.mocked(getCachedVisionModel).mockResolvedValue({
      cacheKey: 'vision:v1:abc',
      uri: 'file:///cache/model.onnx',
      bytes: new Uint8Array([1, 2, 3, 4]).buffer,
    });

    jpegDecodeMock.mockReturnValue({
      width: 224,
      height: 224,
      data: new Uint8Array(224 * 224 * 4).fill(127),
    });

    ortMocks.create.mockResolvedValue({
      run: ortMocks.run,
    });
    ortMocks.run.mockResolvedValue({
      logits: {
        data: new Float32Array([0.1, 1.8, 0.2, -0.1]),
      },
    });

    delete (globalThis as any).__DEV__;
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('runs ONNX inference from manifest metadata and cached model bytes', async () => {
    const result = await inferMoveFromImage('file:///tmp/frame.jpg', manifest, 'Basic abc');

    expect(result.move).toBe('paper');
    expect(result.source).toBe('onnx');
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.cacheKey).toBe('vision:v1:abc');
    expect(ortMocks.create).toHaveBeenCalledTimes(1);
    expect(ortMocks.run).toHaveBeenCalledTimes(1);
  });

  it('throws when ONNX inference fails and debug fallback is disabled', async () => {
    ortMocks.create.mockRejectedValue(new Error('No ONNX runtime'));

    await expect(inferMoveFromImage('file:///tmp/frame.jpg', manifest, 'Basic abc')).rejects.toThrow(
      'Vision ONNX inference failed',
    );
  });

  it('allows deterministic fallback only in debug mode', async () => {
    ortMocks.create.mockRejectedValue(new Error('No ONNX runtime'));
    (globalThis as any).__DEV__ = true;

    const result = await inferMoveFromImage('file:///tmp/frame.jpg', manifest, 'Basic abc');

    expect(result.source).toBe('image-hash');
    expect(['rock', 'paper', 'scissors', 'none']).toContain(result.move);
    expect(result.cacheKey).toBe('vision:v1:abc');
  });
});
