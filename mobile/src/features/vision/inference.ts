import * as ImageManipulator from 'expo-image-manipulator';
import { Buffer } from 'buffer';
import { decode as decodeJpeg } from 'jpeg-js';
import type { Manifest } from '../game/schemas';
import { getCachedVisionModel } from './modelCache';

export type VisionMove = 'rock' | 'paper' | 'scissors' | 'none';

export type VisionPrediction = {
  move: VisionMove;
  confidence: number;
  source: 'onnx' | 'image-hash';
  cacheKey?: string;
};

const DEFAULT_LABELS: VisionMove[] = ['rock', 'paper', 'scissors'];

type VisionInputLayout = 'NCHW' | 'NHWC';

type OnnxTensor = {
  data: Float32Array | number[];
};

type OnnxSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OnnxTensor>>;
};

type OnnxTensorCtor = new (type: string, data: Float32Array, dims: number[]) => unknown;

type OnnxNamespace = {
  Tensor: OnnxTensorCtor;
  InferenceSession: {
    create: (model: unknown, options?: unknown) => Promise<OnnxSession>;
  };
  env?: {
    wasm?: {
      wasmPaths?: string;
    };
  };
};

let onnxRuntime: OnnxNamespace | null | undefined;
let onnxRuntimePromise: Promise<OnnxNamespace | null> | null = null;
let cachedOnnxSession: { cacheKey: string; session: OnnxSession } | null = null;

async function getOnnxRuntime(): Promise<OnnxNamespace | null> {
  if (onnxRuntime !== undefined) {
    return onnxRuntime;
  }

  if (onnxRuntimePromise) {
    return onnxRuntimePromise;
  }

  onnxRuntimePromise = (async () => {
    try {
      const nativeRuntime = await import('onnxruntime-react-native');
      onnxRuntime = nativeRuntime as unknown as OnnxNamespace;
      return onnxRuntime;
    } catch {
      onnxRuntime = null;
    }

    return onnxRuntime;
  })();

  return onnxRuntimePromise;
}

function normalizeMoveLabel(label: string): VisionMove {
  const normalized = label.trim().toLowerCase();
  if (normalized === 'rock') {
    return 'rock';
  }
  if (normalized === 'paper') {
    return 'paper';
  }
  if (normalized === 'scissors' || normalized === 'scissor') {
    return 'scissors';
  }
  if (normalized === 'none') {
    return 'none';
  }
  return 'none';
}

function hashBase64(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function resolveVisionLayout(manifest: Manifest): VisionInputLayout {
  const raw = manifest.vision?.input?.layout?.toUpperCase();
  if (raw === 'NHWC') {
    return 'NHWC';
  }
  return 'NCHW';
}

function resolveMeanStd(manifest: Manifest): { mean: [number, number, number]; std: [number, number, number] } {
  const defaultMean: [number, number, number] = [0.485, 0.456, 0.406];
  const defaultStd: [number, number, number] = [0.229, 0.224, 0.225];

  const rawMean = manifest.vision?.input?.mean;
  const rawStd = manifest.vision?.input?.std;
  if (rawMean?.length === 3 && rawStd?.length === 3) {
    return {
      mean: [rawMean[0], rawMean[1], rawMean[2]],
      std: [rawStd[0], rawStd[1], rawStd[2]],
    };
  }

  return {
    mean: defaultMean,
    std: defaultStd,
  };
}

function buildInputTensor(
  rgbaData: Uint8Array,
  width: number,
  height: number,
  layout: VisionInputLayout,
  mean: [number, number, number],
  std: [number, number, number],
): { data: Float32Array; dims: number[] } {
  if (layout === 'NHWC') {
    const data = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const i = (y * width + x) * 4;
        const out = (y * width + x) * 3;
        data[out] = (rgbaData[i] / 255 - mean[0]) / std[0];
        data[out + 1] = (rgbaData[i + 1] / 255 - mean[1]) / std[1];
        data[out + 2] = (rgbaData[i + 2] / 255 - mean[2]) / std[2];
      }
    }

    return {
      data,
      dims: [1, height, width, 3],
    };
  }

  const channelSize = width * height;
  const data = new Float32Array(channelSize * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const idx = y * width + x;
      data[idx] = (rgbaData[i] / 255 - mean[0]) / std[0];
      data[channelSize + idx] = (rgbaData[i + 1] / 255 - mean[1]) / std[1];
      data[channelSize * 2 + idx] = (rgbaData[i + 2] / 255 - mean[2]) / std[2];
    }
  }

  return {
    data,
    dims: [1, 3, height, width],
  };
}

function argmax(values: number[]): number {
  let index = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[index]) {
      index = i;
    }
  }
  return index;
}

function softmaxConfidence(values: number[], bestIndex: number): number {
  if (!values.length) {
    return 0;
  }

  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  if (sum <= 0) {
    return 0;
  }
  return exps[bestIndex] / sum;
}

function shouldAllowDebugFallback(): boolean {
  const enabledByEnv = process.env.EXPO_PUBLIC_ALLOW_VISION_HASH_FALLBACK === '1';
  const enabledByDevFlag = Boolean((globalThis as { __DEV__?: boolean }).__DEV__);
  return enabledByEnv || enabledByDevFlag;
}

async function getOrCreateOnnxSession(
  manifest: Manifest,
  authHeader: string,
): Promise<{ cacheKey: string; session: OnnxSession }> {
  const cachedModel = await getCachedVisionModel(manifest, authHeader);
  if (cachedOnnxSession?.cacheKey === cachedModel.cacheKey) {
    return {
      cacheKey: cachedModel.cacheKey,
      session: cachedOnnxSession.session,
    };
  }

  const ort = await getOnnxRuntime();
  if (!ort) {
    throw new Error('ONNX runtime is unavailable');
  }

  const modelBytes = new Uint8Array(cachedModel.bytes);
  let session: OnnxSession;
  try {
    session = await ort.InferenceSession.create(modelBytes);
  } catch {
    session = await ort.InferenceSession.create(cachedModel.uri);
  }

  cachedOnnxSession = {
    cacheKey: cachedModel.cacheKey,
    session,
  };

  return {
    cacheKey: cachedModel.cacheKey,
    session,
  };
}

async function runOnnxVisionInference(
  imageUri: string,
  manifest: Manifest,
  authHeader: string,
): Promise<VisionPrediction> {
  const visionInput = manifest.vision?.input;
  const width = visionInput?.width ?? 224;
  const height = visionInput?.height ?? 224;
  const inputName = visionInput?.name ?? 'input';
  const outputName = (manifest.vision?.output?.name as string | undefined) ?? 'logits';

  const { cacheKey, session } = await getOrCreateOnnxSession(manifest, authHeader);

  const resized = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width, height } }],
    {
      base64: true,
      compress: 0.95,
      format: ImageManipulator.SaveFormat.JPEG,
    },
  );

  if (!resized.base64) {
    throw new Error('Image preprocessing did not return base64 pixels');
  }

  const decoded = decodeJpeg(Buffer.from(resized.base64, 'base64'), { useTArray: true });
  if (!decoded?.data || !decoded.width || !decoded.height) {
    throw new Error('Unable to decode processed image');
  }

  const layout = resolveVisionLayout(manifest);
  const { mean, std } = resolveMeanStd(manifest);
  const { data, dims } = buildInputTensor(decoded.data, decoded.width, decoded.height, layout, mean, std);

  const ort = await getOnnxRuntime();
  if (!ort) {
    throw new Error('ONNX runtime is unavailable');
  }

  const tensor = new ort.Tensor('float32', data, dims);
  const outputs = await session.run({ [inputName]: tensor });
  const outputTensor = outputs[outputName] ?? outputs[Object.keys(outputs)[0]];
  if (!outputTensor?.data) {
    throw new Error('ONNX model returned empty output tensor');
  }

  const logits = Array.from(outputTensor.data as Float32Array | number[]);
  if (!logits.length) {
    throw new Error('ONNX model returned no logits');
  }

  const bestIndex = argmax(logits);
  const labels = manifest.vision?.labels?.length ? manifest.vision.labels : DEFAULT_LABELS;
  const move = normalizeMoveLabel(labels[bestIndex] ?? 'none');
  const confidence = softmaxConfidence(logits, bestIndex);

  return {
    move,
    confidence: Number(confidence.toFixed(2)),
    source: 'onnx',
    cacheKey,
  };
}

export async function inferMoveFromImage(
  imageUri: string,
  manifest: Manifest,
  authHeader: string,
  _fileName?: string,
): Promise<VisionPrediction> {
  try {
    return await runOnnxVisionInference(imageUri, manifest, authHeader);
  } catch {
    if (!shouldAllowDebugFallback()) {
      throw new Error('Vision ONNX inference failed');
    }

    const cachedModel = await getCachedVisionModel(manifest, authHeader);
    const output = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: manifest.vision?.input?.width ?? 224, height: manifest.vision?.input?.height ?? 224 } }],
      {
        base64: true,
        compress: 0.9,
        format: ImageManipulator.SaveFormat.JPEG,
      },
    );

    const labels = manifest.vision?.labels ?? DEFAULT_LABELS;
    const candidateLabels = labels.length ? labels : DEFAULT_LABELS;
    const normalizedLabels = candidateLabels
      .map((label) => normalizeMoveLabel(label))
      .filter((label) => label === 'rock' || label === 'paper' || label === 'scissors' || label === 'none');

    const effectiveLabels = normalizedLabels.length ? normalizedLabels : DEFAULT_LABELS;
    const hash = hashBase64(output.base64 ?? imageUri);
    const selectedIndex = hash % effectiveLabels.length;
    const confidence = 0.55 + ((hash % 40) / 100);

    return {
      move: effectiveLabels[selectedIndex],
      confidence: Number(confidence.toFixed(2)),
      source: 'image-hash',
      cacheKey: cachedModel.cacheKey,
    };
  }
}

export function __resetVisionInferenceCacheForTests(): void {
  cachedOnnxSession = null;
  onnxRuntime = undefined;
  onnxRuntimePromise = null;
}
