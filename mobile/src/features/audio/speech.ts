import { useEffect } from 'react';

type SpeechEventName = 'start' | 'end' | 'result' | 'error';

type SpeechListener = (event: unknown) => void;

type SpeechPermissions = {
  granted?: boolean;
};

type SpeechStartOptions = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
};

type SpeechModule = {
  requestPermissionsAsync?: () => Promise<SpeechPermissions>;
  start?: (options: SpeechStartOptions) => void;
  stop?: () => void;
};

type SpeechPackage = {
  ExpoSpeechRecognitionModule?: SpeechModule;
  useSpeechRecognitionEvent?: (eventName: SpeechEventName, listener: SpeechListener) => void;
};

let speechPackage: SpeechPackage | null = null;

try {
  speechPackage = require('expo-speech-recognition') as SpeechPackage;
} catch {
  speechPackage = null;
}

const speechModule = speechPackage?.ExpoSpeechRecognitionModule;
const speechEventHook = speechPackage?.useSpeechRecognitionEvent;

type WebSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  onstart: ((event: unknown) => void) | null;
  onend: ((event: unknown) => void) | null;
  onresult: ((event: unknown) => void) | null;
  onerror: ((event: unknown) => void) | null;
  start: () => void;
  stop: () => void;
};

type WebSpeechConstructor = new () => WebSpeechRecognition;

function getWebSpeechConstructor(): WebSpeechConstructor | null {
  const scope = globalThis as typeof globalThis & {
    SpeechRecognition?: WebSpeechConstructor;
    webkitSpeechRecognition?: WebSpeechConstructor;
  };

  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition ?? null;
}

const webListeners = new Map<SpeechEventName, Set<SpeechListener>>();
let activeWebRecognition: WebSpeechRecognition | null = null;

function emitWebSpeechEvent(eventName: SpeechEventName, event: unknown): void {
  webListeners.get(eventName)?.forEach((listener) => listener(event));
}

function addWebSpeechListener(eventName: SpeechEventName, listener: SpeechListener): () => void {
  const listeners = webListeners.get(eventName) ?? new Set<SpeechListener>();
  listeners.add(listener);
  webListeners.set(eventName, listeners);

  return () => {
    listeners.delete(listener);
    if (!listeners.size) {
      webListeners.delete(eventName);
    }
  };
}

export const isSpeechRecognitionAvailable =
  (
    typeof speechEventHook === 'function' &&
    Boolean(speechModule?.requestPermissionsAsync && speechModule?.start && speechModule?.stop)
  ) ||
  Boolean(getWebSpeechConstructor());

const isExpoSpeechAvailable =
  typeof speechEventHook === 'function' &&
  Boolean(speechModule?.requestPermissionsAsync && speechModule?.start && speechModule?.stop);

export function useSafeSpeechRecognitionEvent(
  eventName: SpeechEventName,
  listener: SpeechListener,
): void {
  if (isExpoSpeechAvailable && typeof speechEventHook === 'function') {
    speechEventHook(eventName, listener);
    return;
  }

  useEffect(() => addWebSpeechListener(eventName, listener), [eventName, listener]);
}

export async function requestSpeechPermissions(): Promise<boolean> {
  if (getWebSpeechConstructor()) {
    return true;
  }

  if (!speechModule?.requestPermissionsAsync) {
    return false;
  }

  try {
    const result = await speechModule.requestPermissionsAsync();
    return Boolean(result?.granted);
  } catch {
    return false;
  }
}

export function startSpeechRecognition(options: SpeechStartOptions): boolean {
  const WebSpeech = getWebSpeechConstructor();
  if (WebSpeech) {
    const recognition = new WebSpeech();
    recognition.lang = options.lang;
    recognition.interimResults = options.interimResults;
    recognition.continuous = options.continuous;
    recognition.maxAlternatives = options.maxAlternatives;
    recognition.onstart = (event) => emitWebSpeechEvent('start', event);
    recognition.onend = (event) => emitWebSpeechEvent('end', event);
    recognition.onresult = (event) => emitWebSpeechEvent('result', event);
    recognition.onerror = (event) => emitWebSpeechEvent('error', event);
    activeWebRecognition = recognition;
    recognition.start();
    return true;
  }

  if (!speechModule?.start) {
    return false;
  }

  speechModule.start(options);
  return true;
}

export function stopSpeechRecognition(): void {
  if (activeWebRecognition) {
    activeWebRecognition.stop();
    activeWebRecognition = null;
    return;
  }

  speechModule?.stop?.();
}
