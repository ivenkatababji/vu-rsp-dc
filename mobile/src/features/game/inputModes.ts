import type { InputMode, Manifest } from './schemas';

type InputModeSupport = {
  availableModes: InputMode[];
  visionEnabled: boolean;
  audioEnabled: boolean;
  visionModelAvailable: boolean;
  audioSpeechAvailable: boolean;
};

type RuntimeCapabilitySupport = {
  speechAvailable?: boolean;
};

const ALL_CLIENT_MODES: InputMode[] = ['buttons', 'vision', 'audio'];

function uniqueModes(modes: InputMode[]): InputMode[] {
  return ALL_CLIENT_MODES.filter((mode) => modes.includes(mode));
}

export function deriveInputModeSupport(
  manifest?: Manifest,
  runtimeCapabilitySupport: RuntimeCapabilitySupport = {},
): InputModeSupport {
  const manifestModes: InputMode[] = manifest?.input_modes?.length ? manifest.input_modes : ['buttons'];
  const normalizedModes = uniqueModes(manifestModes);
  const modeSet = new Set<InputMode>(normalizedModes);
  modeSet.add('buttons');

  const visionModelAvailable = Boolean(
    modeSet.has('vision') && manifest?.vision?.available && manifest.vision.model_url,
  );
  const audioSpeechAvailable = Boolean(
    modeSet.has('audio') &&
      manifest?.audio?.browser_speech?.enabled !== false &&
      (runtimeCapabilitySupport.speechAvailable ?? true),
  );

  const availableModes: InputMode[] = ['buttons'];
  if (visionModelAvailable) {
    availableModes.push('vision');
  }
  if (audioSpeechAvailable) {
    availableModes.push('audio');
  }

  return {
    availableModes,
    visionEnabled: visionModelAvailable,
    audioEnabled: audioSpeechAvailable,
    visionModelAvailable,
    audioSpeechAvailable,
  };
}
