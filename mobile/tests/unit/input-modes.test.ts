import { deriveInputModeSupport } from '../../src/features/game/inputModes';
import type { Manifest } from '../../src/features/game/schemas';

describe('input mode support', () => {
  it('keeps only buttons enabled when server advertises buttons mode only', () => {
    const manifest: Manifest = {
      input_modes: ['buttons'],
      vision: {
        available: false,
        version: 'none',
        sha256: null,
        model_url: null,
        labels: ['rock', 'paper', 'scissors', 'none'],
      },
      audio: {
        browser_speech: {
          enabled: true,
          locale: 'en-US',
        },
      },
    };

    const support = deriveInputModeSupport(manifest, { speechAvailable: true });

    expect(support.availableModes).toEqual(['buttons']);
    expect(support.visionEnabled).toBe(false);
    expect(support.audioEnabled).toBe(false);
  });

  it('enables vision only when server mode and model availability are both true', () => {
    const manifest: Manifest = {
      input_modes: ['buttons', 'vision'],
      vision: {
        available: true,
        version: 'vision-v1',
        sha256: 'abc',
        model_url: '/me/ml/models/vision',
        labels: ['rock', 'paper', 'scissors'],
      },
    };

    const support = deriveInputModeSupport(manifest, { speechAvailable: true });

    expect(support.availableModes).toEqual(['buttons', 'vision']);
    expect(support.visionEnabled).toBe(true);
    expect(support.audioEnabled).toBe(false);
  });

  it('disables audio when runtime speech capability is unavailable', () => {
    const manifest: Manifest = {
      input_modes: ['buttons', 'audio'],
      audio: {
        browser_speech: {
          enabled: true,
          locale: 'en-US',
        },
      },
    };

    const support = deriveInputModeSupport(manifest, { speechAvailable: false });

    expect(support.availableModes).toEqual(['buttons']);
    expect(support.audioEnabled).toBe(false);
  });
});
