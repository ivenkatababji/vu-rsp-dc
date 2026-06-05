describe('speech runtime adapter', () => {
  const originalSpeechRecognition = (globalThis as any).SpeechRecognition;
  const originalWebkitSpeechRecognition = (globalThis as any).webkitSpeechRecognition;

  afterEach(() => {
    vi.resetModules();
    (globalThis as any).SpeechRecognition = originalSpeechRecognition;
    (globalThis as any).webkitSpeechRecognition = originalWebkitSpeechRecognition;
  });

  it('uses the browser Web Speech API when Expo speech recognition is unavailable', async () => {
    class FakeSpeechRecognition {
      lang = '';
      interimResults = true;
      continuous = true;
      maxAlternatives = 0;
      start = vi.fn();
      stop = vi.fn();
    }

    (globalThis as any).SpeechRecognition = FakeSpeechRecognition;
    (globalThis as any).webkitSpeechRecognition = undefined;

    const speech = await import('../../src/features/audio/speech');

    expect(speech.isSpeechRecognitionAvailable).toBe(true);
    expect(speech.startSpeechRecognition({
      lang: 'en-US',
      interimResults: false,
      continuous: false,
      maxAlternatives: 1,
    })).toBe(true);
  });
});
