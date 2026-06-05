import {
  createSessionResponseSchema,
  manifestSchema,
  playRoundResponseSchema,
  sessionStateSchema,
  statsSchema,
} from '../../src/features/game/schemas';

describe('API contract schemas', () => {
  it('parses session payload shape', () => {
    const parsed = createSessionResponseSchema.parse({
      session_id: 'abc-123',
      user_id: 'pilot',
      max_rounds: 5,
    });

    expect(parsed.session_id).toBe('abc-123');
    expect(parsed.max_rounds).toBe(5);
  });

  it('parses stats payload shape', () => {
    const parsed = statsSchema.parse({
      sessions_started: 4,
      matches_completed: 3,
      matches_won: 2,
      matches_lost: 1,
      matches_draw: 0,
      rounds_played: 11,
    });

    expect(parsed.rounds_played).toBe(11);
    expect(parsed.matches_won).toBe(2);
  });

  it('applies fallback modes when manifest input_modes is empty', () => {
    const parsed = manifestSchema.parse({
      input_modes: [],
      vision: {
        available: true,
        labels: ['rock', 'paper', 'scissors'],
      },
    });

    expect(parsed.input_modes).toEqual([]);
    expect(parsed.vision?.labels).toEqual(['rock', 'paper', 'scissors']);
  });

  it('parses full ML manifest metadata used by the mobile runtime', () => {
    const parsed = manifestSchema.parse({
      input_modes: ['buttons', 'vision', 'audio'],
      vision_model_slot: 'b',
      vision: {
        available: true,
        version: 'vision-v2',
        sha256: 'a'.repeat(64),
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
        output: { name: 'logits' },
      },
      audio: {
        browser_speech: {
          enabled: true,
          locale: 'en-US',
        },
        onnx: {
          available: false,
          version: 'none',
          sha256: null,
          model_url: null,
        },
      },
      onnx_runtime_web: {
        version: '1.20.1',
        ort_min_js: 'https://cdn.example/ort.min.js',
        wasm_base: 'https://cdn.example/',
      },
    });

    expect(parsed.vision_model_slot).toBe('b');
    expect(parsed.vision?.input?.mean).toEqual([0.485, 0.456, 0.406]);
    expect(parsed.audio?.browser_speech?.locale).toBe('en-US');
    expect(parsed.onnx_runtime_web?.version).toBe('1.20.1');
  });

  it('parses play result payload shape', () => {
    const parsed = playRoundResponseSchema.parse({
      match_complete: false,
      round: 2,
      player_move: 'rock',
      server_move: 'paper',
      round_winner: 'server',
      player_score: 0,
      server_score: 1,
    });

    expect(parsed.round).toBe(2);
    expect(parsed.server_move).toBe('paper');
  });

  it('accepts null winner while a match is still in progress', () => {
    const parsed = playRoundResponseSchema.parse({
      match_complete: false,
      round: 1,
      player_move: 'rock',
      server_move: 'scissors',
      round_winner: 'player',
      player_score: 1,
      server_score: 0,
      winner: null,
    });

    expect(parsed.winner).toBeNull();
  });

  it('keeps session history score fields for foreground resync', () => {
    const parsed = sessionStateSchema.parse({
      session_id: 'session-1',
      user_id: 'pilot',
      round_number: 1,
      max_rounds: 3,
      player_score: 1,
      server_score: 0,
      match_complete: false,
      round_history: [
        {
          round: 1,
          player_move: 'rock',
          server_move: 'scissors',
          round_winner: 'player',
          player_score: 1,
          server_score: 0,
        },
      ],
    });

    expect(parsed.round_history[0].player_score).toBe(1);
    expect(parsed.round_number).toBe(1);
  });
});
