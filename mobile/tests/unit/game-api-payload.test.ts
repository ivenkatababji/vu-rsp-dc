import { beforeEach, describe, expect, it, vi } from 'vitest';

import { playRound } from '../../src/features/game/api';
import { requestJson } from '../../src/lib/http';

vi.mock('../../src/lib/http', () => ({
  requestJson: vi.fn(),
}));

describe('game api payloads', () => {
  beforeEach(() => {
    vi.mocked(requestJson).mockReset();
    vi.mocked(requestJson).mockResolvedValue({
      match_complete: false,
      round: 1,
      player_move: 'rock',
      server_move: 'scissors',
      round_winner: 'player',
      player_score: 1,
      server_score: 0,
    });
  });

  it('sends explicit move and image payload for standard rounds', async () => {
    await playRound('Basic abc', 'session-1', 'rock');

    expect(requestJson).toHaveBeenCalledWith('/play', {
      method: 'POST',
      authHeader: 'Basic abc',
      body: {
        session_id: 'session-1',
        image: 'rock',
        move: 'rock',
      },
      timeoutMs: 15000,
    });
  });

  it('maps none to empty image while preserving explicit move field', async () => {
    await playRound('Basic abc', 'session-1', 'none');

    expect(requestJson).toHaveBeenCalledWith('/play', {
      method: 'POST',
      authHeader: 'Basic abc',
      body: {
        session_id: 'session-1',
        image: '',
        move: 'none',
      },
      timeoutMs: 15000,
    });
  });
});
