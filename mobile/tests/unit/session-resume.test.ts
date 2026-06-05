import { sessionStateToRounds } from '../../src/features/game/session';

describe('session resume adapter', () => {
  it('maps backend round history into local round results', () => {
    const rounds = sessionStateToRounds({
      session_id: 'session-1',
      user_id: 'pilot',
      round_number: 2,
      max_rounds: 3,
      player_score: 1,
      server_score: 1,
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
        {
          round: 2,
          player_move: 'paper',
          server_move: 'scissors',
          round_winner: 'server',
          player_score: 1,
          server_score: 1,
        },
      ],
    });

    expect(rounds).toHaveLength(2);
    expect(rounds[1]).toMatchObject({
      round: 2,
      player_score: 1,
      server_score: 1,
      match_complete: false,
    });
  });

  it('marks the final resumed round as complete and includes the winner', () => {
    const rounds = sessionStateToRounds({
      session_id: 'session-2',
      user_id: 'pilot',
      round_number: 3,
      max_rounds: 3,
      player_score: 2,
      server_score: 1,
      match_complete: true,
      winner: 'player',
      round_history: [
        {
          round: 3,
          player_move: 'rock',
          server_move: 'scissors',
          round_winner: 'player',
          player_score: 2,
          server_score: 1,
        },
      ],
    });

    expect(rounds[0].match_complete).toBe(true);
    expect(rounds[0].winner).toBe('player');
  });
});
