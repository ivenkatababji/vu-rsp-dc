import type { PlayRoundResponse, SessionState } from './schemas';

export function sessionStateToRounds(sessionState: SessionState): PlayRoundResponse[] {
  return sessionState.round_history.map((round) => ({
    match_complete: sessionState.match_complete && round.round >= sessionState.round_number,
    round: round.round,
    player_move: round.player_move,
    server_move: round.server_move,
    round_winner: round.round_winner,
    player_score: round.player_score ?? sessionState.player_score,
    server_score: round.server_score ?? sessionState.server_score,
    winner:
      sessionState.match_complete && round.round >= sessionState.round_number
        ? sessionState.winner ?? undefined
        : undefined,
  }));
}

export function latestRoundFromSession(sessionState: SessionState): PlayRoundResponse | null {
  const rounds = sessionStateToRounds(sessionState);
  return rounds.at(-1) ?? null;
}
