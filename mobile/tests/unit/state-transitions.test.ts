import { useGameStore } from '../../src/store/useGameStore';

describe('game state transitions', () => {
  beforeEach(() => {
    useGameStore.setState({
      authHeader: null,
      username: null,
      session: null,
      selectedMode: 'buttons',
      busy: false,
      lastRound: null,
    });
  });

  it('sets auth and session then clears on sign-out', () => {
    const store = useGameStore.getState();

    store.setAuth('Basic abc', 'pilot');
    store.setSession('session-1', 5);

    let snapshot = useGameStore.getState();
    expect(snapshot.authHeader).toBe('Basic abc');
    expect(snapshot.username).toBe('pilot');
    expect(snapshot.session?.sessionId).toBe('session-1');

    snapshot.clearAuth();
    snapshot = useGameStore.getState();

    expect(snapshot.authHeader).toBeNull();
    expect(snapshot.username).toBeNull();
    expect(snapshot.session).toBeNull();
    expect(snapshot.lastRound).toBeNull();
    expect(snapshot.selectedMode).toBe('buttons');
  });

  it('tracks selected mode changes', () => {
    const store = useGameStore.getState();

    store.setSelectedMode('vision');
    expect(useGameStore.getState().selectedMode).toBe('vision');

    store.setSelectedMode('audio');
    expect(useGameStore.getState().selectedMode).toBe('audio');
  });

  it('stores round result and clears it with clearSession', () => {
    const store = useGameStore.getState();

    store.setSession('session-2', 3);
    store.setLastRound({
      match_complete: false,
      round: 1,
      player_move: 'rock',
      server_move: 'scissors',
      round_winner: 'player',
      player_score: 1,
      server_score: 0,
    });

    expect(useGameStore.getState().lastRound?.round).toBe(1);

    store.clearSession();

    expect(useGameStore.getState().session).toBeNull();
    expect(useGameStore.getState().lastRound).toBeNull();
  });
});
