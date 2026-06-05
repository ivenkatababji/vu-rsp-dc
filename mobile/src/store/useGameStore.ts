import { create } from 'zustand';
import type { InputMode, PlayRoundResponse } from '../features/game/schemas';

type SessionInfo = {
  sessionId: string;
  maxRounds: number;
};

type GameState = {
  authHeader: string | null;
  username: string | null;
  session: SessionInfo | null;
  selectedMode: InputMode;
  busy: boolean;
  lastRound: PlayRoundResponse | null;
  setAuth: (authHeader: string, username: string) => void;
  clearAuth: () => void;
  setSession: (sessionId: string, maxRounds: number) => void;
  clearSession: () => void;
  setSelectedMode: (mode: InputMode) => void;
  setBusy: (busy: boolean) => void;
  setLastRound: (round: PlayRoundResponse | null) => void;
};

export const useGameStore = create<GameState>((set) => ({
  authHeader: null,
  username: null,
  session: null,
  selectedMode: 'buttons',
  busy: false,
  lastRound: null,
  setAuth: (authHeader, username) => set({ authHeader, username }),
  clearAuth: () =>
    set({
      authHeader: null,
      username: null,
      session: null,
      lastRound: null,
      busy: false,
      selectedMode: 'buttons',
    }),
  setSession: (sessionId, maxRounds) => set({ session: { sessionId, maxRounds } }),
  clearSession: () => set({ session: null, lastRound: null }),
  setSelectedMode: (mode) => set({ selectedMode: mode }),
  setBusy: (busy) => set({ busy }),
  setLastRound: (lastRound) => set({ lastRound }),
}));
