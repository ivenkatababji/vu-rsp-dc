import { requestJson } from '../../lib/http';
import {
  createSessionResponseSchema,
  manifestSchema,
  playRoundResponseSchema,
  sessionStateSchema,
  statsSchema,
  type CreateSessionResponse,
  type Manifest,
  type PlayRoundResponse,
  type SessionState,
  type Stats,
} from './schemas';

export async function createSession(authHeader: string): Promise<CreateSessionResponse> {
  const data = await requestJson<unknown>('/sessions', {
    method: 'POST',
    authHeader,
    body: {},
    timeoutMs: 12000,
  });
  return createSessionResponseSchema.parse(data);
}

export async function getStats(authHeader: string): Promise<Stats> {
  const data = await requestJson<unknown>('/me/stats', { authHeader, timeoutMs: 10000 });
  return statsSchema.parse(data);
}

export async function getManifest(authHeader: string): Promise<Manifest> {
  const data = await requestJson<unknown>('/me/ml/manifest', { authHeader, timeoutMs: 12000 });
  return manifestSchema.parse(data);
}

export async function playRound(
  authHeader: string,
  sessionId: string,
  move: string,
): Promise<PlayRoundResponse> {
  const image = move === 'none' ? '' : move;
  const data = await requestJson<unknown>('/play', {
    method: 'POST',
    authHeader,
    body: { session_id: sessionId, image, move },
    timeoutMs: 15000,
  });
  return playRoundResponseSchema.parse(data);
}

export async function getSessionState(authHeader: string, sessionId: string): Promise<SessionState> {
  const data = await requestJson<unknown>(`/sessions/${sessionId}`, { authHeader, timeoutMs: 10000 });
  return sessionStateSchema.parse(data);
}
