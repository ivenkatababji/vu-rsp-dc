import { classifyAppError } from '../../src/features/game/errors';

describe('app error classification', () => {
  it('classifies 401 as auth error', () => {
    expect(classifyAppError(new Error('401: unauthorized'))).toBe('auth_error');
  });

  it('classifies 503 as max sessions reached', () => {
    expect(classifyAppError(new Error('503: Service Unavailable'))).toBe('max_sessions_reached');
  });

  it('classifies 404 as session expired', () => {
    expect(classifyAppError(new Error('404: session expired'))).toBe('session_expired');
  });

  it('classifies network failure text', () => {
    expect(classifyAppError(new Error('Network request failed'))).toBe('network_failure');
  });

  it('classifies match complete conflict as dedicated recovery screen', () => {
    expect(classifyAppError(new Error('400: Match already complete. Create a new session to play again.'))).toBe(
      'match_complete',
    );
  });

  it('returns null for unknown errors', () => {
    expect(classifyAppError(new Error('400: bad request'))).toBeNull();
  });
});
