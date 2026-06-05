export type AppErrorScreen =
  | 'auth_error'
  | 'network_failure'
  | 'max_sessions_reached'
  | 'session_expired'
  | 'match_complete'
  | 'model_unavailable';

function extractStatusCode(message: string): number | null {
  const prefixMatch = message.match(/^(\d{3}):/);
  if (prefixMatch) {
    return Number(prefixMatch[1]);
  }

  const inlineMatch = message.match(/\b(400|401|403|404|409|429|500|502|503|504)\b/);
  if (inlineMatch) {
    return Number(inlineMatch[1]);
  }

  return null;
}

export function classifyAppError(error: unknown): AppErrorScreen | null {
  if (!(error instanceof Error)) {
    return null;
  }

  const message = error.message.toLowerCase();
  const statusCode = extractStatusCode(error.message);

  if (statusCode === 401 || statusCode === 403) {
    return 'auth_error';
  }
  if (statusCode === 503) {
    return 'max_sessions_reached';
  }
  if (statusCode === 404) {
    return 'session_expired';
  }
  if (statusCode === 400 && message.includes('match already complete')) {
    return 'match_complete';
  }

  if (
    message.includes('network request failed') ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('timeout')
  ) {
    return 'network_failure';
  }

  return null;
}
