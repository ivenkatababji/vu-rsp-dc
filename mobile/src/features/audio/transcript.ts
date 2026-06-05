export type GameMove = 'rock' | 'paper' | 'scissors' | 'none';

const ROCK_PATTERNS = /\b(rock|stone|boulder|roc)\b/i;
const PAPER_PATTERNS = /\b(paper|sheet|page|newspaper)\b/i;
const SCISSORS_PATTERNS = /\b(scissor|scissors|shear|shears|snip|cut)\b/i;
const NONE_PATTERNS = /\b(none|random|cancel|skip|pass|no move)\b/i;

export function transcriptToMove(transcript: string): GameMove | null {
  const normalized = transcript.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  if (NONE_PATTERNS.test(normalized)) {
    return 'none';
  }
  if (ROCK_PATTERNS.test(normalized)) {
    return 'rock';
  }
  if (PAPER_PATTERNS.test(normalized)) {
    return 'paper';
  }
  if (SCISSORS_PATTERNS.test(normalized)) {
    return 'scissors';
  }
  if (normalized.includes('rock')) {
    return 'rock';
  }
  if (normalized.includes('paper')) {
    return 'paper';
  }
  if (normalized.includes('scissor')) {
    return 'scissors';
  }

  return null;
}
