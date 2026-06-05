import { transcriptToMove } from '../../src/features/audio/transcript';

describe('audio transcript mapping', () => {
  it('maps rock intent', () => {
    expect(transcriptToMove('I choose rock')).toBe('rock');
  });

  it('maps paper intent', () => {
    expect(transcriptToMove('paper now')).toBe('paper');
  });

  it('maps scissors intent from synonym', () => {
    expect(transcriptToMove('use shears')).toBe('scissors');
  });

  it('maps none intent', () => {
    expect(transcriptToMove('skip this turn')).toBe('none');
  });

  it('maps the web random synonym to none', () => {
    expect(transcriptToMove('pick random')).toBe('none');
  });

  it('matches web substring fallback behavior', () => {
    expect(transcriptToMove('the hand is rocking')).toBe('rock');
    expect(transcriptToMove('newspaper clipping')).toBe('paper');
    expect(transcriptToMove('scissoring motion')).toBe('scissors');
  });

  it('returns null for unrelated transcript', () => {
    expect(transcriptToMove('hello world')).toBeNull();
  });
});
