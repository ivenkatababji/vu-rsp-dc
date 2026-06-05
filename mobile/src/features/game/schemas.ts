import { z } from 'zod';

export const inputModeSchema = z.enum(['buttons', 'vision', 'audio']);

export const manifestSchema = z.object({
  input_modes: z.array(inputModeSchema).default(['buttons']),
  vision_model_slot: z.enum(['a', 'b']).optional(),
  vision: z
    .object({
      available: z.boolean().optional(),
      version: z.string().optional(),
      sha256: z.string().nullable().optional(),
      model_url: z.string().nullable().optional(),
      labels: z.array(z.string()).default(['rock', 'paper', 'scissors']),
      input: z
        .object({
          name: z.string().optional(),
          width: z.number().optional(),
          height: z.number().optional(),
          layout: z.string().optional(),
          mean: z.array(z.number()).optional(),
          std: z.array(z.number()).optional(),
        })
        .optional(),
      output: z.record(z.string(), z.unknown()).optional(),
    })
    .optional(),
  audio: z
    .object({
      browser_speech: z
        .object({
          enabled: z.boolean().optional(),
          locale: z.string().optional(),
        })
        .optional(),
      onnx: z
        .object({
          available: z.boolean().optional(),
          version: z.string().optional(),
          sha256: z.string().nullable().optional(),
          model_url: z.string().nullable().optional(),
        })
        .optional(),
    })
    .optional(),
  onnx_runtime_web: z
    .object({
      version: z.string().optional(),
      ort_min_js: z.string().optional(),
      wasm_base: z.string().optional(),
    })
    .optional(),
});

export const statsSchema = z.object({
  sessions_started: z.number(),
  matches_completed: z.number(),
  matches_won: z.number(),
  matches_lost: z.number(),
  matches_draw: z.number(),
  rounds_played: z.number(),
});

export const createSessionResponseSchema = z.object({
  session_id: z.string(),
  user_id: z.string().nullable().optional(),
  max_rounds: z.number(),
});

export const playRoundResponseSchema = z.object({
  match_complete: z.boolean(),
  round: z.number(),
  player_move: z.string(),
  server_move: z.string(),
  round_winner: z.string(),
  player_score: z.number(),
  server_score: z.number(),
  winner: z.string().nullable().optional(),
});

export const sessionHistoryItemSchema = z.object({
  round: z.number(),
  player_move: z.string(),
  server_move: z.string(),
  round_winner: z.string(),
  player_score: z.number().optional(),
  server_score: z.number().optional(),
});

export const sessionStateSchema = z.object({
  session_id: z.string(),
  user_id: z.string().nullable().optional(),
  max_rounds: z.number(),
  round_number: z.number().default(0),
  player_score: z.number().default(0),
  server_score: z.number().default(0),
  match_complete: z.boolean().default(false),
  winner: z.string().nullable().optional(),
  round_history: z.array(sessionHistoryItemSchema).default([]),
});

export type InputMode = z.infer<typeof inputModeSchema>;
export type Manifest = z.infer<typeof manifestSchema>;
export type Stats = z.infer<typeof statsSchema>;
export type CreateSessionResponse = z.infer<typeof createSessionResponseSchema>;
export type PlayRoundResponse = z.infer<typeof playRoundResponseSchema>;
export type SessionState = z.infer<typeof sessionStateSchema>;
