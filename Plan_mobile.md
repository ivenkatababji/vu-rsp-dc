# Plan_mobile.md

## 1) Objective
Build a production-grade React Native app (Expo framework) that reproduces the current web gameplay behavior exactly, then extends it safely for mobile-first performance and reliability.

This revision is based on a second full pass of the repository, including:
1. Backend API and persistence code.
2. Web client integration logic.
3. CLI and simulator clients.
4. Model training/export/deploy flow.
5. Hosting constraints that affect camera/audio behavior.

---

## 2) Deep As-Is System Map (Code-Verified)

### 2.1 Runtime Components
1. deployment/src/server/main.py
- FastAPI app and all player/admin routes.
- Session creation and one-round play orchestration.
- A/B vision slot assignment wiring.
- Serves web SPAs at /game and /admin (plus /admin/game alias).

2. deployment/src/server/db.py
- SQLite tables: config, game_stats, sessions, user_vision_state.
- Startup migration-style ALTER TABLE additions.
- Active session counting, expiration checks, pruning, aggregates.

3. deployment/src/server/ml_manifest.py
- Builds /me/ml/manifest response.
- Computes ONNX hashes and availability metadata.
- Resolves A/B vision model selection and audio model metadata.

4. deployment/src/server/game_auth.py and admin_auth.py
- HTTP Basic auth validation.
- Game auth suppresses WWW-Authenticate on 401 to avoid browser auth popups.
- Admin auth returns 503 when admin credentials are not configured.

5. deployment/src/server/classifier.py and game.py
- classifier.py is string-stub move extraction for API play requests.
- game.py generates server move and decides winner.

6. deployment/src/server/game.html
- Entire web game state machine and integrations.
- Handles login, session start, round play, stats, mode selection.
- Performs browser-side vision ONNX inference and speech recognition.

7. deployment/src/client/client.py and deployment/src/simulator/simulator.py
- Official non-web clients that use the same API contracts.

8. train/train_export.py and train/deploy_model.py
- Creates model.onnx + manifest.json with NCHW + ImageNet norm.
- Deploys artifacts into server ml_artifacts folder structure.

### 2.2 Backend Endpoint Contracts (Exact)

### 2.2.1 Player Endpoints (all require game HTTP Basic auth)

1. GET /me/ml/manifest
- Purpose: returns enabled input modes and ML runtime bundle.
- Response fields include:
  - input_modes
  - vision (available/version/sha256/model_url/input/labels/output)
  - vision_model_slot (a or b)
  - audio (browser_speech and optional onnx metadata)
  - onnx_runtime_web (version, ort_min_js, wasm_base)

2. GET /me/ml/models/{kind}
- kind supported: vision or audio.
- 404 if kind is unknown.
- 404 if requested model is not deployed.
- Returns model.onnx bytes as application/octet-stream.

3. GET /me/stats
- Returns user-level aggregates:
  - sessions_started
  - matches_completed
  - matches_won
  - matches_lost
  - matches_draw
  - rounds_played

4. POST /sessions
- Body currently optional and ignored for identity (authenticated username is used).
- Returns:
  - session_id
  - user_id
  - max_rounds
- 503 when configured max_sessions limit is reached.
- Stores effective vision_model_slot on session row.

5. GET /sessions/{session_id}
- Returns current session state and round_history.
- 404 if session missing.
- 404 if session expired by effective timeout.

6. POST /play
- Request body:
  - session_id: string
  - image: string
- Behavior:
  - server calls classifier.classify_image(image)
  - computes random server_move and winner
  - increments scores and round history
  - sets winner when match completes
- Response:
  - match_complete
  - round
  - player_move
  - server_move
  - round_winner
  - player_score
  - server_score
  - winner (only at match completion)
- Errors:
  - 400 if match already complete
  - 404 for missing/expired session

### 2.2.2 Admin Endpoints (all require admin HTTP Basic auth)
1. GET /admin
- Serves admin SPA.

2. GET /admin/dashboard
- Redirects to /admin#dashboard.

3. GET /admin/monitor/sessions
- Query: include_expired (default false).

4. GET /admin/monitor/win_breakdown_24h

5. GET /admin/monitor/game_stats

6. GET /admin/cfg

7. PUT /admin/cfg
- Validates numeric ranges and input_modes.
- input_modes must be subset of buttons, vision, audio.
- vision_ab_rollout_percent must be 0..100.

8. POST /admin/sessions/prune
- Optional body override for retention_seconds.

### 2.2.3 Web Route Endpoints
1. GET /game
2. HEAD /game
3. GET /admin/game (alias)
4. HEAD /admin/game (alias)

### 2.3 Persistence and Runtime Rules
1. Session timeout effective value:
- If config.session_timeout_seconds > 0, use it.
- Else timeout = 30 * max_rounds.

2. Expired sessions are retained unless explicitly pruned.

3. A/B vision behavior:
- If vision_b/model.onnx missing, everyone is slot a.
- If present, assignment is sticky per user in user_vision_state.
- New user assignment uses hash(user_id) bucket with rollout percent.

4. Config stores input modes as JSON string in input_modes_json.

### 2.4 Web Frontend Integration (Exact Behavior)

### 2.4.1 API Base Prefix Logic
1. game.html computes apiBase from current path ending in /game.
2. If served at /api/game, API calls become /api/sessions, /api/play, etc.
3. Special case: /admin/game uses empty prefix so calls remain /sessions, /play on same origin.

### 2.4.2 Login and Session Flow
1. On login button:
- Builds Basic header from username/password.
- Calls POST /sessions immediately.
- If successful:
  - shows game view
  - calls GET /me/stats
  - calls GET /me/ml/manifest
- If failed:
  - shows error
  - clears auth header

2. Web starts a fresh session at login, not lazily on first move.

### 2.4.3 Round Flow
1. onPlayMove sends POST /play with:
- session_id
- image where move none maps to empty string.

2. UI behavior:
- Sets busy lock before request.
- Updates scoreboard and round result from response.
- On match_complete:
  - keeps controls disabled
  - marks New Match button as attention state
  - refreshes stats

3. New Match:
- Calls POST /sessions again.
- Resets local galleries/scores/round labels.
- Refreshes stats and manifest.

### 2.4.4 Vision Flow
1. Model session prep:
- Reads manifest from /me/ml/manifest.
- Loads ONNX Runtime Web script from manifest.onnx_runtime_web.ort_min_js.
- Uses manifest.wasm_base for wasm path.

2. Model caching:
- IndexedDB db: rps-ml-cache
- Store: models
- Key: vision:<version>:<sha256>

3. File mode:
- User picks image file.
- Client runs ONNX inference in browser.
- Argmax label is posted via onPlayMove.

4. Live camera mode (important parity detail):
- Webcam opens in modal.
- User taps Capture.
- One frame is captured, webcam modal closes, inference runs.
- This is single-capture, not persistent streaming.

### 2.4.5 Audio Flow
1. Uses browser SpeechRecognition/webkitSpeechRecognition.
2. Transcript mapped to move via regex/synonym checks.
3. Sends mapped move through onPlayMove.
4. If no mapping, shows explicit error text.

5. Even if audio ONNX model exists, current web UI path is browser speech-driven.

### 2.4.6 Input Mode Settings and Persistence
1. Server controls allowed modes via manifest input_modes.
2. Client guarantees buttons fallback if manifest fetch fails.
3. Selected mode persisted in localStorage key rps-input-mode.
4. Settings modal only appears after login.

### 2.5 Additional Constraints from Other Clients
1. CLI and simulator also post one /play request per round.
2. Both encode none as empty string.
3. This confirms backend contract assumptions are client-wide, not web-only.

### 2.6 Hosting and Runtime Environment Constraints
1. deployment/HOSTING.md confirms HTTPS is important for camera features in browser contexts.
2. Reverse-proxy pathing exists in real deployment scenarios; mobile must avoid hardcoded path assumptions.

---

## 3) Exact Web-to-Mobile Parity Contract

The mobile app must first implement strict parity before introducing enhancements.

### 3.1 Parity Baseline (Must Match Web)
1. Use HTTP Basic auth against same game endpoints.
2. Start session immediately after successful login.
3. Use one POST /play call per round.
4. Map move none to empty image string.
5. Show round result with same semantics:
- player_move, server_move, round_winner.
6. Keep controls locked at match completion until New Match starts.
7. Refresh user stats after login and after match completion.
8. Fetch and apply /me/ml/manifest after login and new match.
9. Respect server input_modes when deciding visible controls.
10. Use model slot and hash/version exactly from manifest.

### 3.2 Vision Parity Mode (Strict)
1. Implement manual photo import.
2. Implement live camera single-capture flow:
- open camera
- capture one frame
- close camera
- infer and submit move
3. Use argmax over model output logits.
4. Use manifest input shape/layout/mean/std exactly.

### 3.3 Audio Parity Mode (Strict)
1. Implement speech-to-text driven move selection.
2. Mirror transcript mapping rules for rock/paper/scissors/none equivalents.

### 3.4 Non-Parity Enhancements (After Baseline)
1. Persistent live camera.
2. Continuous real-time inference.
3. Temporal smoothing and confidence gating.

These are valid improvements, but must be feature-flagged so exact web parity remains selectable.

---

## 4) Target Mobile Architecture (Expo, Production Grade)

### 4.1 Platform and Build
1. Expo + React Native + TypeScript strict mode.
2. Expo Dev Client + EAS Build for native camera/inference requirements.
3. EAS Update channels: development, staging, production.

### 4.2 App Layers
1. Presentation layer:
- Screens/components and navigation only.

2. Application layer:
- Use-cases: SignIn, StartSession, PlayRound, RefreshMlBundle, etc.

3. Domain layer:
- Entities and parity rules independent of framework.

4. Data layer:
- API repositories, DTO mappers, storage adapters.

5. Infrastructure layer:
- Camera adapter, inference runtime adapter, speech adapter, telemetry, networking.

### 4.3 Suggested Mobile Structure
```text
mobile/
  app/
    screens/
    navigation/
    components/
    features/
      auth/
      session/
      play/
      stats/
      ml/
      vision/
      audio/
    domain/
    data/
    infra/
    config/
    hooks/
    utils/
  tests/
    unit/
    integration/
    e2e/
```

---

## 5) Networking and Auth Strategy

### 5.1 Phase A (Parity)
1. Keep current Basic auth for compatibility.
2. Mobile API client mirrors existing routes and payloads.
3. Add strict request/response runtime validation (Zod) around all endpoints.

### 5.2 Phase B (Hardened)
1. Introduce token auth endpoints while preserving legacy compatibility.
2. Add play/v2 payload with explicit move/source fields.
3. Add request correlation IDs end-to-end.

### 5.3 Mobile HTTP Policy
1. Route-specific timeout budgets.
2. Retries only for safe operations by default.
3. Guarded replay for session creation and play where idempotency is available.

---

## 6) Vision Architecture

### 6.1 Parity Mode (Default first release)
1. File inference path:
- user selects image
- preprocess by manifest
- run ONNX
- submit move via /play

2. Live camera parity path:
- open camera modal/screen
- capture single frame
- close camera
- infer and submit

### 6.2 Enhancement Mode (Feature Flag)
1. Keep camera active while Vision screen is focused.
2. Sample frames at controlled FPS.
3. Run single-flight inference queue to avoid backlog.
4. Apply smoothing window + confidence gating.
5. Allow manual submit of stable move (or optional auto-submit with cooldown).

---

## 7) Audio Architecture

### 7.1 Parity Path
1. Device speech recognition.
2. Map transcript to rock/paper/scissors/none.
3. Submit through same PlayRound use-case.

### 7.2 Future Audio ML Path
1. Optional on-device audio model integration.
2. Keep same final move contract to /play.

---

## 8) State Management and Persistence

### 8.1 Remote State (TanStack Query)
1. session
2. stats
3. mlBundle

### 8.2 Local State (Zustand)
1. auth header/token state
2. active input mode
3. busy/in-flight lock
4. vision pipeline state
5. transient errors

### 8.3 Storage
1. SecureStore for sensitive auth material.
2. AsyncStorage/MMKV for non-sensitive preferences (input mode).
3. File-system or KV metadata for model cache identity (version/hash/slot).

---

## 9) Reliability, Security, and Performance

### 9.1 Reliability
1. Resume-safe session handling on app foreground.
2. Distinguish expired session (404) vs auth failure (401) vs limit (503).
3. Surface deterministic user-facing errors with retry action.

### 9.2 Security
1. Do not store plain credentials beyond immediate Basic header creation in parity mode.
2. Use HTTPS endpoints in non-local environments.
3. Redact auth and PII from logs.

### 9.3 Performance
1. Lazy-load camera/inference dependencies only in Vision mode.
2. Reuse tensors/buffers where runtime allows.
3. Track inference and round-trip latency metrics.

---

## 10) Testing Plan for Exact Web-Match Behavior

### 10.1 Contract Tests
1. Validate response schema for all player endpoints.
2. Validate error status matrix:
- /sessions can return 503 limit reached.
- /play can return 400 match already complete.
- /sessions/{id} can return 404 not found/expired.

### 10.2 Integration Tests
1. Login -> session creation -> stats + manifest fetch sequence.
2. Play rounds until match_complete and ensure controls lock behavior.
3. New match resets session state.

### 10.3 Vision Tests
1. File path classification and move submit.
2. Single-capture live camera parity path.
3. Model cache key parity: vision:<version>:<sha256> equivalent metadata.

### 10.4 Audio Tests
1. Transcript parser equivalence with web mapping semantics.
2. Unknown transcript path shows actionable error.

### 10.5 E2E Device Tests
1. Full match in buttons mode.
2. Full match in vision mode.
3. Full match in audio mode.
4. Foreground/background recovery.

---

## 11) Delivery Roadmap (Parity-First)

### Phase P0: Exact Parity Foundation
1. Expo app skeleton with typed API client.
2. Login + immediate session creation.
3. Buttons mode + scoreboard + new match + stats.
4. Manifest-driven mode visibility.

### Phase P1: Vision and Audio Parity
1. ONNX model fetch/cache and inference by manifest settings.
2. File inference path.
3. Live single-capture camera parity path.
4. Speech recognition parity path.

### Phase P2: Hardening
1. Error-state refinement.
2. Telemetry, crash monitoring, performance budgets.
3. CI/CD gates and staged rollout.

### Phase P3: Feature-Flagged Enhancements
1. Persistent real-time camera mode.
2. Continuous inference and smoothing.
3. Token auth and /play/v2 backend evolution.

---

## 12) Definition of Done

### 12.1 Parity Done
1. Mobile behavior matches web for login/session/play/stats/new match.
2. Input mode availability is manifest-driven and matches web outcomes.
3. Vision and audio paths produce equivalent move submission semantics.
4. Endpoint error handling matches backend contracts.

### 12.2 Production Done
1. Crash-free sessions and latency SLOs met.
2. Security controls and telemetry in place.
3. CI/CD release gates active.
4. Enhancement mode behind feature flags and does not break parity mode.

---

## 13) Key Notes from Latest Code Pass
1. Web currently uses non-persistent webcam capture flow in parity baseline.
2. /sessions body user_id is not the authority; authenticated username is.
3. /play input field remains image string; unknown strings may lead to random classifier fallback.
4. /admin/game alias and path-prefix logic exist for proxied deployments.
5. Session timeout is derived when explicit timeout is zero.
6. A/B slot assignment is sticky and hash-based for new users when slot B model exists.

This plan now explicitly separates exact web parity from optional post-parity improvements, and it maps backend contracts and frontend integrations to mobile implementation requirements line by line.