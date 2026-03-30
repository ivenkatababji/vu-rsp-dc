# Rock Paper Scissors -- ML Experiment (FastAPI)

This project demonstrates a **Rock–Paper–Scissors** game API plus a **browser
client** that can run **on-device vision inference** (ONNX in the browser) when
a model is deployed.

Flow in short:

1.  The player submits a move (**text tiles**, **vision** image, or **audio**
    transcript, depending on server manifest and browser support).
2.  For the **HTTP API** `POST /play`, the server still classifies a **string**
    stub via `classifier.py` (simple heuristic / random). For **vision in the
    web UI**, classification runs **client-side** with ONNX Runtime Web when
    `ml_artifacts/vision/` is present.
3.  The **server** draws a random house move and decides the round winner.
4.  After **N rounds** (default 5), the server records the **match** outcome.

------------------------------------------------------------------------

## Project Structure (`deployment/`)

    requirements.txt
    README.md

    src/server/
        main.py                  # FastAPI app, routes, serves /game
        ml_manifest.py           # Builds /me/ml/manifest from ml_artifacts
        admin_auth.py            # Admin HTTP Basic Auth
        admin_config.json        # Admin credentials (from .example; do not commit)
        game_auth.py             # Game user verification (users_config.json)
        users_config.json        # Game users (optional; see Game user authentication)
        server_config.json       # Optional: db_path, vision defaults, …
        db.py                    # SQLite: sessions, config, stats
        game.py                  # Rules and random server move
        classifier.py            # Stub classifier for API `image` string field
        game.html                # Web SPA at /game
        ml_artifacts/            # Optional: vision/, vision_b/ (A/B), audio/ ONNX + manifest.json

    src/client/
        client.py                # CLI client

    src/simulator/
        simulator.py
        simulator_config.json

------------------------------------------------------------------------

Sibling of this folder (repo root):

    train/
        train_export.py           # Train/export ONNX; single arg: JSON config path
        deploy_model.py           # Copy into deployment/.../ml_artifacts/vision; single arg: JSON config
        train_config.example.json
        deploy_config.example.json
        json_config.py
        requirements-train.txt
        README.md

------------------------------------------------------------------------

## Vision model (training and deploy)

Train a browser-compatible ONNX classifier from a labeled CSV and image folder, then deploy into `src/server/ml_artifacts/vision/`. The tooling lives in **`../train/`** (sibling of `deployment/`). Pass a **JSON config** path to `train_export.py` and `deploy_model.py` (see `../train/train_config.example.json` and `../train/deploy_config.example.json`). Details in **`../train/README.md`**. For an A/B variant, deploy a second model (or copy) into **`ml_artifacts/vision_b/`** with the same layout (`model.onnx` and optional `manifest.json`).

Optional **`server_config.json`** keys **`vision_input_size`** (square) or **`vision_input_width`** / **`vision_input_height`** set default manifest dimensions when no `manifest.json` is present yet. Deployed **`ml_artifacts/vision/manifest.json`** overrides those for the live model.

### Vision A/B experiment (optional)

You can run two browser-side vision models side by side:

| Slot | Directory | Role |
| --- | --- | --- |
| **A** (default) | `src/server/ml_artifacts/vision/` | Primary model (`model.onnx`, optional `manifest.json`). |
| **B** (experiment) | `src/server/ml_artifacts/vision_b/` | Second model; only used if **`vision_b/model.onnx`** exists on disk. |

**Behavior**

- If **`vision_b/model.onnx`** is missing, every user gets slot **A** (same as a single-model setup). No per-user assignment is written for new users until **B** is deployed.
- When **B** exists, the server assigns each **authenticated game user** a sticky slot **A** or **B** and reuses it on later visits. Assignment uses a **stable hash** of the username (`SHA256(user_id)`), not per-request randomness.
- **Rollout:** In the admin UI (**Settings** → *Vision A/B rollout (% to B)*) or via **`PUT /admin/cfg`** field **`vision_ab_rollout_percent`** (integer **0–100**), you control what fraction of **newly assigned** users (those without a row in `user_vision_state` yet) land in **B**. Existing users keep their stored slot when you change the percentage.
  - **0** → all new assignments are **A**.
  - **100** → all new assignments are **B**.
  - Values in between → approximately that percentage of new user IDs hash into **B** (deterministic per username).
- **`GET /me/ml/manifest`** returns the vision bundle for the caller’s slot. The JSON includes **`vision_model_slot`** (`"a"` or `"b"`) so clients can tell which arm they received. Vision model download (`GET /me/ml/models/vision`) serves the matching file; the IndexedDB key still uses manifest **`version`** + **`sha256`**, so **A** and **B** cache separately when their hashes differ.

**Session reporting**

- Each **`POST /sessions`** stores **`vision_model_slot`** on the session row: **the effective model used for that session** (`b` only if slot B was assigned **and** the B ONNX file was present at session creation). Admin **Dashboard → Active sessions** shows a **Vision** column (**A** / **B**). Older sessions created before this feature may show an empty value.

------------------------------------------------------------------------

## Installation

From the **`deployment/`** directory:

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Server

```bash
cd deployment/src/server
uvicorn main:app --reload
```

The server will run at:

    http://localhost:8000

Interactive API documentation is available at:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Web game (browser)

Open **`/game`** (e.g. `http://localhost:8000/game`) for the SPA. Sign in with a
game username and password (HTTP Basic on API calls).

- **Text:** 2×2 tiles (Rock, Paper, Scissors, None).
- **Vision:** file upload or live webcam when the server exposes an ONNX vision
  bundle under `ml_artifacts/vision/` and the browser loads ONNX Runtime Web
  from the manifest (see **Client-side ONNX cache** below).
- **Audio:** browser speech recognition when enabled in the audio manifest and
  the browser supports Web Speech API.

**Settings** (vertical ⋮ menu) appears **after** login only. It lists allowed
**input modes** from the server manifest. The chosen mode is persisted in the
browser as **`localStorage`** key **`rps-input-mode`** (`buttons` | `vision` |
`audio`).

Round results, optional match history thumbnails, and **New match** live in the
same card; layout is tuned for small viewports.

------------------------------------------------------------------------

## Client-side ONNX cache (browser)

The vision **`.onnx`** file is **not** stored in `localStorage`. After the first
authenticated download, the game caches the model bytes in **IndexedDB**:

| | |
| --- | --- |
| **Database** | `rps-ml-cache` (version `1`) |
| **Object store** | `models` |
| **Key** | `vision:<manifest_version>:<sha256>` |

A cache hit skips re-downloading when the version and hash match the manifest.
ONNX Runtime **JavaScript and WASM** are loaded from URLs in the manifest, not
from this database.

------------------------------------------------------------------------

## Game user authentication

Only **pre-provisioned users** can create sessions and play. Credentials are checked
with **HTTP Basic Auth** against a config file.

- **Default:** If `src/server/users_config.json` is missing, a built-in user **guest** / **guest** is allowed.
- **Config file:** Copy `src/server/users_config.json.example` to `src/server/users_config.json` and add users as `"username": "password"` entries. The default user `guest` / `guest` can be overridden or extended there.

**Invalid username or password** returns **401** with a JSON `detail` message.
The server **does not** send `WWW-Authenticate: Basic` on that failure, so the
browser will not open a second system Basic-auth dialog on top of the in-page
login form.

The CLI sends the chosen username and password (e.g. `python client.py guest guest` or `python client.py alice` and then the password when prompted).

------------------------------------------------------------------------

## Running the Client

The CLI uses **one `POST /play` per round** (simulation stays on the client). You can enter moves one-by-one, pass them all via `--batch-moves` (still sent as separate `/play` calls), or use **`--loops`** for many random games.

``` bash
cd client
python client.py
```

With default user (guest/guest):

``` bash
python client.py
# or
python client.py guest guest
```

With another user (password prompted if omitted):

``` bash
python client.py alice
# or
python client.py alice your-password
```

You will be prompted for each move (`Move 1/N`, …); each prompt triggers a **`POST /play`**. After the last round, the match summary is printed.

Provide all moves up front (each still sent as its own **`POST /play`**, in order):

``` bash
python client.py guest guest --batch-moves "rock paper scissors rock paper"
```

Optional **`--url`** (or env **`RPS_BASE_URL`**) for the server base URL.

Run **many games** with **random** moves (no prompts), e.g. 100 matches:

``` bash
python client.py guest guest --loops 100
```

------------------------------------------------------------------------

## Load simulator

Run many complete matches in one go. Each match uses **`POST /sessions`** then a **client-side loop** of **`POST /play`** (one per round); the server does not accept batch play.

``` bash
cd simulator
pip install -r ../requirements.txt   # if needed (uses `requests`)
cp simulator_config.json.example simulator_config.json
# Ensure server users exist (see users_config.json) — example includes usr1/pwd1, usr2/pwd2, usr3/pwd3, guest/guest
python simulator.py --games 100
```

- **`--games N`** — number of matches (default `100`).
- **`--config path`** — JSON with optional `base_url` and optional `users` array
  `[{ "username": "...", "password": "..." }, ...]`.
- If `users` is missing or empty, users are read from **`src/server/users_config.json`**
  (username/password map) relative to the repo layout.
- Each round’s “image” stub is chosen at random: **rock**, **paper**, **scissors**, or **none**
  (empty string → classifier picks a random move).

------------------------------------------------------------------------

## Game Flow

**Web UI (vision path):** image → **browser ONNX** → label → `POST /play` with
that move.

**API / CLI `POST /play`:** body still carries an **`image` string**; the server
runs **`classifier.classify_image`** (stub / heuristic) to derive the player
move.

Then for every round:

    Player move (from client or stub classifier)
            ↓
    Server random move
            ↓
    Round winner and scores

The match runs until **`max_rounds`** (default **5**), then the server records the
match winner.

------------------------------------------------------------------------

## Multiple Sessions

The server supports **multiple concurrent sessions**. Each game is
tracked independently:

-   **Create session:** `POST /sessions` returns a `session_id`.
-   **Play rounds:** `POST /play` with body `{ "session_id": "<id>", "image": "..." }`.
-   **Session status:** `GET /sessions/{session_id}` returns current scores,
    full round history, and match winner when complete.

Player–server engagement and the match winner are stored **per session**.

------------------------------------------------------------------------

## Example Output

    Round: 1
    Player move: rock
    Server move: scissors
    Round winner: player
    Score: 1 - 0

After five rounds:

    MATCH COMPLETE
    Player Score: 3
    Server Score: 2
    Winner: player

------------------------------------------------------------------------

## Admin console and APIs

The admin UI (`/admin`) and all admin APIs (`/admin/cfg`, `/admin/monitor/*`) are
protected with **HTTP Basic Auth**. Credentials are read from a config file.

**Configuration** (`GET` / `PUT /admin/cfg`) includes **`vision_ab_rollout_percent`** (0–100) for the vision A/B experiment when `ml_artifacts/vision_b/model.onnx` exists. The dashboard summarizes this value next to other server settings.

**Monitoring** (`GET /admin/monitor/sessions`) returns each session’s **`vision_model_slot`** for correlating outcomes with model arm A vs B.

1. Copy the example config and set a password:
   ```bash
   cd deployment/src/server
   cp admin_config.json.example admin_config.json
   # Edit admin_config.json: set admin_username and admin_password
   ```
2. Open `http://localhost:9000/admin` in a browser. When prompted, enter the
   username and password from `admin_config.json`.
3. Keep `admin_config.json` out of version control (e.g. add it to `.gitignore`)
   so production credentials are not committed.

If `admin_config.json` is missing, admin routes return **503** with instructions
to create the file.

------------------------------------------------------------------------

## State Storage

Sessions, configuration, and game statistics are stored in **SQLite**. By
default the database is **in-memory** (`:memory:`), so data is lost when the
server stops. To persist state, create `src/server/server_config.json` from
`src/server/server_config.json.example` and set **`db_path`** to a file path
(e.g. `"state.db"`). The server reads this file at startup.

Relevant tables and columns (created or migrated on startup):

- **`config`** — includes **`vision_ab_rollout_percent`** (default `0`) for vision A/B rollout.
- **`sessions`** — includes **`vision_model_slot`** (`a` / `b` / `NULL` for legacy rows): effective vision arm when the session was created.
- **`user_vision_state`** — one row per game user: sticky **`vision_slot`** (`a` or `b`) and **`updated_at`**, used when `vision_b` is deployed.

If **`db_path`** points to a file, sticky assignments and session-level model labels survive restarts; in-memory mode resets them when the process exits.

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   Server-side **real image** classification for `POST /play` (multipart or
    base64) instead of the string stub
-   Persist game results for **training data collection**
-   Deploy inference as a **scalable ML microservice** behind the same API

------------------------------------------------------------------------

## Purpose

Demonstrate **FastAPI** session/play APIs together with a **browser client** that
can run **exported ONNX** vision models locally and cache them for repeat visits.
