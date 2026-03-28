# Rock Paper Scissors -- ML Experiment (FastAPI)

This project demonstrates a simple **machine learning inference pipeline
simulation** using a Rock--Paper--Scissors game.

The goal is to mimic a real-world ML workflow where:

1.  A user provides an **image input** (stubbed as a string for now)
2.  A **classifier model** infers the user's move
3.  The **server generates a random move**
4.  The system determines the **winner of the round**
5.  After **5 rounds**, the server declares the **match winner**

The ML inference stage is currently **stubbed** and can later be
replaced with a real image classification model.

------------------------------------------------------------------------

## Project Structure

    rps_ml_match_experiment/

    requirements.txt
    README.md

    src/server/
        main.py                  # FastAPI server
        admin_auth.py            # Admin HTTP Basic Auth (credentials from config file)
        admin_config.json        # Admin credentials (create from .example; do not commit secrets)
        game_auth.py            # Game user auth (pre-provisioned users from config file)
        users_config.json        # Game users username:password (optional; see Game user authentication)
        server_config.json       # Server config (optional; db_path for SQLite file or :memory:)
        db.py                    # SQLite state (sessions, config, game stats)
        game.py             # Game rules and random server move
        classifier.py       # Stub image classifier
        game.html           # Web game SPA at /game

    src/client/
        client.py        # Simple CLI client

    src/simulator/
        simulator.py              # Simulator (many games; client loops POST /play)
        simulator_config.json     # Optional: base_url + users list

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

Train a browser-compatible ONNX classifier from a labeled CSV and image folder, then deploy into `src/server/ml_artifacts/vision/`. The tooling lives in **`../train/`** (sibling of `deployment/`). Pass a **JSON config** path to `train_export.py` and `deploy_model.py` (see `../train/train_config.example.json` and `../train/deploy_config.example.json`). Details in **`../train/README.md`**.

Optional **`server_config.json`** keys **`vision_input_size`** (square) or **`vision_input_width`** / **`vision_input_height`** set default manifest dimensions when no `manifest.json` is present yet. Deployed **`ml_artifacts/vision/manifest.json`** overrides those for the live model.

------------------------------------------------------------------------

## Installation

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Server

Start the FastAPI server:

``` bash
cd src/server
uvicorn main:app --reload
```

The server will run at:

    http://localhost:8000

Interactive API documentation is available at:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Web game (browser)

Open **`/game`** (e.g. `http://localhost:9000/game` if you use port 9000) for a single-page
client: sign in with a game username/password, then use a **2×2** tile grid
(Rock, Paper, Scissors, None). Each round shows who won; after the last round a
match summary appears with **Play again**. Styling uses the same host as the API
(no CORS issues).

------------------------------------------------------------------------

## Game user authentication

Only **pre-provisioned users** can create sessions and play. Credentials are checked
with **HTTP Basic Auth** against a config file.

- **Default:** If `server/users_config.json` is missing, a built-in user **guest** / **guest** is allowed.
- **Config file:** Copy `server/users_config.json.example` to `server/users_config.json` and add users as `"username": "password"` entries. The default user `guest` / `guest` can be overridden or extended there.

The client sends the chosen username and password (e.g. `python client.py guest guest` or `python client.py alice` and then the password when prompted).

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
- If `users` is missing or empty, users are read from **`server/users_config.json`**
  (username/password map) relative to the repo layout.
- Each round’s “image” stub is chosen at random: **rock**, **paper**, **scissors**, or **none**
  (empty string → classifier picks a random move).

------------------------------------------------------------------------

## Game Flow

Each round performs the following steps:

    User Input (image stub)
            ↓
    Image Classification (stub)
            ↓
    Player Move
            ↓
    Server Random Move
            ↓
    Round Winner

The match runs for **5 rounds**, after which the server determines the
winner based on the majority score.

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

1. Copy the example config and set a password:
   ``` bash
   cd server
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
server stops. To persist state, create `server/server_config.json` from
`server/server_config.json.example` and set **`db_path`** to a file path
(e.g. `"state.db"`). The server reads this file at startup.

------------------------------------------------------------------------

## Future Improvements

Possible extensions include:

-   Replace the stub classifier with a **real image classification
    model**
-   Accept **actual image uploads** from the client
-   Persist game results for **training data collection**
-   Deploy the inference service as a **scalable ML microservice**

------------------------------------------------------------------------

## Purpose

This project is intended as a **simple demonstration of integrating an
ML inference step with a REST API service** using FastAPI.
