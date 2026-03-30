import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel

from admin_auth import verify_admin
from game_auth import verify_game_user
from classifier import classify_image
from game import random_move, decide_winner
import db
import ml_manifest

app = FastAPI(title="Rock Paper Scissors ML Experiment")

_VALID_INPUT_MODES = frozenset({"buttons", "vision", "audio"})


def _normalize_input_modes(modes: list[str]) -> list[str]:
    out: list[str] = []
    for m in modes:
        if m in _VALID_INPUT_MODES and m not in out:
            out.append(m)
    if "buttons" not in out:
        out.insert(0, "buttons")
    return out

# Admin routes: protected by HTTP Basic Auth (credentials in admin_config.json)
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(verify_admin)])

# Per-session state (from DB): { session_id?, user_id, max_rounds, round_number, ... }
SessionState = dict

_SERVER_CONFIG_PATH = Path(__file__).parent / "server_config.json"


def _load_server_config() -> dict[str, Any]:
    """Load server config from server_config.json. Missing file or key => defaults."""
    if not _SERVER_CONFIG_PATH.exists():
        return {"db_path": ":memory:"}
    try:
        data = json.loads(_SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
        return {"db_path": data.get("db_path") or ":memory:"}
    except (json.JSONDecodeError, OSError):
        return {"db_path": ":memory:"}


@app.on_event("startup")
def startup():
    config = _load_server_config()
    db.init_db(path=config.get("db_path"))


class PlayRequest(BaseModel):
    session_id: str
    image: str


class PlayRoundResponse(BaseModel):
    match_complete: bool
    round: int
    player_move: str
    server_move: str
    round_winner: str
    player_score: int
    server_score: int
    winner: Optional[str] = None


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    max_rounds: int


class RoundResult(BaseModel):
    round: int
    player_move: str
    server_move: str
    round_winner: str
    player_score: int
    server_score: int


class SessionStatus(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    round_number: int
    player_score: int
    server_score: int
    round_history: list[RoundResult]
    match_complete: bool
    winner: Optional[str] = None


class UserStatsResponse(BaseModel):
    sessions_started: int
    matches_completed: int
    matches_won: int
    matches_lost: int
    matches_draw: int
    rounds_played: int


# --- Admin: monitoring & config models ---

class MonitorSessionSummary(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    max_rounds: int
    round_number: int
    player_score: int
    server_score: int
    winner: Optional[str] = None
    match_complete: bool
    created_at: float
    last_activity_at: float
    expired: bool
    vision_model_slot: Optional[str] = None


class GameStatsResponse(BaseModel):
    total_sessions_created: int
    total_matches_completed: int
    total_rounds_played: int
    player_wins: int
    server_wins: int
    draws: int
    active_sessions: int


class WinBreakdown24hResponse(BaseModel):
    player_wins: int
    server_wins: int
    draws: int


class ConfigResponse(BaseModel):
    max_rounds: int
    max_sessions: int
    session_timeout_seconds: int
    retention_seconds: int
    input_modes: list[str]
    vision_ab_rollout_percent: int


class ConfigUpdateRequest(BaseModel):
    max_rounds: Optional[int] = None
    max_sessions: Optional[int] = None
    session_timeout_seconds: Optional[int] = None
    retention_seconds: Optional[int] = None
    input_modes: Optional[list[str]] = None
    vision_ab_rollout_percent: Optional[int] = None


class PruneRequest(BaseModel):
    retention_seconds: Optional[int] = None


class PruneResponse(BaseModel):
    pruned_count: int


def _is_session_expired(state: SessionState) -> bool:
    timeout = db.get_effective_session_timeout_seconds()
    if timeout <= 0:
        return False
    last = state.get("last_activity_at") or state.get("created_at") or 0
    return (time.time() - last) > timeout


def _resolve_vision_slot(user_id: str) -> str:
    """
    Sticky A/B vision slot per authenticated user when ml_artifacts/vision_b/model.onnx exists.
    New users: rollout uses config vision_ab_rollout_percent (0–100); hash(user_id) % 100 < pct → B.
    """
    if not ml_manifest.vision_b_has_model():
        return "a"
    existing = db.get_user_vision_slot(user_id)
    if existing in ("a", "b"):
        return existing
    cfg = db.get_config()
    pct = max(0, min(100, int(cfg.get("vision_ab_rollout_percent") or 0)))
    if pct <= 0:
        slot = "a"
    elif pct >= 100:
        slot = "b"
    else:
        bucket = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        slot = "b" if bucket < pct else "a"
    db.set_user_vision_slot(user_id, slot)
    return slot


def _effective_vision_model_slot(user_id: str) -> str:
    """Which vision ONNX arm is actually served (B only if deployed)."""
    assigned = _resolve_vision_slot(user_id)
    return "b" if assigned == "b" and ml_manifest.vision_b_has_model() else "a"


def _get_session(session_id: str) -> SessionState:
    state = db.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if _is_session_expired(state):
        raise HTTPException(status_code=404, detail="Session expired")
    state["session_id"] = session_id
    return state


@app.get("/me/ml/manifest")
def get_ml_manifest(username: str = Depends(verify_game_user)):
    """
    Client-side ML bundle: enabled input modes, ONNX manifests, ORT Web CDN pins.
    Vision/audio models are optional; when absent, available=false and UI falls back to buttons (and browser STT for audio if enabled).
    """
    cfg = db.get_config()
    modes = _normalize_input_modes(cfg.get("input_modes") or ["buttons"])
    slot = _resolve_vision_slot(username)
    return ml_manifest.build_ml_bundle(modes, vision_slot=slot)


@app.get("/me/ml/models/{kind}")
def download_ml_model(kind: str, username: str = Depends(verify_game_user)):
    """Authenticated download of ONNX artifact (vision or audio)."""
    if kind not in ("vision", "audio"):
        raise HTTPException(status_code=404, detail="Unknown model kind")
    slot = _resolve_vision_slot(username) if kind == "vision" else "a"
    path = ml_manifest.model_file_for_kind(kind, vision_slot=slot)
    if path is None:
        raise HTTPException(status_code=404, detail="Model not deployed")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="model.onnx",
    )


@app.get("/me/stats", response_model=UserStatsResponse)
def get_my_stats(username: str = Depends(verify_game_user)):
    """Career-style stats for the authenticated game user (from sessions still in the database)."""
    s = db.get_user_stats(username)
    return UserStatsResponse(
        sessions_started=s["sessions_started"],
        matches_completed=s["matches_completed"],
        matches_won=s["matches_won"],
        matches_lost=s["matches_lost"],
        matches_draw=s["matches_draw"],
        rounds_played=s["rounds_played"],
    )


@app.post("/sessions", response_model=SessionResponse)
def create_session(
    username: str = Depends(verify_game_user),
    body: Optional[CreateSessionRequest] = None,
):
    """Create a new game session. Requires game user auth (Basic). Session user_id is set from authenticated username."""
    config = db.get_config()
    max_sessions = config.get("max_sessions") or 0
    if max_sessions > 0 and db.count_active_sessions() >= max_sessions:
        raise HTTPException(
            status_code=503,
            detail=f"Max sessions limit reached ({max_sessions}). Try again later.",
        )
    max_rounds = config.get("max_rounds") or 5
    now = time.time()
    session_id = str(uuid.uuid4())
    vision_slot = _effective_vision_model_slot(username)
    db.create_session(session_id, username, max_rounds, now, now, vision_model_slot=vision_slot)
    return SessionResponse(session_id=session_id, user_id=username, max_rounds=max_rounds)


def _max_rounds_for(state: SessionState) -> int:
    return state.get("max_rounds") or db.get_config().get("max_rounds") or 5


@app.get("/sessions/{session_id}", response_model=SessionStatus)
def get_session(session_id: str, _username: str = Depends(verify_game_user)):
    """Get current status and full history for a session. Requires game user auth."""
    state = _get_session(session_id)
    max_rounds = _max_rounds_for(state)
    return SessionStatus(
        session_id=session_id,
        user_id=state.get("user_id"),
        round_number=state["round_number"],
        player_score=state["player_score"],
        server_score=state["server_score"],
        round_history=state["round_history"],
        match_complete=state["round_number"] >= max_rounds,
        winner=state["winner"],
    )


def _run_one_play_round(session_id: str, image: str) -> Optional[dict[str, Any]]:
    """
    Play one round for an active session. Returns response dict, or None if match already complete.
    """
    state = _get_session(session_id)
    now = time.time()
    max_rounds = _max_rounds_for(state)

    if state["round_number"] >= max_rounds:
        return None

    player_move = classify_image(image)
    server_move = random_move()
    round_winner = decide_winner(player_move, server_move)

    player_score = state["player_score"]
    server_score = state["server_score"]
    if round_winner == "player":
        player_score += 1
    elif round_winner == "server":
        server_score += 1

    round_number = state["round_number"] + 1
    round_result = RoundResult(
        round=round_number,
        player_move=player_move,
        server_move=server_move,
        round_winner=round_winner,
        player_score=player_score,
        server_score=server_score,
    )
    round_history = state["round_history"] + [round_result.model_dump()]
    winner = None
    if round_number >= max_rounds:
        if player_score > server_score:
            winner = "player"
        elif server_score > player_score:
            winner = "server"
        else:
            winner = "draw"
        db.record_match_result(winner)

    db.update_session_after_play(
        session_id,
        round_number,
        player_score,
        server_score,
        winner,
        round_history,
        now,
    )

    if round_number >= max_rounds:
        return {
            "match_complete": True,
            "round": round_number,
            "player_move": player_move,
            "server_move": server_move,
            "round_winner": round_winner,
            "player_score": player_score,
            "server_score": server_score,
            "winner": winner,
        }

    return {
        "match_complete": False,
        "round": round_number,
        "player_move": player_move,
        "server_move": server_move,
        "round_winner": round_winner,
        "player_score": player_score,
        "server_score": server_score,
    }


@app.post("/play", response_model=PlayRoundResponse)
def play(req: PlayRequest, _username: str = Depends(verify_game_user)):
    """Play one round in the given session. Requires game user auth."""
    out = _run_one_play_round(req.session_id, req.image)
    if out is None:
        raise HTTPException(
            status_code=400,
            detail="Match already complete. Create a new session to play again.",
        )
    return PlayRoundResponse(**out)


# --- Admin: SPA (dashboard + settings) ---

_ADMIN_HTML_PATH = Path(__file__).parent / "admin.html"


@admin_router.get("", response_class=HTMLResponse)
def admin_app():
    """Serve the admin SPA (Dashboard, Settings, etc.)."""
    return HTMLResponse(content=_ADMIN_HTML_PATH.read_text(encoding="utf-8"))


@admin_router.get("/dashboard", response_class=RedirectResponse)
def admin_dashboard_redirect():
    """Redirect to admin SPA dashboard view."""
    return RedirectResponse(url="/admin#dashboard", status_code=302)


# --- Admin: monitoring APIs ---

@admin_router.get("/monitor/sessions", response_model=list[MonitorSessionSummary])
def admin_list_sessions(include_expired: bool = False):
    """List sessions. By default only active. Set include_expired=true to also return expired sessions (kept for analysis)."""
    result = []
    for sid, state in db.list_sessions():
        expired = _is_session_expired(state)
        if not include_expired and expired:
            continue
        max_rounds = state.get("max_rounds") or 5
        result.append(
            MonitorSessionSummary(
                session_id=sid,
                user_id=state.get("user_id"),
                max_rounds=max_rounds,
                round_number=state["round_number"],
                player_score=state["player_score"],
                server_score=state["server_score"],
                winner=state.get("winner"),
                match_complete=state["round_number"] >= max_rounds,
                created_at=state.get("created_at", 0),
                last_activity_at=state.get("last_activity_at", 0),
                expired=expired,
                vision_model_slot=state.get("vision_model_slot"),
            )
        )
    result.sort(key=lambda x: (x.last_activity_at or 0, x.created_at or 0), reverse=True)
    return result


@admin_router.get("/monitor/win_breakdown_24h", response_model=WinBreakdown24hResponse)
def admin_win_breakdown_24h():
    """Completed matches by winner in the last 24 hours (by session last_activity_at)."""
    since = time.time() - 86400
    b = db.get_win_breakdown_since(since)
    return WinBreakdown24hResponse(
        player_wins=b["player_wins"],
        server_wins=b["server_wins"],
        draws=b["draws"],
    )


@admin_router.get("/monitor/game_stats", response_model=GameStatsResponse)
def admin_game_stats():
    """Aggregate game statistics (sessions, matches, rounds, wins)."""
    stats = db.get_game_stats()
    return GameStatsResponse(
        total_sessions_created=stats["total_sessions_created"],
        total_matches_completed=stats["total_matches_completed"],
        total_rounds_played=stats["total_rounds_played"],
        player_wins=stats["player_wins"],
        server_wins=stats["server_wins"],
        draws=stats["draws"],
        active_sessions=db.count_active_sessions(),
    )


# --- Admin: configuration APIs ---

@admin_router.get("/cfg", response_model=ConfigResponse)
def admin_get_config():
    """Return current server configuration."""
    config = db.get_config()
    return ConfigResponse(
        max_rounds=config.get("max_rounds", 5),
        max_sessions=config.get("max_sessions", 0),
        session_timeout_seconds=config.get("session_timeout_seconds", 0),
        retention_seconds=config.get("retention_seconds", 604800),
        input_modes=_normalize_input_modes(config.get("input_modes") or ["buttons"]),
        vision_ab_rollout_percent=int(config.get("vision_ab_rollout_percent") or 0),
    )


@admin_router.put("/cfg", response_model=ConfigResponse)
def admin_update_config(body: ConfigUpdateRequest):
    """Update configuration. Only provided fields are changed. 0 = unlimited for max_sessions, no timeout for session_timeout_seconds."""
    if body.max_rounds is not None and body.max_rounds < 1:
        raise HTTPException(status_code=400, detail="max_rounds must be at least 1")
    if body.max_sessions is not None and body.max_sessions < 0:
        raise HTTPException(status_code=400, detail="max_sessions cannot be negative")
    if body.session_timeout_seconds is not None and body.session_timeout_seconds < 0:
        raise HTTPException(status_code=400, detail="session_timeout_seconds cannot be negative")
    if body.retention_seconds is not None and body.retention_seconds < 0:
        raise HTTPException(status_code=400, detail="retention_seconds cannot be negative")
    if body.vision_ab_rollout_percent is not None and not 0 <= body.vision_ab_rollout_percent <= 100:
        raise HTTPException(
            status_code=400,
            detail="vision_ab_rollout_percent must be between 0 and 100",
        )
    input_modes_norm: Optional[list[str]] = None
    if body.input_modes is not None:
        if not body.input_modes:
            raise HTTPException(status_code=400, detail="input_modes cannot be empty")
        unknown = [m for m in body.input_modes if m not in _VALID_INPUT_MODES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input_modes: {unknown}. Use buttons, vision, audio.",
            )
        input_modes_norm = _normalize_input_modes(body.input_modes)
    db.set_config(
        max_rounds=body.max_rounds,
        max_sessions=body.max_sessions,
        session_timeout_seconds=body.session_timeout_seconds,
        retention_seconds=body.retention_seconds,
        input_modes=input_modes_norm,
        vision_ab_rollout_percent=body.vision_ab_rollout_percent,
    )
    return admin_get_config()


@admin_router.post("/sessions/prune", response_model=PruneResponse)
def admin_prune_sessions(body: Optional[PruneRequest] = None):
    """Delete expired sessions older than retention period. Uses config retention_seconds unless overridden in body."""
    pruned = db.prune_expired_sessions(body.retention_seconds if body else None)
    return PruneResponse(pruned_count=pruned)


app.include_router(admin_router)

# --- Web game client (SPA): registered after /admin so paths stay predictable.
# /admin/game is a public alias for setups where the proxy routes /admin/* (and /docs) but not /game.

_GAME_HTML_PATH = Path(__file__).parent / "game.html"


@app.get("/game", response_class=HTMLResponse)
@app.get("/admin/game", response_class=HTMLResponse, include_in_schema=False)
def game_spa_get():
    """Rock Paper Scissors web client: login and play with rock/paper/scissors/none tiles."""
    return HTMLResponse(content=_GAME_HTML_PATH.read_text(encoding="utf-8"))


@app.head("/game")
@app.head("/admin/game", include_in_schema=False)
def game_spa_head():
    return Response(status_code=200, media_type="text/html")
