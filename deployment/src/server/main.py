import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from admin_auth import verify_admin
from classifier import classify_image
from game import random_move, decide_winner
import db

app = FastAPI(title="Rock Paper Scissors ML Experiment")

# Admin routes: protected by HTTP Basic Auth (credentials in admin_config.json)
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(verify_admin)])

# Per-session state (from DB): { session_id?, user_id, max_rounds, round_number, ... }
SessionState = dict


@app.on_event("startup")
def startup():
    db.init_db()


class PlayRequest(BaseModel):
    session_id: str
    image: str


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None


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


class GameStatsResponse(BaseModel):
    total_sessions_created: int
    total_matches_completed: int
    total_rounds_played: int
    player_wins: int
    server_wins: int
    draws: int
    active_sessions: int


class ConfigResponse(BaseModel):
    max_rounds: int
    max_sessions: int
    session_timeout_seconds: int


class ConfigUpdateRequest(BaseModel):
    max_rounds: Optional[int] = None
    max_sessions: Optional[int] = None
    session_timeout_seconds: Optional[int] = None


def _is_session_expired(state: SessionState) -> bool:
    config = db.get_config()
    timeout = config.get("session_timeout_seconds") or 0
    if timeout <= 0:
        return False
    last = state.get("last_activity_at") or state.get("created_at") or 0
    return (time.time() - last) > timeout


def _get_session(session_id: str) -> SessionState:
    state = db.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if _is_session_expired(state):
        db.delete_session(session_id)
        raise HTTPException(status_code=404, detail="Session expired")
    state["session_id"] = session_id
    return state


@app.post("/sessions", response_model=SessionResponse)
def create_session(body: Optional[CreateSessionRequest] = None):
    """Create a new game session. Optionally pass user_id in the request body. Use the returned session_id for /play."""
    db.evict_expired_sessions()
    config = db.get_config()
    max_sessions = config.get("max_sessions") or 0
    if max_sessions > 0 and db.count_sessions() >= max_sessions:
        raise HTTPException(
            status_code=503,
            detail=f"Max sessions limit reached ({max_sessions}). Try again later.",
        )
    user_id = body.user_id if body else None
    max_rounds = config.get("max_rounds") or 5
    now = time.time()
    session_id = str(uuid.uuid4())
    db.create_session(session_id, user_id, max_rounds, now, now)
    return SessionResponse(session_id=session_id, user_id=user_id)


def _max_rounds_for(state: SessionState) -> int:
    return state.get("max_rounds") or db.get_config().get("max_rounds") or 5


@app.get("/sessions/{session_id}", response_model=SessionStatus)
def get_session(session_id: str):
    """Get current status and full history for a session."""
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


@app.post("/play")
def play(req: PlayRequest):
    """Play one round in the given session. Match is complete after max_rounds (config)."""
    state = _get_session(req.session_id)
    now = time.time()
    max_rounds = _max_rounds_for(state)

    if state["round_number"] >= max_rounds:
        raise HTTPException(
            status_code=400,
            detail="Match already complete. Create a new session to play again.",
        )

    player_move = classify_image(req.image)
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
        req.session_id,
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
def admin_list_sessions():
    """List all sessions (active and expired). Expired sessions are still shown with expired=true until evicted."""
    db.evict_expired_sessions()
    result = []
    for sid, state in db.list_sessions():
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
                expired=_is_session_expired(state),
            )
        )
    return result


@admin_router.get("/monitor/game_stats", response_model=GameStatsResponse)
def admin_game_stats():
    """Aggregate game statistics (sessions, matches, rounds, wins)."""
    db.evict_expired_sessions()
    stats = db.get_game_stats()
    return GameStatsResponse(
        total_sessions_created=stats["total_sessions_created"],
        total_matches_completed=stats["total_matches_completed"],
        total_rounds_played=stats["total_rounds_played"],
        player_wins=stats["player_wins"],
        server_wins=stats["server_wins"],
        draws=stats["draws"],
        active_sessions=db.count_sessions(),
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
    db.set_config(
        max_rounds=body.max_rounds,
        max_sessions=body.max_sessions,
        session_timeout_seconds=body.session_timeout_seconds,
    )
    return admin_get_config()


app.include_router(admin_router)
