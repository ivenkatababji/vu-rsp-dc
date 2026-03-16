import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from classifier import classify_image
from game import random_move, decide_winner

app = FastAPI(title="Rock Paper Scissors ML Experiment")

# Per-session state: session_id -> { user_id, max_rounds, round_number, player_score, server_score, round_history, winner, created_at, last_activity_at }
SessionState = dict
sessions: dict[str, SessionState] = {}

# In-memory config (0 = unlimited for max_sessions; 0 = no timeout for session_timeout_seconds)
config: dict[str, Any] = {
    "max_rounds": 5,
    "max_sessions": 0,
    "session_timeout_seconds": 0,
}

# Aggregate game statistics
game_stats: dict[str, int] = {
    "total_sessions_created": 0,
    "total_matches_completed": 0,
    "total_rounds_played": 0,
    "player_wins": 0,
    "server_wins": 0,
    "draws": 0,
}


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
    timeout = config.get("session_timeout_seconds") or 0
    if timeout <= 0:
        return False
    last = state.get("last_activity_at") or state.get("created_at") or 0
    return (time.time() - last) > timeout


def _evict_expired_sessions() -> None:
    to_remove = [sid for sid, state in sessions.items() if _is_session_expired(state)]
    for sid in to_remove:
        del sessions[sid]


def _get_session(session_id: str) -> SessionState:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[session_id]
    if _is_session_expired(state):
        del sessions[session_id]
        raise HTTPException(status_code=404, detail="Session expired")
    return state


@app.post("/sessions", response_model=SessionResponse)
def create_session(body: Optional[CreateSessionRequest] = None):
    """Create a new game session. Optionally pass user_id in the request body. Use the returned session_id for /play."""
    _evict_expired_sessions()
    max_sessions = config.get("max_sessions") or 0
    if max_sessions > 0 and len(sessions) >= max_sessions:
        raise HTTPException(
            status_code=503,
            detail=f"Max sessions limit reached ({max_sessions}). Try again later.",
        )
    user_id = body.user_id if body else None
    max_rounds = config.get("max_rounds") or 5
    now = time.time()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_id": user_id,
        "max_rounds": max_rounds,
        "round_number": 0,
        "player_score": 0,
        "server_score": 0,
        "round_history": [],
        "winner": None,
        "created_at": now,
        "last_activity_at": now,
    }
    game_stats["total_sessions_created"] += 1
    return SessionResponse(session_id=session_id, user_id=user_id)


def _max_rounds_for(state: SessionState) -> int:
    return state.get("max_rounds") or config.get("max_rounds") or 5


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
    state["last_activity_at"] = time.time()
    max_rounds = _max_rounds_for(state)

    if state["round_number"] >= max_rounds:
        raise HTTPException(
            status_code=400,
            detail="Match already complete. Create a new session to play again.",
        )

    player_move = classify_image(req.image)
    server_move = random_move()
    round_winner = decide_winner(player_move, server_move)

    if round_winner == "player":
        state["player_score"] += 1
    elif round_winner == "server":
        state["server_score"] += 1

    state["round_number"] += 1
    game_stats["total_rounds_played"] += 1
    round_result = RoundResult(
        round=state["round_number"],
        player_move=player_move,
        server_move=server_move,
        round_winner=round_winner,
        player_score=state["player_score"],
        server_score=state["server_score"],
    )
    state["round_history"].append(round_result.model_dump())

    if state["round_number"] >= max_rounds:
        if state["player_score"] > state["server_score"]:
            state["winner"] = "player"
            game_stats["player_wins"] += 1
        elif state["server_score"] > state["player_score"]:
            state["winner"] = "server"
            game_stats["server_wins"] += 1
        else:
            state["winner"] = "draw"
            game_stats["draws"] += 1
        game_stats["total_matches_completed"] += 1

        return {
            "match_complete": True,
            "round": state["round_number"],
            "player_move": player_move,
            "server_move": server_move,
            "round_winner": round_winner,
            "player_score": state["player_score"],
            "server_score": state["server_score"],
            "winner": state["winner"],
        }

    return {
        "match_complete": False,
        "round": state["round_number"],
        "player_move": player_move,
        "server_move": server_move,
        "round_winner": round_winner,
        "player_score": state["player_score"],
        "server_score": state["server_score"],
    }


# --- Admin: monitoring APIs ---

@app.get("/admin/monitor/sessions", response_model=list[MonitorSessionSummary])
def admin_list_sessions():
    """List all sessions (active and expired). Expired sessions are still shown with expired=true until evicted."""
    _evict_expired_sessions()
    result = []
    for sid, state in sessions.items():
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


@app.get("/admin/monitor/game_stats", response_model=GameStatsResponse)
def admin_game_stats():
    """Aggregate game statistics (sessions, matches, rounds, wins)."""
    _evict_expired_sessions()
    return GameStatsResponse(
        total_sessions_created=game_stats["total_sessions_created"],
        total_matches_completed=game_stats["total_matches_completed"],
        total_rounds_played=game_stats["total_rounds_played"],
        player_wins=game_stats["player_wins"],
        server_wins=game_stats["server_wins"],
        draws=game_stats["draws"],
        active_sessions=len(sessions),
    )


# --- Admin: configuration APIs ---

@app.get("/admin/cfg", response_model=ConfigResponse)
def admin_get_config():
    """Return current server configuration."""
    return ConfigResponse(
        max_rounds=config.get("max_rounds", 5),
        max_sessions=config.get("max_sessions", 0),
        session_timeout_seconds=config.get("session_timeout_seconds", 0),
    )


@app.put("/admin/cfg", response_model=ConfigResponse)
def admin_update_config(body: ConfigUpdateRequest):
    """Update configuration. Only provided fields are changed. 0 = unlimited for max_sessions, no timeout for session_timeout_seconds."""
    if body.max_rounds is not None:
        if body.max_rounds < 1:
            raise HTTPException(status_code=400, detail="max_rounds must be at least 1")
        config["max_rounds"] = body.max_rounds
    if body.max_sessions is not None:
        if body.max_sessions < 0:
            raise HTTPException(status_code=400, detail="max_sessions cannot be negative")
        config["max_sessions"] = body.max_sessions
    if body.session_timeout_seconds is not None:
        if body.session_timeout_seconds < 0:
            raise HTTPException(status_code=400, detail="session_timeout_seconds cannot be negative")
        config["session_timeout_seconds"] = body.session_timeout_seconds
    return admin_get_config()
