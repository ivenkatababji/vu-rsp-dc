"""
SQLite-backed state for sessions, config, and game stats.
Use :memory: for in-memory DB (single connection).
Access is serialized with a lock to avoid "database is locked" under concurrent requests.
"""
import json
import sqlite3
import threading
import time
from typing import Any, Optional

# Set by init_db() from server config; default in-memory.
DB_PATH = ":memory:"

_conn: Optional[sqlite3.Connection] = None
_lock = threading.RLock()


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA busy_timeout = 10000")  # wait up to 10s if locked
        if DB_PATH != ":memory:":
            _conn.execute("PRAGMA journal_mode = WAL")
    return _conn


def init_db(path: Optional[str] = None) -> None:
    """Create tables and seed default config and game_stats. path: SQLite DB path (e.g. ':memory:' or 'state.db')."""
    global DB_PATH
    if path is not None:
        DB_PATH = path
    with _lock:
        conn = get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            max_rounds INTEGER NOT NULL DEFAULT 5,
            max_sessions INTEGER NOT NULL DEFAULT 10,
            session_timeout_seconds INTEGER NOT NULL DEFAULT 0,
            retention_seconds INTEGER NOT NULL DEFAULT 604800
        );
        INSERT OR IGNORE INTO config (id, max_rounds, max_sessions, session_timeout_seconds, retention_seconds)
        VALUES (1, 5, 10, 0, 604800);
        """)
        try:
            conn.execute("ALTER TABLE config ADD COLUMN retention_seconds INTEGER NOT NULL DEFAULT 604800")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "ALTER TABLE config ADD COLUMN input_modes_json TEXT NOT NULL DEFAULT '[\"buttons\"]'"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "ALTER TABLE config ADD COLUMN vision_ab_rollout_percent INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        conn.executescript("""

        CREATE TABLE IF NOT EXISTS game_stats (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_sessions_created INTEGER NOT NULL DEFAULT 0,
            total_matches_completed INTEGER NOT NULL DEFAULT 0,
            total_rounds_played INTEGER NOT NULL DEFAULT 0,
            player_wins INTEGER NOT NULL DEFAULT 0,
            server_wins INTEGER NOT NULL DEFAULT 0,
            draws INTEGER NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO game_stats (id) VALUES (1);

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            max_rounds INTEGER NOT NULL,
            round_number INTEGER NOT NULL DEFAULT 0,
            player_score INTEGER NOT NULL DEFAULT 0,
            server_score INTEGER NOT NULL DEFAULT 0,
            winner TEXT,
            created_at REAL NOT NULL,
            last_activity_at REAL NOT NULL,
            round_history TEXT NOT NULL DEFAULT '[]'
        );
        """)
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN vision_model_slot TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_vision_state (
            user_id TEXT PRIMARY KEY,
            vision_slot TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """)
        conn.commit()


def _parse_input_modes(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return ["buttons"]
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except (json.JSONDecodeError, TypeError):
        pass
    return ["buttons"]


def get_config() -> dict[str, Any]:
    with _lock:
        conn = get_conn()
        try:
            row = conn.execute(
                """SELECT max_rounds, max_sessions, session_timeout_seconds, retention_seconds,
                          input_modes_json, vision_ab_rollout_percent FROM config WHERE id = 1"""
            ).fetchone()
        except sqlite3.OperationalError:
            try:
                row = conn.execute(
                    "SELECT max_rounds, max_sessions, session_timeout_seconds, retention_seconds, input_modes_json FROM config WHERE id = 1"
                ).fetchone()
            except sqlite3.OperationalError:
                try:
                    row = conn.execute(
                        "SELECT max_rounds, max_sessions, session_timeout_seconds, retention_seconds FROM config WHERE id = 1"
                    ).fetchone()
                except sqlite3.OperationalError:
                    row = conn.execute(
                        "SELECT max_rounds, max_sessions, session_timeout_seconds FROM config WHERE id = 1"
                    ).fetchone()
        if not row:
            return {
                "max_rounds": 5,
                "max_sessions": 10,
                "session_timeout_seconds": 0,
                "retention_seconds": 604800,
                "input_modes": ["buttons"],
                "vision_ab_rollout_percent": 0,
            }
        out = {
            "max_rounds": row["max_rounds"],
            "max_sessions": row["max_sessions"],
            "session_timeout_seconds": row["session_timeout_seconds"],
        }
        try:
            out["retention_seconds"] = row["retention_seconds"]
        except (KeyError, IndexError):
            out["retention_seconds"] = 604800
        try:
            out["input_modes"] = _parse_input_modes(row["input_modes_json"])
        except (KeyError, IndexError):
            out["input_modes"] = ["buttons"]
        try:
            out["vision_ab_rollout_percent"] = int(row["vision_ab_rollout_percent"] or 0)
        except (KeyError, IndexError):
            out["vision_ab_rollout_percent"] = 0
        return out


def set_config(
    max_rounds: Optional[int] = None,
    max_sessions: Optional[int] = None,
    session_timeout_seconds: Optional[int] = None,
    retention_seconds: Optional[int] = None,
    input_modes: Optional[list[str]] = None,
    vision_ab_rollout_percent: Optional[int] = None,
) -> None:
    conn = get_conn()
    updates = []
    params = []
    if max_rounds is not None:
        updates.append("max_rounds = ?")
        params.append(max_rounds)
    if max_sessions is not None:
        updates.append("max_sessions = ?")
        params.append(max_sessions)
    if session_timeout_seconds is not None:
        updates.append("session_timeout_seconds = ?")
        params.append(session_timeout_seconds)
    if retention_seconds is not None:
        updates.append("retention_seconds = ?")
        params.append(retention_seconds)
    if input_modes is not None:
        updates.append("input_modes_json = ?")
        params.append(json.dumps(input_modes))
    if vision_ab_rollout_percent is not None:
        updates.append("vision_ab_rollout_percent = ?")
        params.append(vision_ab_rollout_percent)
    if updates:
        params.append(1)
        with _lock:
            conn = get_conn()
            conn.execute(f"UPDATE config SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    vms = None
    if "vision_model_slot" in keys and row["vision_model_slot"] is not None:
        vms = str(row["vision_model_slot"])
    return {
        "user_id": row["user_id"],
        "max_rounds": row["max_rounds"],
        "round_number": row["round_number"],
        "player_score": row["player_score"],
        "server_score": row["server_score"],
        "winner": row["winner"],
        "created_at": row["created_at"],
        "last_activity_at": row["last_activity_at"],
        "round_history": json.loads(row["round_history"] or "[]"),
        "vision_model_slot": vms,
    }


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        state = _row_to_session(row)
        state["session_id"] = row["session_id"]
        return state


def list_sessions() -> list[tuple[str, dict[str, Any]]]:
    with _lock:
        conn = get_conn()
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        return [(row["session_id"], _row_to_session(row)) for row in rows]


def get_user_vision_slot(user_id: str) -> Optional[str]:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT vision_slot FROM user_vision_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return str(row["vision_slot"]) if row else None


def set_user_vision_slot(user_id: str, slot: str) -> None:
    now = time.time()
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO user_vision_state (user_id, vision_slot, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 vision_slot = excluded.vision_slot,
                 updated_at = excluded.updated_at""",
            (user_id, slot, now),
        )
        conn.commit()


def create_session(
    session_id: str,
    user_id: Optional[str],
    max_rounds: int,
    created_at: float,
    last_activity_at: float,
    vision_model_slot: Optional[str] = None,
) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO sessions (session_id, user_id, max_rounds, round_number, player_score, server_score, winner, created_at, last_activity_at, round_history, vision_model_slot)
               VALUES (?, ?, ?, 0, 0, 0, NULL, ?, ?, '[]', ?)""",
            (session_id, user_id, max_rounds, created_at, last_activity_at, vision_model_slot),
        )
        conn.execute(
            "UPDATE game_stats SET total_sessions_created = total_sessions_created + 1 WHERE id = 1"
        )
        conn.commit()


def update_session_after_play(
    session_id: str,
    round_number: int,
    player_score: int,
    server_score: int,
    winner: Optional[str],
    round_history: list[dict],
    last_activity_at: float,
) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """UPDATE sessions SET round_number = ?, player_score = ?, server_score = ?, winner = ?, round_history = ?, last_activity_at = ?
               WHERE session_id = ?""",
            (round_number, player_score, server_score, winner, json.dumps(round_history), last_activity_at, session_id),
        )
        conn.execute(
            "UPDATE game_stats SET total_rounds_played = total_rounds_played + 1 WHERE id = 1"
        )
        conn.commit()


def record_match_result(winner: str) -> None:
    """winner is 'player', 'server', or 'draw'."""
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE game_stats SET total_matches_completed = total_matches_completed + 1 WHERE id = 1"
        )
        if winner == "player":
            conn.execute("UPDATE game_stats SET player_wins = player_wins + 1 WHERE id = 1")
        elif winner == "server":
            conn.execute("UPDATE game_stats SET server_wins = server_wins + 1 WHERE id = 1")
        else:
            conn.execute("UPDATE game_stats SET draws = draws + 1 WHERE id = 1")
        conn.commit()


def delete_session(session_id: str) -> None:
    with _lock:
        conn = get_conn()
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def count_sessions() -> int:
    with _lock:
        conn = get_conn()
        row = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return row["n"] if row else 0


def count_active_sessions() -> int:
    """Count sessions that are not yet expired (within effective timeout)."""
    timeout = get_effective_session_timeout_seconds()
    if timeout <= 0:
        return count_sessions()
    now = time.time()
    with _lock:
        conn = get_conn()
        rows = conn.execute(
            "SELECT session_id, last_activity_at, created_at FROM sessions"
        ).fetchall()
        n = 0
        for row in rows:
            last = row["last_activity_at"] or row["created_at"] or 0
            if (now - last) <= timeout:
                n += 1
        return n


def get_effective_session_timeout_seconds() -> int:
    """Session expiry in seconds. When config.session_timeout_seconds is 0, returns 30 * max_rounds."""
    cfg = get_config()
    timeout = cfg.get("session_timeout_seconds") or 0
    if timeout > 0:
        return timeout
    return 30 * (cfg.get("max_rounds") or 5)


def evict_expired_sessions() -> None:
    """No-op: expired sessions are kept for analysis. Use prune_expired_sessions() to delete by retention."""


def prune_expired_sessions(retention_seconds: Optional[int] = None) -> int:
    """Delete sessions that expired more than retention_seconds ago. Returns number pruned."""
    cfg = get_config()
    retention = retention_seconds if retention_seconds is not None else (cfg.get("retention_seconds") or 604800)
    timeout = get_effective_session_timeout_seconds()
    cutoff = time.time() - timeout - retention
    with _lock:
        conn = get_conn()
        rows = conn.execute(
            "SELECT session_id, last_activity_at, created_at FROM sessions"
        ).fetchall()
        deleted = 0
        for row in rows:
            last = row["last_activity_at"] or row["created_at"] or 0
            if last < cutoff:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (row["session_id"],))
                deleted += 1
        conn.commit()
        return deleted


def get_game_stats() -> dict[str, int]:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            """SELECT total_sessions_created, total_matches_completed, total_rounds_played,
                      player_wins, server_wins, draws FROM game_stats WHERE id = 1"""
        ).fetchone()
        if not row:
            return {
                "total_sessions_created": 0,
                "total_matches_completed": 0,
                "total_rounds_played": 0,
                "player_wins": 0,
                "server_wins": 0,
                "draws": 0,
            }
        return dict(row)


def get_user_stats(user_id: str) -> dict[str, int]:
    """Aggregate stats for one game user from retained sessions rows."""
    with _lock:
        conn = get_conn()
        row = conn.execute(
            """
            SELECT
              COUNT(*) AS sessions_started,
              COALESCE(SUM(CASE WHEN winner IS NOT NULL THEN 1 ELSE 0 END), 0) AS matches_completed,
              COALESCE(SUM(CASE WHEN winner = 'player' THEN 1 ELSE 0 END), 0) AS matches_won,
              COALESCE(SUM(CASE WHEN winner = 'server' THEN 1 ELSE 0 END), 0) AS matches_lost,
              COALESCE(SUM(CASE WHEN winner = 'draw' THEN 1 ELSE 0 END), 0) AS matches_draw,
              COALESCE(SUM(CASE WHEN winner IS NOT NULL THEN round_number ELSE 0 END), 0) AS rounds_played
            FROM sessions
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        return {
            "sessions_started": 0,
            "matches_completed": 0,
            "matches_won": 0,
            "matches_lost": 0,
            "matches_draw": 0,
            "rounds_played": 0,
        }
    return {
        "sessions_started": int(row["sessions_started"] or 0),
        "matches_completed": int(row["matches_completed"] or 0),
        "matches_won": int(row["matches_won"] or 0),
        "matches_lost": int(row["matches_lost"] or 0),
        "matches_draw": int(row["matches_draw"] or 0),
        "rounds_played": int(row["rounds_played"] or 0),
    }


def get_win_breakdown_since(since_timestamp: float) -> dict[str, int]:
    """Count completed matches by winner since since_timestamp (Unix epoch)."""
    with _lock:
        conn = get_conn()
        rows = conn.execute(
            """
            SELECT winner, COUNT(*) AS c FROM sessions
            WHERE winner IS NOT NULL AND last_activity_at >= ?
            GROUP BY winner
            """,
            (since_timestamp,),
        ).fetchall()
    out = {"player_wins": 0, "server_wins": 0, "draws": 0}
    for row in rows:
        w = row["winner"]
        c = row["c"]
        if w == "player":
            out["player_wins"] = c
        elif w == "server":
            out["server_wins"] = c
        elif w == "draw":
            out["draws"] = c
    return out
