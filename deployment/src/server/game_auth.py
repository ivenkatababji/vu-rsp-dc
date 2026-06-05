"""
Game user authentication: pre-provisioned users from config file.
Valid credentials are required to create sessions and play.
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

USERS_CONFIG_PATH = Path(__file__).parent / "users_config.json"

# Default user when no config file exists
DEFAULT_USERS = {"guest": "guest"}

_security = HTTPBasic()
_cached: Optional[tuple[str, dict[str, str]]] = None


def _normalize_users(data: object) -> dict[str, str]:
    if isinstance(data, dict):
        users = {str(k).strip(): str(v) for k, v in data.items() if k and v is not None}
        if users:
            return users
    return dict(DEFAULT_USERS)


def _load_users() -> dict[str, str]:
    """Load username -> password map from users_config.json. Uses DEFAULT_USERS if file missing."""
    global _cached
    env_users_json = (os.getenv("RPS_USERS_JSON") or "").strip()
    if env_users_json:
        cache_key = f"env:{env_users_json}"
        if _cached is not None and _cached[0] == cache_key:
            return _cached[1]
        try:
            users = _normalize_users(json.loads(env_users_json))
        except json.JSONDecodeError:
            users = dict(DEFAULT_USERS)
        _cached = (cache_key, users)
        return users

    if not USERS_CONFIG_PATH.exists():
        users = dict(DEFAULT_USERS)
        _cached = ("default", users)
        return users

    try:
        data = json.loads(USERS_CONFIG_PATH.read_text(encoding="utf-8"))
        users = _normalize_users(data)
    except (json.JSONDecodeError, OSError):
        users = dict(DEFAULT_USERS)

    _cached = (f"file:{USERS_CONFIG_PATH}", users)
    return users


def verify_game_user(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    """
    Validate game user credentials. Returns the username if valid.
    Raises 401 if invalid. Uses users_config.json; falls back to guest/guest if file missing.
    """
    users = _load_users()
    username = (credentials.username or "").strip()
    password = credentials.password or ""
    if not username or users.get(username) != password:
        # Do not send WWW-Authenticate: browsers show a native Basic Auth dialog on 401+Basic
        # when the page used fetch() with Authorization — the game UI handles errors in-page.
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return username
