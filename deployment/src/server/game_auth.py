"""
Game user authentication: pre-provisioned users from config file.
Valid credentials are required to create sessions and play.
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

USERS_CONFIG_PATH = Path(__file__).parent / "users_config.json"

# Default user when no config file exists
DEFAULT_USERS = {"guest": "guest"}

_security = HTTPBasic()
_cached: Optional[dict[str, str]] = None


def _load_users() -> dict[str, str]:
    """Load username -> password map from users_config.json. Uses DEFAULT_USERS if file missing."""
    global _cached
    if _cached is not None:
        return _cached
    if not USERS_CONFIG_PATH.exists():
        _cached = dict(DEFAULT_USERS)
        return _cached
    try:
        data = json.loads(USERS_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _cached = {str(k).strip(): str(v) for k, v in data.items() if k and v is not None}
        else:
            _cached = dict(DEFAULT_USERS)
    except (json.JSONDecodeError, OSError):
        _cached = dict(DEFAULT_USERS)
    return _cached


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
