"""
Admin authentication: load credentials from config file and verify HTTP Basic Auth.
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ADMIN_CONFIG_PATH = Path(__file__).parent / "admin_config.json"

_security = HTTPBasic()
_cached: Optional[tuple[str, Optional[tuple[str, str]]]] = None


def _load_credentials() -> Optional[tuple[str, str]]:
    global _cached
    env_username = (os.getenv("RPS_ADMIN_USERNAME") or "").strip()
    env_password = (os.getenv("RPS_ADMIN_PASSWORD") or "").strip()
    if env_username and env_password:
        cache_key = f"env:{env_username}:{env_password}"
        creds = (env_username, env_password)
        if _cached is not None and _cached[0] == cache_key:
            return _cached[1]
        _cached = (cache_key, creds)
        return creds

    if not ADMIN_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(ADMIN_CONFIG_PATH.read_text(encoding="utf-8"))
        username = (data.get("admin_username") or "").strip()
        password = (data.get("admin_password") or "").strip()
        if username and password:
            creds = (username, password)
            _cached = (f"file:{ADMIN_CONFIG_PATH}", creds)
            return creds
    except (json.JSONDecodeError, OSError):
        pass
    return None


def verify_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """Validate admin HTTP Basic credentials against admin_config.json. Raises 401 if invalid or not configured."""
    creds = _load_credentials()
    if not creds:
        raise HTTPException(
            status_code=503,
            detail="Admin not configured. Create admin_config.json from admin_config.json.example with admin_username and admin_password.",
        )
    username, password = creds
    if not (
        credentials.username == username
        and credentials.password == password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
