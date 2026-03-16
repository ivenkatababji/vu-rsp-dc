"""
Admin authentication: load credentials from config file and verify HTTP Basic Auth.
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ADMIN_CONFIG_PATH = Path(__file__).parent / "admin_config.json"

_security = HTTPBasic()
_cached: Optional[tuple[str, str]] = None


def _load_credentials() -> Optional[tuple[str, str]]:
    global _cached
    if _cached is not None:
        return _cached
    if not ADMIN_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(ADMIN_CONFIG_PATH.read_text(encoding="utf-8"))
        username = (data.get("admin_username") or "").strip()
        password = (data.get("admin_password") or "").strip()
        if username and password:
            _cached = (username, password)
            return _cached
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
