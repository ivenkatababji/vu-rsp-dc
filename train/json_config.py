"""Load JSON training/deploy configs; resolve paths relative to the config file directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_dict(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{config_path}: root must be a JSON object")
    return raw, config_path.parent


def config_path(value: Any, base: Path, *, key: str) -> Path:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SystemExit(f"Config missing or empty required string: {key!r}")
    p = Path(str(value).strip()).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def config_path_optional(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    p = Path(str(value).strip()).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()
