"""
Build client ML bundle: ONNX manifests for vision/audio, optional sidecar JSON per modality.
Place models at ml_artifacts/{vision,audio}/model.onnx and optional manifest.json overrides.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

ARTIFACTS_ROOT = Path(__file__).parent / "ml_artifacts"
VISION_DIR = ARTIFACTS_ROOT / "vision"
AUDIO_DIR = ARTIFACTS_ROOT / "audio"

# Pin ORT Web so client wasm matches loaded script (bump together when upgrading).
ONNX_RUNTIME_WEB_VERSION = "1.20.1"
ORT_CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ONNX_RUNTIME_WEB_VERSION}/dist"

DEFAULT_LABELS = ["rock", "paper", "scissors", "none"]
_SERVER_DIR = Path(__file__).parent


def _resolve_default_vision_hw() -> tuple[int, int]:
    """
    Defaults for manifest hints when ml_artifacts/.../manifest.json has no input block.
    Override with server_config.json (vision_input_size or vision_input_width/height)
    or env RPS_VISION_INPUT_SIZE or RPS_VISION_INPUT_WIDTH + HEIGHT.
    """
    cfg_path = _SERVER_DIR / "server_config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if data.get("vision_input_width") is not None and data.get("vision_input_height") is not None:
                return int(data["vision_input_width"]), int(data["vision_input_height"])
            if data.get("vision_input_size") is not None:
                s = int(data["vision_input_size"])
                return s, s
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    ew, eh = os.environ.get("RPS_VISION_INPUT_WIDTH"), os.environ.get("RPS_VISION_INPUT_HEIGHT")
    if ew and eh:
        try:
            return int(ew), int(eh)
        except ValueError:
            pass
    es = os.environ.get("RPS_VISION_INPUT_SIZE")
    if es:
        try:
            s = int(es)
            return s, s
        except ValueError:
            pass
    return 224, 224


def get_default_vision_input() -> dict[str, Any]:
    w, h = _resolve_default_vision_hw()
    return {
        "name": "input",
        "width": w,
        "height": h,
        "layout": "NCHW",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def _load_sidecar(dir_path: Path) -> dict[str, Any]:
    p = dir_path / "manifest.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _merge_input(side: dict[str, Any]) -> dict[str, Any]:
    inp = dict(get_default_vision_input())
    if "input" in side and isinstance(side["input"], dict):
        inp.update(side["input"])
    return inp


def build_vision_manifest() -> dict[str, Any]:
    side = _load_sidecar(VISION_DIR)
    model_path = VISION_DIR / "model.onnx"
    labels = side.get("labels")
    if not isinstance(labels, list) or not labels:
        labels = list(DEFAULT_LABELS)
    else:
        labels = [str(x) for x in labels]
    inp = _merge_input(side)
    out: dict[str, Any] = {
        "runtime": "onnx-web",
        "labels": labels,
        "input": inp,
        "output": side.get("output") if isinstance(side.get("output"), dict) else {},
    }
    if model_path.is_file():
        raw = model_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        version = str(side.get("version") or digest[:20])
        out.update(
            {
                "available": True,
                "version": version,
                "sha256": digest,
                "model_url": "/me/ml/models/vision",
            }
        )
    else:
        out.update(
            {
                "available": False,
                "version": "none",
                "sha256": None,
                "model_url": None,
            }
        )
    return out


def build_audio_manifest() -> dict[str, Any]:
    side = _load_sidecar(AUDIO_DIR)
    model_path = AUDIO_DIR / "model.onnx"
    bs = side.get("browser_speech") if isinstance(side.get("browser_speech"), dict) else {}
    browser_enabled = bs.get("enabled", True)
    if isinstance(browser_enabled, str):
        browser_enabled = browser_enabled.lower() in ("1", "true", "yes")
    locale = str(bs.get("locale") or "en-US")

    labels = side.get("labels")
    if not isinstance(labels, list) or not labels:
        labels = list(DEFAULT_LABELS)
    else:
        labels = [str(x) for x in labels]
    inp = _merge_input(side)

    out: dict[str, Any] = {
        "runtime": "onnx-web",
        "labels": labels,
        "input": inp,
        "output": side.get("output") if isinstance(side.get("output"), dict) else {},
        "browser_speech": {
            "enabled": bool(browser_enabled),
            "locale": locale,
        },
    }
    if model_path.is_file():
        raw = model_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        version = str(side.get("version") or digest[:20])
        out["onnx"] = {
            "available": True,
            "version": version,
            "sha256": digest,
            "model_url": "/me/ml/models/audio",
        }
    else:
        out["onnx"] = {
            "available": False,
            "version": "none",
            "sha256": None,
            "model_url": None,
        }
    return out


def build_ml_bundle(input_modes: list[str]) -> dict[str, Any]:
    return {
        "input_modes": list(input_modes),
        "vision": build_vision_manifest(),
        "audio": build_audio_manifest(),
        "onnx_runtime_web": {
            "version": ONNX_RUNTIME_WEB_VERSION,
            "ort_min_js": f"{ORT_CDN}/ort.min.js",
            "wasm_base": f"{ORT_CDN}/",
        },
    }


def model_file_for_kind(kind: str) -> Optional[Path]:
    if kind == "vision":
        p = VISION_DIR / "model.onnx"
        return p if p.is_file() else None
    if kind == "audio":
        p = AUDIO_DIR / "model.onnx"
        return p if p.is_file() else None
    return None
