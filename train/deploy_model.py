#!/usr/bin/env python3
"""Copy exported model.onnx (+ manifest.json) into the FastAPI server's ml_artifacts/vision/.

Paths come from a JSON config file; pass its path as the only CLI argument.
See deploy_config.example.json. Relative paths are resolved from the config file's directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from json_config import config_path, config_path_optional, load_config_dict


def main() -> None:
    here = Path(__file__).resolve().parent
    default_dest = here.parent / "deployment" / "src" / "server" / "ml_artifacts" / "vision"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config",
        type=Path,
        help="JSON config (see deploy_config.example.json)",
    )
    cli = ap.parse_args()
    if not cli.config.is_file():
        raise SystemExit(f"Config not found: {cli.config}")

    cfg, base = load_config_dict(cli.config)
    export_dir = config_path(cfg.get("export_dir"), base, key="export_dir")
    dest = config_path_optional(cfg.get("dest"), base) or default_dest

    src_onnx = export_dir / "model.onnx"
    if not src_onnx.is_file():
        raise SystemExit(f"Missing {src_onnx}")

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_onnx, dest / "model.onnx")
    src_man = export_dir / "manifest.json"
    if src_man.is_file():
        shutil.copy2(src_man, dest / "manifest.json")
    print(f"Copied model.onnx to {dest / 'model.onnx'}")
    if src_man.is_file():
        print(f"Copied manifest.json to {dest / 'manifest.json'}")
    else:
        print("No manifest.json in export_dir (server will use server_config.json / defaults).")


if __name__ == "__main__":
    main()
