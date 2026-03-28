#!/usr/bin/env python3
"""
Train a 4-class rock / paper / scissors / none classifier from a CSV + image folder,
then export ONNX + manifest.json aligned with the browser client (NCHW, ImageNet norm).

All paths and hyperparameters come from a JSON config file; pass its path as the only
CLI argument. See train_config.example.json. Relative paths are resolved from the config file's directory.

Device: optional JSON key `device`: `auto` (default), `cpu`, `cuda`, or `mps`. With `auto`, pick CUDA if
available, else MPS if available, else CPU. ONNX export runs on CPU when training used MPS.

CSV: default columns `filename` (relative to data_dir or basename) and `label`.
Labels (case-insensitive): rock, paper, scissors, none
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from json_config import config_path, load_config_dict

LABELS = ["rock", "paper", "scissors", "none"]
LABEL_TO_IDX = {k: i for i, k in enumerate(LABELS)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _norm_label(raw: str) -> Optional[int]:
    s = (raw or "").strip().lower()
    return LABEL_TO_IDX.get(s)


class RpsCsvDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[Path, int]],
        image_size: int,
        train: bool,
    ) -> None:
        self.rows = rows
        aug = [transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.15, 0.15, 0.1, 0.05)]
        t_list = [
            transforms.Resize((image_size, image_size)),
            *(aug if train else []),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.tf = transforms.Compose(t_list)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, y = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y


def load_rows(data_dir: Path, csv_path: Path, fname_col: str, label_col: str) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fname_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise SystemExit(f"CSV must have columns {fname_col!r} and {label_col!r}; got {reader.fieldnames}")
        for r in reader:
            fn = (r.get(fname_col) or "").strip()
            lab = _norm_label(r.get(label_col) or "")
            if not fn or lab is None:
                continue
            p = Path(fn)
            if not p.is_file():
                p = data_dir / fn
            if not p.is_file():
                continue
            rows.append((p.resolve(), lab))
    if len(rows) < 4:
        raise SystemExit(f"Need at least a few valid rows; found {len(rows)}")
    return rows


def stratified_split(
    rows: list[tuple[Path, int]], val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_c: dict[int, list[int]] = {i: [] for i in range(len(LABELS))}
    for i, (_, y) in enumerate(rows):
        by_c[y].append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for indices in by_c.values():
        rng.shuffle(indices)
        n_val = int(len(indices) * val_frac)
        if len(indices) >= 2 and n_val == 0:
            n_val = 1
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    return train_idx, val_idx


def build_model(num_classes: int) -> nn.Module:
    w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = mobilenet_v3_small(weights=w)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, num_classes)
    return m


def write_manifest(out_dir: Path, image_size: int, version: str) -> None:
    manifest = {
        "version": version,
        "labels": LABELS,
        "input": {
            "name": "input",
            "width": image_size,
            "height": image_size,
            "layout": "NCHW",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "output": {"name": "logits"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def pick_training_device(cfg: dict[str, Any]) -> torch.device:
    raw = _cfg_str(cfg, "device", "auto").lower()
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("device=cuda but CUDA is not available")
        return torch.device("cuda")
    if raw == "mps":
        if not _mps_available():
            raise SystemExit("device=mps but MPS is not available (needs Apple Silicon + supported PyTorch)")
        return torch.device("mps")
    if raw == "cpu":
        return torch.device("cpu")
    if raw != "auto":
        raise SystemExit(f"unknown device: {raw!r}; use auto, cpu, cuda, or mps")
    # auto: cuda → mps → cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_onnx(model: nn.Module, path: Path, image_size: int, train_device: torch.device) -> None:
    model.eval()
    # ONNX export on MPS is unreliable; CPU is fast for a single forward.
    if train_device.type == "mps":
        model = model.cpu()
        export_dev = torch.device("cpu")
    else:
        export_dev = train_device
    dummy = torch.randn(1, 3, image_size, image_size, device=export_dev)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


def _cfg_str(cfg: dict[str, Any], key: str, default: str) -> str:
    v = cfg.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    if key not in cfg or cfg[key] is None:
        return default
    return int(cfg[key])


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    if key not in cfg or cfg[key] is None:
        return default
    return float(cfg[key])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config",
        type=Path,
        help="JSON config (see train_config.example.json); relative paths are from its directory",
    )
    cli = ap.parse_args()
    if not cli.config.is_file():
        raise SystemExit(f"Config not found: {cli.config}")

    cfg, base = load_config_dict(cli.config)
    data_dir = config_path(cfg.get("data_dir"), base, key="data_dir")
    csv_path = config_path(cfg.get("csv"), base, key="csv")
    filename_column = _cfg_str(cfg, "filename_column", "filename")
    label_column = _cfg_str(cfg, "label_column", "label")
    image_size = _cfg_int(cfg, "image_size", 224)
    epochs = _cfg_int(cfg, "epochs", 20)
    batch_size = _cfg_int(cfg, "batch_size", 16)
    lr = _cfg_float(cfg, "lr", 3e-4)
    val_frac = _cfg_float(cfg, "val_frac", 0.15)
    seed = _cfg_int(cfg, "seed", 42)
    out_dir_s = cfg.get("out_dir", "export_rps")
    if out_dir_s is None or (isinstance(out_dir_s, str) and not str(out_dir_s).strip()):
        raise SystemExit("Config missing or empty: out_dir")
    out_dir = Path(str(out_dir_s).strip()).expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (base / out_dir).resolve()
    manifest_version = _cfg_str(cfg, "manifest_version", "1.0.0")

    if image_size < 96 or image_size > 640:
        raise SystemExit("image_size should be between 96 and 640")

    torch.manual_seed(seed)
    random.seed(seed)

    device = pick_training_device(cfg)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    rows = load_rows(data_dir, csv_path, filename_column, label_column)
    train_i, val_i = stratified_split(rows, val_frac, seed)
    full = RpsCsvDataset(rows, image_size, train=True)
    full_eval = RpsCsvDataset(rows, image_size, train=False)
    ds_tr = Subset(full, train_i)
    ds_va = Subset(full_eval, val_i)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
            n_tr += x.size(0)
        loss_tr /= max(n_tr, 1)

        model.eval()
        correct = 0
        n_va = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                n_va += x.size(0)
        acc = (correct / n_va) if n_va else 0.0
        print(f"epoch {epoch + 1}/{epochs}  train_loss={loss_tr:.4f}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, onnx_path, image_size, train_device=device)
    write_manifest(out_dir, image_size, manifest_version)
    meta = {
        "image_size": image_size,
        "val_acc": best_acc,
        "n_train": len(train_i),
        "n_val": len(val_i),
        "labels": LABELS,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {onnx_path}, manifest.json, training_meta.json under {out_dir}")
    print(f"Deploy: create a deploy JSON with \"export_dir\": \"{out_dir}\" then: python deploy_model.py that.json")


if __name__ == "__main__":
    main()
