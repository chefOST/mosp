"""Run metadata and checkpoint I/O shared by the small command-line tools."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .model import MOSPConfig, MOSPSystem


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    try:
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_commit": revision, "git_dirty": dirty, "torch": str(torch.__version__),
            "numpy": str(np.__version__)}


def save_checkpoint(path, model, optimizer, *, epoch, metadata, manifest, correlations):
    checkpoint = {"format_version": 1, "model_config": asdict(model.config),
                  "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch, "metadata": metadata,
                  "manifest_sha256": file_sha256(manifest),
                  "correlations": {"before": correlations[0], "after": correlations[1]},
                  "environment": environment()}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint; expected ChefOST MOSP format_version=1")
    config = MOSPConfig(**checkpoint["model_config"])
    correlations = checkpoint["correlations"]
    model = MOSPSystem(config, correlations["before"], correlations["after"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # PyTorch inference mode disables dropout; it does not score predictions.
    model.to(device).eval()
    return model, checkpoint
