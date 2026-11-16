"""Run metadata and checkpoint I/O shared by the small command-line tools."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
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


def capture_rng_state():
    """Keep RNG state in weights_only-compatible Python values and tensors."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"generator": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, model, optimizer, *, epoch, metadata, manifest, correlations,
                    training_state=None, rng_state=None):
    checkpoint = {"format_version": 1, "model_config": asdict(model.config),
                  "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch, "metadata": metadata,
                  "manifest_sha256": file_sha256(manifest),
                  "correlations": {"before": correlations[0], "after": correlations[1]},
                  "environment": environment()}
    if training_state is not None:
        checkpoint["training_state"] = training_state
        checkpoint["rng_state"] = capture_rng_state() if rng_state is None else rng_state
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint; expected ChefOST MOSP format_version=1")
    return checkpoint


def load_checkpoint(path, device="cpu"):
    checkpoint = read_checkpoint(path)
    config = MOSPConfig(**checkpoint["model_config"])
    correlations = checkpoint["correlations"]
    model = MOSPSystem(config, correlations["before"], correlations["after"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # PyTorch inference mode disables dropout; it does not score predictions.
    model.to(device).eval()
    return model, checkpoint
