"""Joint MOSP/TAS/SAI training on an explicit, disjoint prepared split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import PreparedDataset, compute_state_action_correlations
from .losses import joint_loss
from .model import MOSPConfig, MOSPSystem
from .runtime import environment, file_sha256, save_checkpoint, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="New run directory; existing directories are refused")
    p.add_argument("--epochs", type=int, required=True, help="Fixed training budget")
    p.add_argument("--beta", type=float, required=True, help="SAI probability fusion weight saved with the model")
    p.add_argument("--without-sai", action="store_true", help="MOSP-only output; requires beta=0")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--positive-weight", type=float, default=5.0)
    p.add_argument("--smoothing-weight", type=float, default=0.15)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--refinements", type=int, default=3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--sai-fusion", choices=["classwise", "dense"], default="classwise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    return p


def train(args):
    if args.epochs < 1 or args.threads < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, threads and learning rate must be positive")
    if args.positive_weight <= 0 or args.smoothing_weight < 0:
        raise ValueError("Loss weights are invalid")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {args.out}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_data = PreparedDataset(args.manifest, "train")
    if len(train_data) == 0:
        raise ValueError("Training split is empty")
    metadata = dict(train_data.metadata)
    metadata["training_video_ids"] = list(train_data.video_ids)
    metadata["reserved_splits"] = {name: list(train_data.manifest["splits"][name])
                                   for name in ("validation", "test")}
    config = MOSPConfig(num_state_classes=len(metadata["objects"]) * len(metadata["states"]),
                        num_actions=len(metadata["actions"]), beta=args.beta,
                        feature_dim=metadata["feature_dimension"], num_layers=args.layers,
                        num_refinements=args.refinements, num_f_maps=args.channels,
                        dropout=args.dropout, use_sai=not args.without_sai, sai_fusion=args.sai_fusion)
    before, after = compute_state_action_correlations(train_data)
    before = torch.as_tensor(before, dtype=torch.float32)
    after = torch.as_tensor(after, dtype=torch.float32)
    if config.use_sai and not bool((before.sum() + after.sum()) > 0):
        raise ValueError("SAI has no observed training action-state correlations")
    model = MOSPSystem(config, before, after).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.out.mkdir(parents=True)
    write_json(args.out / "manifest.snapshot.json", json.loads(args.manifest.read_text()))
    run = {"status": "running", "model_config": asdict(config), "metadata": metadata,
           "source_manifest": str(args.manifest.resolve()),
           "manifest_sha256": file_sha256(args.manifest), "seed": args.seed,
           "epochs_requested": args.epochs, "learning_rate": args.learning_rate,
           "positive_weight": args.positive_weight, "smoothing_weight": args.smoothing_weight,
           "checkpoint_policy": "latest completed training epoch; no score-based selection", "device": args.device,
           "environment": environment(), "history": [],
           "scope": "implementation run; not evidence of published-result reproduction"}
    write_json(args.out / "run.json", run)
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = {}
            for index in np.random.permutation(len(train_data)):
                sample = train_data[int(index)]
                features = torch.as_tensor(sample.features, dtype=torch.float32, device=args.device).unsqueeze(0)
                labels = torch.as_tensor(sample.states, dtype=torch.float32, device=args.device).unsqueeze(0)
                actions = torch.as_tensor(sample.actions, dtype=torch.long, device=args.device).unsqueeze(0)
                if not torch.any(actions != -100):
                    raise ValueError(f"Training video {sample.video_id} has no action supervision for TAS")
                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)
                losses = joint_loss(outputs, labels, actions, pos_weight=args.positive_weight,
                                    smoothing_weight=args.smoothing_weight)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(f"Non-finite training loss for {sample.video_id}")
                losses["total"].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"Non-finite gradients for {sample.video_id}")
                optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
            record = {"epoch": epoch, **{key: value / len(train_data) for key, value in sums.items()}}
            run["history"].append(record)
            save_checkpoint(args.out / "last.pt", model, optimizer, epoch=epoch, metadata=metadata,
                            manifest=args.manifest, correlations=(before, after))
            run["last_completed_epoch"] = epoch
            write_json(args.out / "run.json", run)
            print(json.dumps(record), flush=True)
        run["status"] = "completed"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "run.json", run)
    return run


if __name__ == "__main__":
    train(parser().parse_args())
