"""Joint MOSP/TAS/SAI training on an explicit, disjoint prepared split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from .data import PreparedDataset, compute_state_action_correlations
from .losses import joint_loss
from .model import MOSPConfig, MOSPSystem
from .runtime import (capture_rng_state, environment, file_sha256, read_checkpoint,
                      restore_rng_state, save_checkpoint, write_json)


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
    p.add_argument("--resume", action="store_true", help="Resume out/last.pt with the same training contract")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Retain an epoch checkpoint every N epochs; last.pt is always saved (0 disables snapshots)")
    p.add_argument("--wandb-project", help="Enable W&B in this exact project name")
    p.add_argument("--wandb-entity", default="chefOST")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-job-type", default="train")
    p.add_argument("--wandb-name")
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    return p


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_fingerprint(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def _data_fingerprint(dataset):
    """Fingerprint actual training inputs, never load reserved split samples."""
    samples = []
    for index in range(len(dataset)):
        sample = dataset[index]
        samples.append({"video_id": sample.video_id,
                        "features": _array_fingerprint(sample.features),
                        "states": _array_fingerprint(sample.states),
                        "actions": _array_fingerprint(sample.actions),
                        "action_segments": getattr(sample, "action_segments", ()),
                        "correlation_segments": getattr(sample, "correlation_segments", ())})
    sources = {name: file_sha256(path) for name, path in getattr(dataset, "sources", {}).items()
               if name != "ground_truth"}
    # Ground-truth membership is an exclusion rule, not training supervision.
    excluded_ids = sorted(getattr(dataset, "ground_truth_ids", ()))
    result = {"samples": samples, "annotation_sources": sources,
              "ground_truth_exclusion_ids_sha256": _json_hash(excluded_ids)}
    return {**result, "sha256": _json_hash(result)}


def _implementation_fingerprint():
    root = Path(__file__).parent
    names = ("train.py", "runtime.py", "model.py", "losses.py", "data.py")
    return {name: file_sha256(root / name) for name in names}


def _wandb_configuration(args):
    project = getattr(args, "wandb_project", None)
    if project is None:
        return None
    if not isinstance(project, str) or not project.strip():
        raise ValueError("wandb_project must be an explicit nonempty project name")
    return {"project": project, "entity": getattr(args, "wandb_entity", "chefOST"),
            "group": getattr(args, "wandb_group", None),
            "job_type": getattr(args, "wandb_job_type", "train"),
            "name": getattr(args, "wandb_name", None),
            "mode": getattr(args, "wandb_mode", "online")}


def _start_wandb(run, out):
    identity = run.get("wandb")
    if identity is None:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested; install the optional wandb dependency") from exc
    logged = wandb.init(**identity["configuration"], id=identity["id"], resume="allow",
                        dir=str(out), config={"model": run["model_config"],
                                              "training": run["training_config"],
                                              "metadata": run["metadata"],
                                              "manifest_sha256": run["manifest_sha256"],
                                              "training_data_sha256": run["training_data_sha256"],
                                              "checkpoint_policy": run["checkpoint_policy"]},
                        settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True,
                                                x_disable_machine_info=True))
    logged.define_metric("epoch")
    logged.define_metric("train/*", step_metric="epoch")
    identity["url"] = getattr(logged, "url", None)
    return wandb, logged


def _log_checkpoint_artifact(wandb, logged, path, run, *, final=False):
    epoch = run["last_completed_epoch"]
    artifact = wandb.Artifact(f"mosp-{run['wandb']['id']}", type="model",
                             metadata={"epoch": epoch, "dataset": run["metadata"].get("dataset"),
                                       "task": run["metadata"].get("task"),
                                       "manifest_sha256": run["manifest_sha256"],
                                       "training_data_sha256": run["training_data_sha256"],
                                       "checkpoint_policy": run["checkpoint_policy"]})
    artifact.add_file(str(path), name="last.pt")
    artifact.add_file(str(path.parent / "run.json"), name="run.json")
    artifact.add_file(str(path.parent / "manifest.snapshot.json"), name="manifest.snapshot.json")
    aliases = ["latest", f"epoch-{epoch:06d}"] + (["final"] if final else [])
    saved = logged.log_artifact(artifact, aliases=aliases)
    if run["wandb"]["configuration"]["mode"] == "online":
        saved.wait()


def train(args, *, checkpoint_callback=None):
    """Train or resume; callback(last.pt) runs after each durable local checkpoint.

    The caller supplies a persistent --out directory. Cloud callers can commit
    their mounted volume in checkpoint_callback. No validation split is scored,
    no best checkpoint is selected, and W&B system/GPU telemetry is disabled.
    """
    if (args.epochs < 1 or args.threads < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0):
        raise ValueError("Epochs, threads and learning rate must be positive")
    if (not math.isfinite(args.positive_weight) or not math.isfinite(args.smoothing_weight)
            or args.positive_weight <= 0 or args.smoothing_weight < 0):
        raise ValueError("Loss weights are invalid")
    args.out, args.manifest = Path(args.out), Path(args.manifest)
    resume = getattr(args, "resume", False)
    checkpoint_every = getattr(args, "checkpoint_every", 10)
    if type(checkpoint_every) is not int or checkpoint_every < 0:
        raise ValueError("checkpoint_every must be a nonnegative integer")
    if args.out.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite run directory: {args.out}")
    if resume and not (args.out / "last.pt").is_file():
        raise FileNotFoundError(f"Resume requires an existing checkpoint: {args.out / 'last.pt'}")
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
    training_config = {"epochs": args.epochs, "optimizer": "Adam", "seed": args.seed,
                       "learning_rate": args.learning_rate, "positive_weight": args.positive_weight,
                       "smoothing_weight": args.smoothing_weight}
    fingerprint = _data_fingerprint(train_data)
    contract = {"version": 1, "model_config": asdict(config), "training_config": training_config,
                "metadata": metadata, "manifest_sha256": file_sha256(args.manifest),
                "training_data": fingerprint, "implementation": _implementation_fingerprint(),
                "libraries": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
                "wandb_configuration": _wandb_configuration(args)}
    if not resume:
        args.out.mkdir(parents=True)
    with (args.out / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another trainer is already using {args.out}") from exc
        return _train_locked(args, train_data, metadata, config, contract, checkpoint_every,
                             checkpoint_callback, resume)


def _train_locked(args, train_data, metadata, config, contract, checkpoint_every,
                  checkpoint_callback, resume):
    checkpoint = read_checkpoint(args.out / "last.pt") if resume else None
    if checkpoint is not None:
        state = checkpoint.get("training_state")
        if not isinstance(state, dict) or "rng_state" not in checkpoint:
            raise ValueError("Checkpoint lacks resumable training state; inference remains supported")
        saved_contract = state.get("contract", {})
        if saved_contract != contract:
            changed = [key for key in sorted(set(contract) | set(saved_contract))
                       if saved_contract.get(key) != contract.get(key)]
            raise ValueError(f"Resume training contract changed: {', '.join(changed)}")
        if checkpoint["model_config"] != asdict(config) or checkpoint["metadata"] != metadata:
            raise ValueError("Checkpoint model/metadata differs from its training contract")
        start_epoch = checkpoint["epoch"]
        history = state.get("history", [])
        if (type(start_epoch) is not int or not 0 <= start_epoch <= args.epochs
                or [row.get("epoch") for row in history] != list(range(1, start_epoch + 1))):
            raise ValueError("Checkpoint epoch/history is inconsistent")
        before, after = (checkpoint["correlations"][key] for key in ("before", "after"))
        wandb_identity = state.get("wandb")
        if (wandb_identity is None) != (contract["wandb_configuration"] is None):
            raise ValueError("Checkpoint W&B identity differs from its training contract")
        if wandb_identity is not None and (
                not wandb_identity.get("id")
                or wandb_identity.get("configuration") != contract["wandb_configuration"]):
            raise ValueError("Checkpoint W&B identity differs from its training contract")
    else:
        state, start_epoch, history = {}, 0, []
        before, after = compute_state_action_correlations(train_data)
        before, after = torch.as_tensor(before, dtype=torch.float32), torch.as_tensor(after, dtype=torch.float32)
        if config.use_sai and not bool((before.sum() + after.sum()) > 0):
            raise ValueError("SAI has no observed training action-state correlations")
        wandb_identity = ({"id": uuid.uuid4().hex[:8], "configuration": contract["wandb_configuration"]}
                          if contract["wandb_configuration"] else None)
    model = MOSPSystem(config, before, after).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if checkpoint["optimizer"].get("param_groups") != optimizer.state_dict()["param_groups"]:
            raise ValueError("Checkpoint optimizer settings differ from the training contract")
        optimizer.load_state_dict(checkpoint["optimizer"])
    run = {"status": "running", "model_config": asdict(config), "training_config": contract["training_config"],
           "metadata": metadata, "source_manifest": str(args.manifest.resolve()),
           "manifest_sha256": contract["manifest_sha256"], "training_data_sha256": contract["training_data"]["sha256"],
           "seed": args.seed, "epochs_requested": args.epochs, "learning_rate": args.learning_rate,
           "positive_weight": args.positive_weight, "smoothing_weight": args.smoothing_weight,
           "checkpoint_policy": "latest completed training epoch; no score-based selection", "device": args.device,
           "environment": environment(), "history": list(history), "last_completed_epoch": start_epoch,
           "resume_count": state.get("resume_count", 0) + int(resume), "wandb": wandb_identity,
           "scope": "implementation run; not evidence of published-result reproduction"}
    write_json(args.out / "manifest.snapshot.json", json.loads(args.manifest.read_text()))
    write_json(args.out / "run.json", run)

    def persist(epoch):
        training_state = {"contract": contract, "history": run["history"],
                          "wandb": run["wandb"], "resume_count": run["resume_count"]}
        rng = capture_rng_state()
        paths = [args.out / "last.pt"]
        if epoch and checkpoint_every and epoch % checkpoint_every == 0:
            paths.append(args.out / f"epoch_{epoch:06d}.pt")
        for path in paths:
            save_checkpoint(path, model, optimizer, epoch=epoch, metadata=metadata,
                            manifest=args.manifest, correlations=(before, after),
                            training_state=training_state, rng_state=rng)
        write_json(args.out / "run.json", run)
        if checkpoint_callback is not None:
            checkpoint_callback(args.out / "last.pt")
        return rng

    wandb = logged = None
    try:
        # Epoch zero makes a first-epoch interruption resumable too. W&B identity
        # is persisted before the SDK starts, so retries reuse the same run.
        rng = checkpoint["rng_state"] if checkpoint is not None else persist(0)
        wandb, logged = _start_wandb(run, args.out)
        restore_rng_state(rng)
        for epoch in range(start_epoch + 1, args.epochs + 1):
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
            run["last_completed_epoch"] = epoch
            if logged is not None:
                logged.log({"epoch": epoch, **{f"train/{key}": value for key, value in record.items()
                                               if key != "epoch"}}, step=epoch)
            persist(epoch)
            if logged is not None and checkpoint_every and epoch % checkpoint_every == 0 and epoch < args.epochs:
                _log_checkpoint_artifact(wandb, logged, args.out / "last.pt", run)
            print(json.dumps(record), flush=True)
        run["status"] = "completed"
        write_json(args.out / "run.json", run)
        if checkpoint_callback is not None:
            checkpoint_callback(args.out / "last.pt")
        if logged is not None:
            logged.summary.update({"status": "completed", "completed_epochs": run["last_completed_epoch"],
                                   "checkpoint_policy": run["checkpoint_policy"]})
            _log_checkpoint_artifact(wandb, logged, args.out / "last.pt", run, final=True)
    except BaseException as error:
        run["status"] = "failed"
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "run.json", run)
        if logged is not None:
            logged.summary.update({"status": run["status"], "completed_epochs": run["last_completed_epoch"]})
            if "error" in run:
                logged.summary["error"] = run["error"]
            logged.finish(exit_code=0 if run["status"] == "completed" else 1)
    return run


if __name__ == "__main__":
    train(parser().parse_args())
