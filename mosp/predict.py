"""Feature-only inference: no state or action annotation files are accepted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import file_sha256, load_checkpoint, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--layout", choices=["DT", "TD"], required=True)
    p.add_argument("--feature-provenance", type=Path, required=True,
                   help="JSON describing actual input extractor/timeline; must match checkpoint feature_provenance")
    p.add_argument("--out", type=Path, required=True, help="New .npz output path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    return p


@torch.no_grad()
def predict(args):
    if args.out.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz")
    metadata_path = args.out.with_suffix(".json")
    if args.out.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite prediction outputs")
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    provenance = json.loads(args.feature_provenance.read_text())
    if provenance != checkpoint["metadata"]["feature_provenance"]:
        raise ValueError("Input feature provenance differs from the checkpoint's expected extractor/timeline")
    if provenance.get("layout") != args.layout:
        raise ValueError("Input layout differs from the declared feature provenance")
    features = np.load(args.features, allow_pickle=False)
    if features.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix")
    if args.layout == "TD":
        features = features.T
    if features.shape[0] != model.config.feature_dim or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("Invalid feature dimensions/values for this checkpoint")
    output = model(torch.as_tensor(features, dtype=torch.float32, device=args.device).unsqueeze(0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        state_probabilities=output["probabilities"][0].cpu().numpy().T,
                        action_probabilities=output["action_logits"][-1, 0].softmax(0).cpu().numpy().T)
    write_json(metadata_path, {"checkpoint_sha256": file_sha256(args.checkpoint),
                              "feature_sha256": file_sha256(args.features), "input_layout": args.layout,
                              "output_layout": "time, class", "metadata": checkpoint["metadata"],
                              "model_config": checkpoint["model_config"],
                              "input_feature_provenance": provenance,
                              "state_class_order": "object-major, then state; C = O*S",
                              "scope": "offline feature-only prediction; no test annotations used"})
    return args.out


if __name__ == "__main__":
    print(predict(parser().parse_args()))
