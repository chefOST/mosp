"""Evaluate saved MOSP predictions with explicit metric decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import PreparedDataset
from .metrics import state_metrics
from .runtime import file_sha256, load_checkpoint, validate_checkpoint_data, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--evaluation-timeline", choices=["annotation", "feature"], required=True,
                   help="Full annotation scoring requires explicit annotation-to-feature indices")
    p.add_argument("--segment-threshold", type=float, required=True)
    p.add_argument("--gap-frames", type=int, required=True,
                   help="Merge prediction gaps shorter than this many evaluated samples; paper uses 10 frames")
    p.add_argument("--map-empty-classes", choices=["exclude", "zero"], required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    return p


@torch.no_grad()
def evaluate(args):
    if args.out and args.out.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {args.out}")
    torch.set_num_threads(args.threads)
    dataset = PreparedDataset(args.manifest, args.split)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    validate_checkpoint_data(checkpoint, dataset.metadata)
    forbidden = set(checkpoint["metadata"]["training_video_ids"])
    if args.split == "test":
        forbidden.update(checkpoint["metadata"]["validation_video_ids"])
    overlap = forbidden.intersection(dataset.video_ids)
    if overlap:
        raise ValueError(f"Evaluation IDs overlap checkpoint training/selection data: {sorted(overlap)}")
    scored = []
    for sample in dataset:
        features = torch.as_tensor(sample.features, dtype=torch.float32, device=args.device).unsqueeze(0)
        # State/action annotations never enter model.forward: test supervision is
        # used only below, to score the already-produced probabilities.
        probabilities = model(features)["probabilities"][0].cpu().numpy().T
        if args.evaluation_timeline == "annotation":
            if sample.evaluation_indices is None or sample.evaluation_states is None:
                raise ValueError(f"Full annotation scoring requires evaluation_indices: {sample.video_id}")
            scored.append((probabilities[sample.evaluation_indices], sample.evaluation_states.T))
        else:
            scored.append((probabilities, sample.states.T))
    report = state_metrics(scored, segment_threshold=args.segment_threshold,
                           gap_frames=args.gap_frames, empty_class_policy=args.map_empty_classes)
    report.update({"checkpoint_sha256": file_sha256(args.checkpoint),
                   "manifest_sha256": file_sha256(args.manifest), "split": args.split,
                   "video_ids": dataset.video_ids, "model_config": checkpoint["model_config"],
                   "checkpoint_selection": checkpoint["selection"],
                   "split_provenance": dataset.metadata["split_provenance"],
                   "evaluation_timeline": args.evaluation_timeline,
                   "limitation": ("Feature-timeline diagnostic; not full annotation-frame scoring"
                                  if args.evaluation_timeline == "feature" else
                                  "Full annotation timeline using explicit prediction index mapping")})
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    evaluate(parser().parse_args())
