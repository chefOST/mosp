"""Exercise the entire pipeline on tiny, explicitly synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import evaluate, predict, train
from .runtime import write_json


def make_fixture(destination):
    """Create synthetic data with disjoint IDs and known time alignment."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "objects.json", {"o2i": {"synthetic_object": 0}})
    write_json(destination / "states.json", {"s2i": {"before": 0, "after": 1}})
    write_json(destination / "actions.json", {"a2i": {"prepare": 0, "transform": 1}})
    rng = np.random.default_rng(7)
    pseudo, ground_truth, records = {}, {}, {}
    split_ids = {"train": ["train_a", "train_b"], "validation": ["validation"], "test": ["test"]}
    for split, ids in split_ids.items():
        for video_id in ids:
            length, midpoint = 16, 8
            states = [["before"] if t < midpoint else ["after"] for t in range(length)]
            features = rng.normal(scale=0.05, size=(4, length)).astype(np.float32)
            features[0, :midpoint] += 1
            features[1, midpoint:] += 1
            features[2] = np.linspace(0, 1, length)
            features[3] += 0.5
            np.save(destination / f"{video_id}.npy", features)
            annotation = {"state": {"synthetic_object": states},
                          "action": [[0, midpoint, "prepare", 0], [midpoint, length, "transform", 1]]}
            # Mimic the release: pseudo files can contain held-out IDs. Explicit
            # split membership, not source-file membership, decides training.
            pseudo[video_id] = annotation
            if split != "train":
                ground_truth[video_id] = {"state": annotation["state"]}
            record = {"feature_path": f"{video_id}.npy", "state_indices": list(range(length)),
                      "state_source": "pseudo_labels" if split == "train" else "ground_truth",
                      "action_source": "pseudo_labels" if split == "train" else "none"}
            if split == "train":
                record.update({"action_indices": list(range(length)), "action_segment_end": "exclusive"})
            else:
                record["evaluation_indices"] = list(range(length))
            records[video_id] = record
    write_json(destination / "pseudo.json", pseudo)
    write_json(destination / "ground_truth.json", ground_truth)
    manifest = {"schema_version": 1, "dataset": "synthetic",
                "split_provenance": {"description": "Disjoint generated fixture IDs; not MOSCATO videos",
                                     "paper_split_verified": False},
                "vocabularies": {"objects": "objects.json", "states": "states.json", "actions": "actions.json"},
                "annotations": {"pseudo_labels": "pseudo.json", "ground_truth": "ground_truth.json"},
                "feature_provenance": {"model": "generated numeric fixture; not I3D", "dimension": 4,
                                       "layout": "DT", "timeline": "one feature per synthetic annotation sample"},
                "splits": split_ids, "videos": records}
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "feature_provenance.json", manifest["feature_provenance"])
    return destination / "manifest.json"


def smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = make_fixture(output / "fixture")
    args = train.parser().parse_args([
        "--manifest", str(manifest), "--out", str(output / "training"), "--epochs", "2",
        "--beta", "0.3", "--layers", "2", "--refinements", "1", "--channels", "4",
        "--dropout", "0", "--learning-rate", "0.001", "--threads", "1"])
    run = train.train(args)
    checkpoint = output / "training" / "selected.pt"
    report = evaluate.evaluate(evaluate.parser().parse_args([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--segment-threshold", "0.5",
        "--evaluation-timeline", "annotation", "--gap-frames", "0", "--map-empty-classes", "exclude",
        "--out", str(output / "metrics.json")]))
    predict.predict(predict.parser().parse_args([
        "--checkpoint", str(checkpoint), "--features", str(output / "fixture" / "test.npy"),
        "--layout", "DT", "--feature-provenance", str(output / "fixture" / "feature_provenance.json"),
        "--out", str(output / "predictions.npz")]))
    summary = {"status": "passed", "synthetic_only": True, "epochs": len(run["history"]),
               "finite_training_losses": all(np.isfinite(h["total"]) for h in run["history"]),
               "test_videos": report["protocol"]["num_videos"],
               "statement": "Tests train/checkpoint/evaluate/feature-only inference; not model quality or paper reproduction"}
    write_json(output / "smoke.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    smoke(parser.parse_args().out)
