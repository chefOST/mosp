"""Separate held-out scoring of a finished checkpoint on explicit frame grids.

The model receives features only. This scorer alone reads ground truth, after
checking checkpoint training membership, source identity and annotation pins.
The reported metric is explicitly frame-micro F1 on the model's 10 Hz anchors;
it is not ChefOST's frame-averaged metric or a claim of exact paper reproduction.
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .cloud_data import (
    _annotation_paths, _artifact, _document, _json, _pin_annotations,
    _records, _source, _timeline_length, _video_metadata,
)
from .data import _vocabulary
from .runtime import file_sha256, load_checkpoint, write_json


THRESHOLDS = tuple(index / 10 for index in range(1, 10))


def _metrics(counts: list[dict[str, int | float]]) -> dict[str, Any]:
    rows = []
    for count in counts:
        tp, fp, fn = int(count["tp"]), int(count["fp"]), int(count["fn"])
        rows.append({**count, "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    defined = [row["f1"] for row in rows if row["f1"] is not None]
    maximum = max(defined) if defined else None
    return {"thresholds": rows, "f1_max": maximum,
            "thresholds_attaining_f1_max": [row["threshold"] for row in rows
                                            if maximum is not None and row["f1"] == maximum],
            "fixed_threshold_0_5": next(row for row in rows if row["threshold"] == .5)}


def frame_micro_scores(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Pool binary decisions over time and object/state pairs at fixed thresholds.

    Predictions use >= threshold. Undefined precision/recall/F1 is JSON null;
    the F1-max sweep is a reporting metric, never checkpoint/model selection.
    """
    probabilities, targets = np.asarray(probabilities), np.asarray(targets)
    if probabilities.ndim != 2 or targets.shape != probabilities.shape or targets.dtype != np.bool_:
        raise ValueError("Probabilities and boolean targets must share [time, object*state] shape")
    counts = [{"threshold": threshold, "tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLDS]
    for start in range(0, len(probabilities), 512):
        values, truth = probabilities[start:start + 512], targets[start:start + 512]
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("State probabilities must be finite values in [0,1]")
        positives = int(np.count_nonzero(truth))
        for row in counts:
            predicted = values >= row["threshold"]
            true_positives = int(np.count_nonzero(predicted & truth))
            row["tp"] += true_positives
            row["fp"] += int(np.count_nonzero(predicted)) - true_positives
            row["fn"] += positives - true_positives
    return _metrics(counts)


def align_raw_frames(timestamps: np.ndarray, video: Mapping[str, Any], ground_truth_length: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Map zero-origin 10 Hz anchors to nearest verified CFR source frames.

    Ties round toward the later frame. No index clipping, rescaling, guessed
    offset, dropped frame or stretching is permitted.
    """
    times = np.asarray(timestamps)
    if (times.ndim != 1 or len(times) == 0 or times.dtype.kind != "f" or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(len(times), dtype=np.float64) / 10, rtol=0, atol=1e-7)):
        raise ValueError("Evaluation timestamps must be exact zero-origin 10 Hz feature anchors")
    try:
        fps = Fraction(video["r_frame_rate"])
        average = Fraction(video["avg_frame_rate"])
        frames = int(video["nb_frames"])
        duration = float(video["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Verified source FPS, average FPS, frame count and duration are required") from exc
    if fps <= 0 or average != fps or frames < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Scoring requires a finite, positive constant-frame-rate source timeline")
    if frames != ground_truth_length:
        raise ValueError("Ground-truth array length does not equal verified source raw-frame count")
    if abs(duration - frames / float(fps)) > 1 / float(fps) + 1e-6:
        raise ValueError("Source duration disagrees with raw-frame count/FPS")
    if abs(len(times) - duration * 10) > 2:
        raise ValueError("Feature timestamps do not cover the source duration at 10 Hz")
    if times[-1] >= duration:
        raise ValueError("Feature timestamp reaches beyond the source duration")
    indices = np.floor(times * float(fps) + .5).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= frames) or np.any(np.diff(indices) <= 0):
        raise ValueError("Mapped ground-truth indices are out of bounds or not strictly increasing")
    error = np.abs(indices / float(fps) - times)
    if np.any(error > .5 / float(fps) + 1e-7):
        raise ValueError("Timestamp mapping exceeds nearest-frame quantization error")
    return indices, {"source_fps": str(fps), "source_raw_frames": frames,
        "source_duration_seconds": duration, "ground_truth_samples": ground_truth_length,
        "feature_samples": len(times), "reporting_fps": 10,
        "first_raw_frame_index": int(indices[0]), "last_raw_frame_index": int(indices[-1]),
        "max_nearest_frame_error_seconds": float(error.max()),
        "rule": "floor(feature_anchor_seconds * source_fps + 0.5); anchors relative to first source frame",
        "temporal_offset_assumption": "zero offset on the matching raw-frame-count grid; no visual offset calibration",
        "interpolation_or_rescaling": False}


def _targets(record: dict[str, Any], indices: np.ndarray, objects, states, *, indexed: bool) -> np.ndarray:
    target = np.zeros((len(indices), len(objects.names) * len(states.names)), dtype=np.bool_)
    for object_name, timeline in record["state"].items():
        try:
            obj = objects.source_id_to_index[int(object_name)] if indexed else objects.name_to_index[object_name]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Ground-truth object is outside the checkpoint vocabulary: {object_name}") from exc
        for row, index in enumerate(indices):
            labels = timeline[int(index)]
            if not isinstance(labels, list):
                raise ValueError("Ground-truth state cells must be lists")
            for state in labels:
                try:
                    column = states.source_id_to_index[int(state)] if indexed else states.name_to_index[state]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Ground-truth state is outside the checkpoint vocabulary: {state}") from exc
                target[row, obj * len(states.names) + column] = True
    return target


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: str | Path,
    annotation_root: str | Path,
    dataset: str,
    source_inventory: Mapping[str, Any] | str | Path,
    feature_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score every reserved GT video after training and save auditable artifacts.

    Source/feature manifests use the same contract as cloud_data. Annotation
    revision and file hashes must be pinned in checkpoint split_provenance.
    The caller owns W&B/artifact publication; no external writes occur here.
    """
    model, saved = load_checkpoint(checkpoint, device)
    metadata = saved["metadata"]
    if metadata.get("dataset") != dataset or dataset not in ("CMU", "EGTEA"):
        raise ValueError("Checkpoint and requested dataset differ")
    split = metadata.get("split_provenance", {})
    revision = split.get("annotation_revision", "")
    root = Path(annotation_root).resolve()
    if (root / "MOSCATO").is_dir():
        root = root / "MOSCATO"
    paths = _annotation_paths(root, dataset)
    _pin_annotations(root, paths, revision, split.get("annotation_sha256"))
    ground_truth, indexed_gt = _json(paths["ground_truth"]), _json(paths["indexed_ground_truth"])
    if not ground_truth or set(ground_truth) != set(indexed_gt):
        raise ValueError("Named/indexed held-out ground-truth membership differs or is empty")
    train_ids = metadata.get("training_video_ids")
    if (not isinstance(train_ids, list) or not train_ids or any(not isinstance(v, str) for v in train_ids)
            or len(train_ids) != len(set(train_ids))):
        raise ValueError("Checkpoint must record explicit, unique training video IDs")
    if set(train_ids) & set(ground_truth):
        raise ValueError("Checkpoint training IDs overlap manually annotated ground truth")
    test_ids = metadata.get("reserved_splits", {}).get("test")
    if not isinstance(test_ids, list) or len(test_ids) != len(set(test_ids)) or set(test_ids) != set(ground_truth):
        raise ValueError("Checkpoint reserved test IDs must equal the current pinned ground-truth IDs")
    vocabularies = {name: _vocabulary(paths[name], prefix)
                    for name, prefix in (("objects", "o"), ("states", "s"), ("actions", "a"))}
    for name, vocabulary in vocabularies.items():
        if list(vocabulary.names) != metadata.get(name) or vocabulary.snapshot != metadata.get("vocabulary_snapshot", {}).get(name):
            raise ValueError(f"Checkpoint {name} vocabulary differs from pinned annotations")
    objects, states = vocabularies["objects"], vocabularies["states"]
    classes = len(objects.names) * len(states.names)
    if classes != model.config.num_state_classes:
        raise ValueError("Checkpoint state class count disagrees with object/state vocabulary")
    source_doc, source_base = _document(source_inventory)
    feature_doc, feature_base = _document(feature_manifest)
    sources = _records(source_doc, source_base, "source inventory")
    features = _records(feature_doc, feature_base, "feature manifest")
    missing = [v for v in test_ids if (dataset, v) not in sources or (dataset, v) not in features]
    if missing:
        raise ValueError(f"Missing held-out source/features: {missing}")
    output = Path(output_dir).resolve()
    if (output / "metrics.json").exists():
        raise FileExistsError("Refusing to overwrite completed evaluation metrics")
    output.mkdir(parents=True, exist_ok=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    totals = [{"threshold": value, "tp": 0, "fp": 0, "fn": 0} for value in THRESHOLDS]
    per_video, alignments = {}, {}
    checkpoint_hash = file_sha256(checkpoint)
    for video_id in sorted(test_ids):
        source_row = sources[(dataset, video_id)][0]
        source = _source(source_row, video_id)
        record, base = features[(dataset, video_id)]
        if _source(record, video_id) != source:
            raise ValueError(f"Held-out feature source differs from verified R2 inventory: {video_id}")
        if record.get("annotation_revision") != revision:
            raise ValueError(f"Held-out feature annotation revision differs: {video_id}")
        if record.get("feature_provenance") != metadata.get("feature_provenance"):
            raise ValueError(f"Held-out extractor provenance differs from checkpoint: {video_id}")
        if record["feature_provenance"].get("layout") != "TD":
            raise ValueError("Cloud scorer requires explicitly declared TD feature layout")
        feature_path, time_path = _artifact(record, base, "features"), _artifact(record, base, "timestamps")
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        times = np.load(time_path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != model.config.feature_dim or array.dtype.kind != "f":
            raise ValueError(f"Held-out features have invalid dimension/type: {video_id}")
        if times.shape != (array.shape[0],) or any(not np.isfinite(array[i:i + 1024]).all() for i in range(0, len(array), 1024)):
            raise ValueError(f"Held-out features/timestamps are invalid: {video_id}")
        count = _timeline_length(ground_truth[video_id], video_id)
        if _timeline_length(indexed_gt[video_id], video_id) != count:
            raise ValueError(f"Named/indexed GT raw-frame lengths differ: {video_id}")
        indices, alignment = align_raw_frames(times, _video_metadata(source_row), count)
        target = _targets(ground_truth[video_id], indices, objects, states, indexed=False)
        indexed_target = _targets(indexed_gt[video_id], indices, objects, states, indexed=True)
        if not np.array_equal(target, indexed_target):
            raise ValueError(f"Named/indexed GT targets disagree on the evaluated frame grid: {video_id}")
        del indexed_target
        final = predictions_dir / f"{video_id}.npz"
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite prediction artifact: {final}")
        # There is intentionally no path from target/indices into model.forward.
        tensor = torch.as_tensor(np.array(array.T, dtype=np.float32, copy=True), device=device).unsqueeze(0)
        result = model(tensor)
        probabilities = result["probabilities"][0].detach().cpu().numpy().T.astype(np.float32, copy=False)
        if probabilities.shape != target.shape:
            raise ValueError("Prediction dimensions do not match the held-out frame/class grid")
        scores = frame_micro_scores(probabilities, target)
        temporary = final.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, state_probabilities=probabilities,
                                timestamps_seconds=times, ground_truth_raw_frame_indices=indices)
        temporary.replace(final)
        for total, row in zip(totals, scores["thresholds"]):
            for name in ("tp", "fp", "fn"):
                total[name] += row[name]
        per_video[video_id] = {"frame_micro": scores, "feature_samples": len(times),
                              "prediction_path": str(final), "prediction_sha256": file_sha256(final)}
        alignments[video_id] = {**alignment, "source": source,
            "feature_sha256": file_sha256(feature_path), "timestamps_sha256": file_sha256(time_path),
            "named_indexed_target_agreement": True}
        del tensor, result, probabilities, target
    metrics = {"schema_version": 1, "status": "complete", "dataset": dataset,
        "scope": "held-out current curated GT; reimplementation; scores sampled on model 10 Hz reporting grid",
        "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": saved["epoch"],
        "annotation_revision": revision, "training_video_ids": sorted(train_ids), "test_video_ids": sorted(test_ids),
        "training_ground_truth_overlap": [], "frame_micro": _metrics(totals), "per_video": per_video,
        "definition": {"pooling": "sum TP/FP/FN across sampled timestamps, all vocabulary object-state pairs and all test videos",
            "threshold_comparison": "probability >= threshold", "threshold_grid": list(THRESHOLDS),
            "f1_max_use": "reporting-only sweep; no checkpoint, model or deployment threshold selected",
            "empty_gt_state_lists": "all-zero target for this frame-micro metric", "undefined_denominators": "null (N/A)",
            "label_normalization": "exact pinned vocabulary aliases/source IDs; object-major then state-minor classes"},
        "omitted_metrics": {"mAP_and_segment_F1": "exact released aggregation/segment definitions not established here",
            "ChefOST": "frame-micro scores are not ChefOST frame-averaged scores; sampling/empty-GT/object-video aggregation choices require a shared protocol",
            "cosine_similarity": "no agreed embedding model or comparison unit"},
        "alignment_audit_path": str(output / "alignment_audit.json")}
    write_json(output / "alignment_audit.json", {"annotation_revision": revision, "videos": alignments})
    write_json(output / "metrics.json", metrics)
    return metrics
