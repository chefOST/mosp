"""Resumable feature-only batch inference, without annotation or scoring inputs.

Each completed video is an atomic directory containing probabilities and its
receipt. The shared index can be reconstructed from those immutable receipts
after interruption. Existing outputs with different inputs are never replaced.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .cloud_data import _artifact, _document, _json, _records, _source
from .runtime import file_sha256, load_checkpoint, write_json


SCHEMA_VERSION = 1
PRODUCER = "mosp.batch_predict"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _ids(values: Any) -> list[str]:
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                   for value in values)):
        raise ValueError("An explicit nonempty list of safe video IDs is required")
    if len(values) != len(set(values)):
        raise ValueError("Requested video IDs must be unique")
    return sorted(values)


@contextmanager
def _single_writer(directory: Path):
    # Modal Volumes do not support distributed locks. The cloud runner must
    # serialize writers; this container-local lock also protects local callers.
    lock_root = Path(tempfile.gettempdir()) / "chefost-mosp-prediction-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(directory).encode()).hexdigest() + ".lock"
    with (lock_root / lock_name).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another prediction process is writing this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _vocabularies(metadata: dict[str, Any], model) -> dict[str, list[str]]:
    result = {}
    for name in ("objects", "states", "actions"):
        values = metadata.get(name)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"Checkpoint must contain an explicit unique {name} vocabulary")
        result[name] = values
    if len(result["objects"]) * len(result["states"]) != model.config.num_state_classes:
        raise ValueError("Checkpoint state dimensions disagree with its vocabulary")
    if len(result["actions"]) != model.config.num_actions:
        raise ValueError("Checkpoint action dimensions disagree with its vocabulary")
    return result


def _prepare_input(video_id, dataset, features, sources, provenance, feature_dim, job_fingerprint):
    key = (dataset, video_id)
    if key not in features or key not in sources:
        raise ValueError(f"Missing requested source or extracted features: {video_id}")
    record, base = features[key]
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError(f"Feature artifact is incomplete or unsupported: {video_id}")
    source = _source(sources[key][0], video_id)
    if _source(record, video_id) != source:
        raise ValueError(f"Feature source differs from the original R2 inventory: {video_id}")
    if record.get("feature_provenance") != provenance:
        raise ValueError(f"Feature extractor provenance differs from checkpoint: {video_id}")
    feature_path = _artifact(record, base, "features")
    timestamps_path = _artifact(record, base, "timestamps")
    array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    layout = provenance["layout"]
    if array.ndim != 2 or array.dtype.kind != "f":
        raise ValueError(f"Features must be a floating two-dimensional array: {video_id}")
    temporal = array if layout == "TD" else array.T
    if temporal.shape[1] != feature_dim or temporal.shape[0] < 1:
        raise ValueError(f"Feature dimensions disagree with checkpoint: {video_id}")
    if any(not np.isfinite(temporal[start:start + 1024]).all()
           for start in range(0, len(temporal), 1024)):
        raise ValueError(f"Nonfinite input features: {video_id}")
    times = np.load(timestamps_path, allow_pickle=False)
    if (times.shape != (len(temporal),) or times.dtype.kind != "f" or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(len(times), dtype=np.float64) / 10, rtol=0, atol=1e-7)):
        raise ValueError(f"Timestamps must preserve the complete zero-origin 10 Hz feature grid: {video_id}")
    if record.get("frame_count", len(times)) != len(times):
        raise ValueError(f"Feature metadata frame count differs from its arrays: {video_id}")
    hashes = record.get("artifact_sha256", {})
    inputs = {"job_fingerprint": job_fingerprint, "dataset": dataset, "video_id": video_id,
              "source": source, "features_sha256": hashes.get("features", record.get("features_sha256")),
              "timestamps_sha256": hashes.get("timestamps", record.get("timestamps_sha256"))}
    return {"inputs": inputs, "input_fingerprint": _fingerprint(inputs),
            "feature_path": feature_path, "timestamps_path": timestamps_path,
            "timestamps": times, "timepoints": len(times)}


def _completed(directory: Path, context: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Prediction destination is not a regular completed directory: {directory}")
    receipt_path, prediction_path = directory / "metadata.json", directory / "predictions.npz"
    if not receipt_path.is_file() or not prediction_path.is_file():
        raise ValueError(f"Incomplete existing prediction directory; refusing to overwrite: {directory}")
    receipt = _json(receipt_path)
    if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("producer") != PRODUCER
            or receipt.get("status") != "complete" or receipt.get("video_id") != directory.name
            or receipt.get("inputs") != context["inputs"]
            or receipt.get("input_fingerprint") != context["input_fingerprint"]
            or any(receipt.get(key) != value for key, value in shared.items())):
        raise ValueError(f"Existing predictions have different inputs/provenance: {directory.name}")
    if receipt.get("prediction_file") != "predictions.npz" or prediction_path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Existing prediction receipt refers to an unexpected file")
    if (prediction_path.stat().st_size != receipt.get("prediction_size_bytes")
            or file_sha256(prediction_path) != receipt.get("prediction_sha256")):
        raise ValueError(f"Existing prediction checksum mismatch: {directory.name}")
    vocabularies = shared["vocabularies"]
    expected_states = [context["timepoints"], len(vocabularies["objects"]) * len(vocabularies["states"])]
    expected_actions = [context["timepoints"], len(vocabularies["actions"])]
    if (receipt.get("state_shape") != expected_states or receipt.get("action_shape") != expected_actions
            or receipt.get("timepoints") != context["timepoints"]):
        raise ValueError(f"Existing prediction dimensions differ: {directory.name}")
    return {"status": "complete", "video_id": directory.name,
            "prediction_path": str(prediction_path), "receipt_path": str(receipt_path),
            "prediction_sha256": receipt["prediction_sha256"],
            "prediction_size_bytes": receipt["prediction_size_bytes"],
            "receipt_sha256": file_sha256(receipt_path), "input_fingerprint": context["input_fingerprint"],
            "timepoints": context["timepoints"]}


def _publish(model, context, directory, shared, device):
    array = np.load(context["feature_path"], mmap_mode="r", allow_pickle=False)
    if shared["feature_provenance"]["layout"] == "TD":
        array = array.T
    tensor = torch.as_tensor(np.array(array, dtype=np.float32, copy=True), device=device).unsqueeze(0)
    result = model(tensor)
    states = result.get("probabilities")
    logits = result.get("action_logits")
    times = context["timestamps"]
    state_classes = len(shared["vocabularies"]["objects"]) * len(shared["vocabularies"]["states"])
    action_classes = len(shared["vocabularies"]["actions"])
    if not isinstance(states, torch.Tensor) or states.shape != (1, state_classes, len(times)):
        raise ValueError("Model returned unexpected state probability dimensions")
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[0] < 1
            or logits.shape[1:] != (1, action_classes, len(times))):
        raise ValueError("Model returned unexpected action-logit dimensions")
    state_probabilities = states[0].detach().float().cpu().numpy().T
    action_probabilities = logits[-1, 0].softmax(dim=0).detach().float().cpu().numpy().T
    for values in (state_probabilities, action_probabilities):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Model returned invalid probabilities")
    if not np.allclose(action_probabilities.sum(axis=1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError("Action probabilities do not sum to one")
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.partial-", dir=directory.parent))
    try:
        prediction = staging / "predictions.npz"
        with prediction.open("wb") as handle:
            np.savez_compressed(handle, state_probabilities=state_probabilities,
                                action_probabilities=action_probabilities, timestamps_seconds=times)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {**shared, "schema_version": SCHEMA_VERSION, "producer": PRODUCER, "status": "complete",
            "video_id": directory.name, "inputs": context["inputs"],
            "input_fingerprint": context["input_fingerprint"], "prediction_file": "predictions.npz",
            "prediction_sha256": file_sha256(prediction), "prediction_size_bytes": prediction.stat().st_size,
            "timepoints": len(times), "state_shape": list(state_probabilities.shape),
            "action_shape": list(action_probabilities.shape),
            "timestamps": {"array_key": "timestamps_seconds", "count": len(times),
                           "first_seconds": float(times[0]), "last_seconds": float(times[-1])}}
        with (staging / "metadata.json").open("w") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if directory.exists():
            # Another completed result is reusable only if its inputs match.
            return _completed(directory, context, shared)
        try:
            staging.rename(directory)
        except OSError:
            if not directory.exists():
                raise
            return _completed(directory, context, shared)
        return _completed(directory, context, shared)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@torch.inference_mode()
def batch_predict(
    checkpoint: str | Path,
    feature_manifest: Mapping[str, Any] | str | Path,
    source_inventory: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    video_ids: Sequence[str] | None = None,
    device: str = "cpu",
    progress_callback: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Load one checkpoint and save all requested feature-only predictions.

    IDs default to the checkpoint's reserved test IDs. That list selects video
    identities only: no annotation files, targets or scorers are accepted/read.
    The complete feature timestamp sequence is preserved without thresholding.

    Files are ``predictions/<ID>/predictions.npz`` and ``metadata.json``. A video
    directory appears only after both files are complete. ``index.json`` is
    atomically updated after each video, then progress_callback(record,
    completed, total) runs so a cloud caller can commit its persistent Volume.
    An interrupted callback is safe: the next invocation reuses that output.
    A single writer is allowed per output directory.
    """
    checkpoint = Path(checkpoint).resolve()
    before = checkpoint.stat()
    checkpoint_hash = file_sha256(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed while it was being loaded")
    model.eval()
    metadata = saved["metadata"]
    dataset = metadata.get("dataset")
    if dataset not in ("CMU", "EGTEA"):
        raise ValueError("Checkpoint must declare its dataset")
    provenance = metadata.get("feature_provenance")
    if (not isinstance(provenance, dict) or provenance.get("layout") not in ("TD", "DT")
            or provenance.get("dimension") != model.config.feature_dim):
        raise ValueError("Checkpoint feature provenance is incomplete or inconsistent")
    vocabularies = _vocabularies(metadata, model)
    requested = _ids(list(video_ids) if video_ids is not None else metadata.get("reserved_splits", {}).get("test"))
    feature_doc, feature_base = _document(feature_manifest)
    source_doc, source_base = _document(source_inventory)
    features = _records(feature_doc, feature_base, "feature manifest")
    sources = _records(source_doc, source_base, "source inventory")
    shared = {"dataset": dataset, "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": saved["epoch"],
        "model_config": saved["model_config"], "feature_provenance": provenance, "vocabularies": vocabularies,
        "state_class_order": "object-major, then state; column = object_index * number_of_states + state_index",
        "action_class_order": "checkpoint actions vocabulary order", "output_layout": "time, class",
        "scope": "offline feature-only predictions", "probabilities_thresholded": False}
    job_fingerprint = _fingerprint({"schema_version": SCHEMA_VERSION, "producer": PRODUCER, **shared})
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _single_writer(output):
        index_path = output / "index.json"
        if index_path.exists():
            previous = _json(index_path)
            if (previous.get("schema_version") != SCHEMA_VERSION or previous.get("producer") != PRODUCER
                    or previous.get("job_fingerprint") != job_fingerprint):
                raise ValueError("Existing output index belongs to different checkpoint/provenance")
            if _ids(previous.get("requested_video_ids")) != requested:
                raise ValueError("Existing output index has different requested video IDs; use a new output directory")
        prediction_root = output / "predictions"
        prediction_root.mkdir(exist_ok=True)
        unexpected = [path.name for path in prediction_root.iterdir()
                      if not path.name.startswith(".") and path.name not in requested]
        if unexpected:
            raise ValueError(f"Unexpected existing prediction outputs: {unexpected}")
        # Validate all requested inputs before issuing any model forward call.
        plan = {video_id: _prepare_input(video_id, dataset, features, sources, provenance,
                                        model.config.feature_dim, job_fingerprint) for video_id in requested}
        completed = {}
        for video_id, context in plan.items():
            directory = prediction_root / video_id
            if directory.exists():
                completed[video_id] = _completed(directory, context, shared)
        index = {**shared, "schema_version": SCHEMA_VERSION, "producer": PRODUCER,
                 "job_fingerprint": job_fingerprint, "status": "running", "requested_video_ids": requested,
                 "completed_video_ids": sorted(completed), "per_video": completed}
        write_json(index_path, index)
        for video_id, context in plan.items():
            if video_id in completed:
                continue
            record = _publish(model, context, prediction_root / video_id, shared, device)
            completed[video_id] = record
            index["completed_video_ids"] = sorted(completed)
            write_json(index_path, index)
            if progress_callback is not None:
                progress_callback(dict(record), len(completed), len(requested))
        index["status"] = "complete"
        write_json(index_path, index)
        return index
