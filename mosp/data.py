"""Explicit, split-safe training adapters for released MOSCATO pseudo-labels.

The version-1 manifest reserves validation/test IDs for an external scorer and
provides source files and index maps only for training videos. It never infers
alignment from array lengths. Features are [D,T], state targets [O*S,T], and
action supervision [T]. Ground-truth files supply exclusion IDs only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SCHEMA_VERSION = 1
IGNORE_ACTION = -100
SPLITS = ("train", "validation", "test")


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _path(base: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} requires an explicit file path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path.resolve()


@dataclass(frozen=True)
class Vocabulary:
    names: tuple[str, ...]
    name_to_index: dict[str, int]
    source_id_to_index: dict[int, int]
    snapshot: dict[str, Any]


def _vocabulary(path: Path, prefix: str) -> Vocabulary:
    raw = _json(path)
    mapping = raw.get(f"{prefix}2i")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Missing nonempty {prefix}2i vocabulary: {path}")
    by_id: dict[int, list[str]] = {}
    for name, value in mapping.items():
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid vocabulary ID for {name!r}: {path}") from exc
        if isinstance(value, bool) or isinstance(value, float) or index < 0:
            raise ValueError(f"Vocabulary IDs must be nonnegative integers: {path}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid vocabulary label: {path}")
        by_id.setdefault(index, []).append(name)
    source_ids = sorted(by_id)
    dense = {old: new for new, old in enumerate(source_ids)}
    inverse = raw.get(f"i2{prefix}", {})
    names = []
    for old in source_ids:
        candidate = inverse.get(str(old)) if isinstance(inverse, dict) else None
        names.append(candidate if candidate in by_id[old] else sorted(by_id[old])[0])
    name_to_index = {name: dense[old] for old, labels in by_id.items() for name in labels}
    aliases = raw.get("all2one", {})
    if not isinstance(aliases, dict):
        raise ValueError(f"all2one must be a label-to-label object: {path}")
    for alias, canonical in aliases.items():
        if canonical not in name_to_index:
            raise ValueError(f"Alias {alias!r} targets unknown label {canonical!r}: {path}")
        target = name_to_index[canonical]
        if alias in name_to_index and name_to_index[alias] != target:
            raise ValueError(f"Conflicting alias {alias!r}: {path}")
        name_to_index[alias] = target
    snapshot = {
        "names": names,
        "source_ids": source_ids,
        "name_to_index": dict(sorted(name_to_index.items())),
    }
    return Vocabulary(tuple(names), name_to_index, dense, snapshot)


@dataclass
class Sample:
    video_id: str
    features: np.ndarray  # [D,T], float32
    states: np.ndarray  # [O*S,T], float32; object-major, state-minor
    actions: np.ndarray  # [T], int64; unknown/absent supervision is -100
    action_segments: tuple[tuple[int, int, int], ...] = ()  # [start,end), action
    correlation_segments: tuple[tuple[int, int, int, int, int], ...] = ()
    # before [start,end), after [start,end), action; midpoint on SOURCE timeline


class PreparedDataset:
    """Load training pseudo-labels; other split IDs are reserved metadata only.

    All ground-truth IDs in the declared ground-truth file are excluded from
    training, including IDs not selected for the reserved test list. Ground-truth
    label contents are never cached or converted into targets.
    """

    def __init__(self, manifest: str | Path, split: str):
        if split != "train":
            raise ValueError("PreparedDataset loads training data only; validation/test IDs are reserved for an external scorer")
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.manifest = _json(self.manifest_path)
        self.split = split
        base = self.manifest_path.parent
        data = self.manifest
        if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Manifest must explicitly set schema_version to 1")
        if data.get("dataset") not in ("CMU", "EGTEA", "EgoPER"):
            raise ValueError("dataset must identify CMU, EGTEA or EgoPER")
        if data["dataset"] == "EgoPER" and data.get("task") not in (
            "coffee", "oatmeal", "pinwheels", "quesadilla", "tea"
        ):
            raise ValueError("EgoPER requires one explicit task per manifest")
        provenance = data.get("split_provenance", {})
        if not isinstance(provenance, dict) or not str(provenance.get("description", "")).strip():
            raise ValueError("split_provenance.description must document the chosen split")
        if type(provenance.get("paper_split_verified")) is not bool:
            raise ValueError("split_provenance.paper_split_verified must be explicit")
        if provenance["paper_split_verified"] and not provenance.get("source"):
            raise ValueError("A verified paper split requires split_provenance.source evidence")

        split_lists = data.get("splits", {})
        if not isinstance(split_lists, dict) or set(split_lists) != set(SPLITS):
            raise ValueError("splits must explicitly contain train, validation and test lists")
        seen: set[str] = set()
        for name in SPLITS:
            ids = split_lists[name]
            if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids):
                raise ValueError(f"splits.{name} must be a list of nonempty video IDs")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate video IDs in {name}")
            overlap = seen.intersection(ids)
            if overlap:
                raise ValueError(f"Split overlap involving {name}: {sorted(overlap)}")
            seen.update(ids)
        self.video_ids = list(split_lists[split])
        records = data.get("videos", {})
        if not isinstance(records, dict) or set(records) != set(self.video_ids):
            raise ValueError("videos must contain exactly the train IDs; reserved splits need no video records")

        source_values = data.get("annotations", {})
        if not isinstance(source_values, dict):
            raise ValueError("annotations must declare source paths")
        self.sources = {
            key: _path(base, source_values.get(key), f"annotations.{key}")
            for key in ("pseudo_labels", "ground_truth")
        }
        if "action_metadata" in source_values:
            self.sources["metadata"] = _path(base, source_values["action_metadata"], "action metadata")
        # Only identifiers are retained here. No GT target can become a train sample.
        ground_truth_ids = set(_json(self.sources["ground_truth"]))
        leaked = set(split_lists["train"]) & ground_truth_ids
        if leaked:
            raise ValueError(f"Training contains manually annotated ground-truth video IDs: {sorted(leaked)}")
        self.ground_truth_ids = frozenset(ground_truth_ids)

        vocab_paths = data.get("vocabularies", {})
        if not isinstance(vocab_paths, dict):
            raise ValueError("vocabularies must declare objects, states and actions paths")
        self.objects = _vocabulary(_path(base, vocab_paths.get("objects"), "object vocabulary"), "o")
        self.states = _vocabulary(_path(base, vocab_paths.get("states"), "state vocabulary"), "s")
        self.actions = _vocabulary(_path(base, vocab_paths.get("actions"), "action vocabulary"), "a")
        feature = data.get("feature_provenance", {})
        if not isinstance(feature, dict):
            raise ValueError("feature_provenance must be an object")
        for key in ("model", "timeline"):
            if not isinstance(feature.get(key), str) or not feature[key].strip():
                raise ValueError(f"feature_provenance.{key} is required")
        if type(feature.get("dimension")) is not int or feature["dimension"] <= 0:
            raise ValueError("feature_provenance.dimension must be a positive integer")
        if feature.get("layout") not in ("DT", "TD"):
            raise ValueError("feature_provenance.layout must explicitly be DT or TD")
        for video_id in self.video_ids:
            record = records[video_id]
            if not isinstance(record, dict):
                raise ValueError(f"Video record must be an object: {video_id}")
            if record.get("state_source") != "pseudo_labels":
                raise ValueError(f"Training labels must come from pseudo_labels: {video_id}")
            if any(key.startswith("evaluation_") for key in record):
                raise ValueError(f"Evaluation fields are not supported by the training adapter: {video_id}")
            _path(base, record.get("feature_path"), f"features for {video_id}")
            if "state_indices" not in record:
                raise ValueError(f"Explicit state_indices mapping required: {video_id}")
            action_source = record.get("action_source")
            if action_source not in ("pseudo_labels", "metadata"):
                raise ValueError(f"Training TAS supervision requires an explicit pseudo_labels or metadata action source: {video_id}")
            if action_source not in self.sources:
                raise ValueError(f"Missing {action_source} action source: {video_id}")
            if "action_indices" not in record:
                raise ValueError(f"Explicit action_indices mapping required: {video_id}")
            if record.get("action_segment_end") not in ("inclusive", "exclusive"):
                raise ValueError(f"Explicit action_segment_end convention required: {video_id}")

        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset": data["dataset"],
            "task": data.get("task"),
            "objects": list(self.objects.names),
            "states": list(self.states.names),
            "actions": list(self.actions.names),
            "feature_dimension": feature["dimension"],
            "feature_provenance": dict(feature),
            "split_provenance": dict(provenance),
            "vocabulary_snapshot": {
                "objects": self.objects.snapshot,
                "states": self.states.snapshot,
                "actions": self.actions.snapshot,
            },
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }
        self._annotation_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.video_ids)

    def __iter__(self) -> Iterator[Sample]:
        for index in range(len(self)):
            yield self[index]

    def _source(self, name: str) -> dict[str, Any]:
        if name not in ("pseudo_labels", "metadata"):
            raise ValueError("Only training pseudo-label/action sources can be cached; ground truth supplies exclusion IDs only")
        if name not in self._annotation_cache:
            self._annotation_cache[name] = _json(self.sources[name])
        return self._annotation_cache[name]

    def _indices(self, value: Any, length: int, description: str) -> np.ndarray:
        if isinstance(value, str):
            path = _path(self.manifest_path.parent, value, description)
            indices = np.load(path, allow_pickle=False)
        elif isinstance(value, list):
            if any(type(item) is not int for item in value):
                raise ValueError(f"{description} must contain integer indices")
            indices = np.asarray(value, dtype=np.int64)
        else:
            raise ValueError(f"{description} must be an integer list or an explicit .npy path")
        if indices.ndim != 1 or len(indices) != length or indices.dtype.kind not in "iu":
            raise ValueError(f"{description} must be a 1-D integer array with {length} entries")
        if np.any(indices < 0) or np.any(indices[1:] < indices[:-1]):
            raise ValueError(f"{description} must be nonnegative and chronological")
        if np.any(indices > np.iinfo(np.int64).max):
            raise ValueError(f"{description} exceeds supported integer range")
        return indices.astype(np.int64, copy=False)

    def __getitem__(self, index: int) -> Sample:
        video_id = self.video_ids[index]
        record = self.manifest["videos"][video_id]
        path = _path(self.manifest_path.parent, record["feature_path"], f"features for {video_id}")
        features = np.load(path, allow_pickle=False)
        if features.ndim != 2 or features.dtype.kind not in "fiu":
            raise ValueError(f"Features must be a numeric 2-D array: {video_id}")
        if self.metadata["feature_provenance"]["layout"] == "TD":
            features = features.T
        if features.shape[0] != self.metadata["feature_dimension"] or features.shape[1] == 0:
            raise ValueError(f"Feature shape does not match declared dimension/layout: {video_id}: {features.shape}")
        features = np.ascontiguousarray(features, dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"Features contain nonfinite values: {video_id}")
        length = features.shape[1]
        state_indices = self._indices(record["state_indices"], length, f"state_indices for {video_id}")
        source = self._source(record["state_source"])
        if video_id not in source or not isinstance(source[video_id].get("state"), dict):
            raise ValueError(f"Missing per-object state annotations: {video_id}")
        state_ann = source[video_id]["state"]
        if not state_ann:
            raise ValueError(f"Empty state annotation timeline: {video_id}")
        target = self._state_targets(state_ann, state_indices, video_id)
        actions = np.full(length, IGNORE_ACTION, dtype=np.int64)
        mapped_segments: list[tuple[int, int, int]] = []
        correlation_segments: list[tuple[int, int, int, int, int]] = []
        action_source = record["action_source"]
        if action_source != "none":
            action_indices = self._indices(record["action_indices"], length, f"action_indices for {video_id}")
            source = self._source(action_source)
            if video_id not in source:
                raise ValueError(f"Missing action annotations: {video_id}")
            key = "action_annts" if action_source == "metadata" else "action"
            segments = source[video_id].get(key)
            if not isinstance(segments, list):
                raise ValueError(f"Missing explicit {key} segments: {video_id}")
            previous_end = 0
            for segment in segments:
                if not isinstance(segment, list) or len(segment) < 3:
                    raise ValueError(f"Malformed action segment: {video_id}: {segment!r}")
                start, end, name = segment[:3]
                if type(start) is not int or type(end) is not int or start < 0:
                    raise ValueError(f"Action boundaries must be nonnegative integer indices: {video_id}")
                if record["action_segment_end"] == "inclusive":
                    end += 1
                if end <= start or start < previous_end:
                    raise ValueError(f"Action segments must be ordered, nonempty and nonoverlapping: {video_id}")
                previous_end = end
                action = self.actions.name_to_index.get(name, IGNORE_ACTION)
                # Unknown names stay ignored; numeric IDs only resolve unnamed segments.
                if name in (None, "") and len(segment) >= 4:
                    try:
                        action = self.actions.source_id_to_index.get(int(segment[3]), IGNORE_ACTION)
                    except (TypeError, ValueError):
                        pass
                left = int(np.searchsorted(action_indices, start, side="left"))
                right = int(np.searchsorted(action_indices, end, side="left"))
                if right > left:
                    actions[left:right] = action
                    if action != IGNORE_ACTION:
                        mapped_segments.append((left, right, action))
                        # Preserve the annotated temporal midpoint even for
                        # nonuniform feature sampling. Never split by sample count.
                        midpoint_floor = (start + end - 1) // 2
                        midpoint_ceil = (start + end) // 2
                        before_end = int(np.searchsorted(action_indices, midpoint_floor, side="right"))
                        after_start = int(np.searchsorted(action_indices, midpoint_ceil, side="left"))
                        correlation_segments.append((left, before_end, after_start, right, action))
        return Sample(video_id, features, target, actions, tuple(mapped_segments), tuple(correlation_segments))

    def _state_targets(self, state_ann: dict[str, Any], indices: np.ndarray, video_id: str) -> np.ndarray:
        target = np.zeros((len(self.objects.names) * len(self.states.names), len(indices)), dtype=np.float32)
        for object_name, timeline in state_ann.items():
            if object_name not in self.objects.name_to_index:
                raise ValueError(f"Unknown object label {object_name!r}: {video_id}")
            if not isinstance(timeline, list) or indices[-1] >= len(timeline):
                raise ValueError(f"state_indices exceed the {object_name!r} annotation timeline: {video_id}")
            offset = self.objects.name_to_index[object_name] * len(self.states.names)
            for t, source_index in enumerate(indices):
                labels = timeline[int(source_index)]
                if not isinstance(labels, list):
                    raise ValueError(f"Expected a state list at frame {source_index}: {video_id}")
                for state_name in labels:
                    if state_name not in self.states.name_to_index:
                        raise ValueError(f"Unknown state label {state_name!r}: {video_id}")
                    target[offset + self.states.name_to_index[state_name], t] = 1.0
        return target


def compute_state_action_correlations(dataset: PreparedDataset) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm-1 before/after counts from TRAIN labels only, as [A,O*S].

    Each explicitly annotated action retains its boundary even if a neighbour
    has the same label. Counts use the supplied feature/label alignment and the
    midpoint on the original action timeline, including a sampled exact midpoint
    in both closed halves. This is an explicit feature-grid estimate, not an
    implicit resampling of native annotations. Zero-count action rows remain
    zero. No validation/test labels are consulted.
    """
    if dataset.split != "train":
        raise ValueError("State-action correlations must use the train split only")
    shape = (len(dataset.actions.names), len(dataset.objects.names) * len(dataset.states.names))
    before = np.zeros(shape, dtype=np.float64)
    after = np.zeros(shape, dtype=np.float64)
    for sample in dataset:
        for start, before_end, after_start, end, action in sample.correlation_segments:
            before[action] += sample.states[:, start:before_end].sum(axis=1)
            after[action] += sample.states[:, after_start:end].sum(axis=1)
    for counts in (before, after):
        totals = counts.sum(axis=1, keepdims=True)
        np.divide(counts, totals, out=counts, where=totals != 0)
    return before.astype(np.float32), after.astype(np.float32)


def validate_manifest(manifest: str | Path, *, check_features: bool = True) -> dict[str, Any]:
    """Check training samples and reserved-ID separation without scoring anything."""
    dataset = PreparedDataset(manifest, "train")
    frames = sum(sample.features.shape[1] for sample in dataset) if check_features else None
    return {
        "schema_version": SCHEMA_VERSION,
        "training": {"videos": len(dataset), "frames": frames},
        "reserved_ids": {name: len(dataset.manifest["splits"][name]) for name in ("validation", "test")},
        "ground_truth_exclusion_ids": len(dataset.ground_truth_ids),
        "metadata": dataset.metadata,
        "features_checked": check_features,
    }
