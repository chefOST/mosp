"""Validate cloud feature artifacts and prepare the current curated training split.

This module never fetches videos, generates labels, or scores held-out targets.
Feature anchors, rather than the I3D window's trailing support, select the
10 Hz pseudo-label samples. Ground truth supplies exclusion IDs and a separate
timeline audit only; its arrays never become training targets.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np

from .data import validate_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _document(value: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(value, Mapping):
        return dict(value), Path.cwd()
    path = Path(value).resolve()
    return _json(path), path.parent


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing explicit {name} path")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path.resolve()


def _records(document: dict[str, Any], base: Path, name: str) -> dict[tuple[str, str], tuple[dict[str, Any], Path]]:
    rows = document.get("videos")
    if not isinstance(rows, list):
        raise ValueError(f"{name}.videos must be an explicit list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Every {name} video must be an object")
        record_base = base
        if "metadata_path" in row:
            path = _path(base, row["metadata_path"], "feature metadata")
            metadata = _json(path)
            for key in ("dataset", "video_id"):
                if key in row and row[key] != metadata.get(key):
                    raise ValueError(f"Feature metadata {key} mismatch")
            row, record_base = metadata, path.parent
        elif "artifact_root" in row:
            record_base = Path(row["artifact_root"])
            if not record_base.is_absolute():
                record_base = base / record_base
        key = (row.get("dataset"), row.get("video_id"))
        if key[0] not in ("CMU", "EGTEA") or not isinstance(key[1], str) or not key[1]:
            raise ValueError(f"Invalid dataset/video identity in {name}")
        if key in result:
            raise ValueError(f"Duplicate {name} video: {key}")
        result[key] = (row, record_base)
    return result


def _source(row: dict[str, Any], video_id: str) -> dict[str, Any]:
    source = row.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"Invalid original R2 source: {video_id}")
    selected = {key: source.get(key) for key in ("bucket", "key", "sha256", "size_bytes")}
    if not all(isinstance(selected[key], str) and selected[key] for key in ("bucket", "key")):
        raise ValueError(f"Missing original R2 bucket/key: {video_id}")
    if Path(selected["key"]).stem != video_id or ".." in Path(selected["key"]).parts:
        raise ValueError(f"R2 key does not identify the exact video: {video_id}")
    if not isinstance(selected["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", selected["sha256"]):
        raise ValueError(f"Missing original video SHA-256: {video_id}")
    if type(selected["size_bytes"]) is not int or selected["size_bytes"] <= 0:
        raise ValueError(f"Missing original video size: {video_id}")
    return selected


def _annotation_paths(root: Path, dataset: str) -> dict[str, Path]:
    stem = dataset.lower()
    return {
        "pseudo_labels": root / f"annotations/pseudo_labels/{dataset}/pseudo_annotations_poc_new_{stem}.json",
        "ground_truth": root / f"annotations/ground_truth/{dataset}/gt_annotations_{stem}.json",
        "indexed_ground_truth": root / f"annotations/ground_truth/{dataset}/gt_annotations_{stem}_idx.json",
        "metadata": root / f"vocabulary/{dataset}/files_dict.json",
        "objects": root / f"vocabulary/{dataset}/{'obj_dict.json' if dataset == 'EGTEA' else 'object_dict.json'}",
        "states": root / f"vocabulary/{dataset}/state_dict.json",
        "actions": root / f"vocabulary/{dataset}/action_dict.json",
    }


def _pin_annotations(root: Path, paths: dict[str, Path], revision: str,
                     expected: Mapping[str, str] | None) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("annotation_revision must be a pinned 40-character Git commit")
    actual = {}
    for name, path in paths.items():
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        if expected is not None:
            wanted = expected.get(relative, expected.get("MOSCATO/" + relative, expected.get(name)))
            if wanted != digest:
                raise ValueError(f"Pinned annotation hash mismatch or missing hash: {relative}")
        else:
            # Read only locally available commit objects; do not trigger lazy fetch.
            env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
            try:
                repository = Path(subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                    env=env, stderr=subprocess.DEVNULL, text=True).strip())
                blob = subprocess.check_output(
                    ["git", "-C", str(repository), "show", f"{revision}:{path.relative_to(repository).as_posix()}"],
                    env=env, stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, ValueError) as exc:
                raise ValueError("Supply pinned annotation_sha256 hashes when Git commit objects are unavailable") from exc
            if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                oid = re.search(rb"(?m)^oid sha256:([0-9a-f]{64})$", blob)
                size = re.search(rb"(?m)^size ([0-9]+)$", blob)
                if not oid or not size or oid[1].decode() != digest or int(size[1]) != path.stat().st_size:
                    raise ValueError(f"Pinned LFS annotation mismatch: {relative}")
            elif hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError(f"Annotation differs from pinned commit: {relative}")
        actual[relative] = digest
    return actual


def _timeline_length(record: dict[str, Any], video_id: str) -> int:
    states = record.get("state")
    if not isinstance(states, dict) or not states or any(not isinstance(x, list) for x in states.values()):
        raise ValueError(f"Missing per-object state timelines: {video_id}")
    lengths = {len(value) for value in states.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"Inconsistent per-object state timeline lengths: {video_id}")
    length = lengths.pop()
    if "object" in record and (not isinstance(record["object"], list) or len(record["object"]) != length):
        raise ValueError(f"Object/state timeline length mismatch: {video_id}")
    return length


def _artifact(record: dict[str, Any], base: Path, name: str) -> Path:
    path = _path(base, record.get(f"{name}_path"), name)
    hashes = record.get("artifact_sha256", {})
    wanted = hashes.get(name, record.get(f"{name}_sha256")) if isinstance(hashes, dict) else None
    if not wanted or _sha256(path) != wanted:
        raise ValueError(f"Feature artifact checksum mismatch or missing hash: {name}")
    return path


def _video_metadata(row: dict[str, Any]) -> dict[str, Any]:
    video = row.get("video", row.get("media", {}))
    if isinstance(video, dict) and "streams" in video:
        streams = video["streams"]
        video = next((s for s in streams if s.get("codec_type", "video") == "video"), {})
    return video if isinstance(video, dict) else {}


def prepare_training_manifest(
    annotation_root: str | Path,
    dataset: str,
    annotation_revision: str,
    source_inventory: Mapping[str, Any] | str | Path,
    feature_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    annotation_sha256: Mapping[str, str] | None = None,
    action_segment_end: str = "exclusive",
) -> Path:
    """Publish a validated manifest from pinned labels and cloud feature records.

    Inventory and feature documents contain ``videos`` lists. Each video has
    ``dataset``, ``video_id`` and ``source`` (bucket, key, sha256, size_bytes).
    A feature row can name ``metadata_path``; otherwise its artifact paths are
    relative to the feature document or its explicit ``artifact_root``. Feature
    metadata includes annotation_revision, feature_provenance, features_path,
    timestamps_path, support_start_path, support_end_path and artifact_sha256.
    All feature anchors must be the complete zero-origin 10 Hz sequence.

    At most two tail samples may differ from the released pseudo-label length;
    feature samples past that length are explicitly removed and counted. This
    accounts for frame rounding only and never stretches or shifts annotations.
    Original feature artifacts remain unchanged. A second call refuses to
    overwrite a published manifest; callers may validate/reuse that artifact.
    """
    if dataset not in ("CMU", "EGTEA"):
        raise ValueError("Cloud preparation currently supports CMU and EGTEA only")
    if action_segment_end not in ("inclusive", "exclusive"):
        raise ValueError("Explicit inclusive/exclusive action_segment_end required")
    root = Path(annotation_root).resolve()
    if (root / "MOSCATO").is_dir():
        root = root / "MOSCATO"
    paths = _annotation_paths(root, dataset)
    annotation_hashes = _pin_annotations(root, paths, annotation_revision, annotation_sha256)
    labels, metadata = _json(paths["pseudo_labels"]), _json(paths["metadata"])
    ground_truth, indexed_gt = _json(paths["ground_truth"]), _json(paths["indexed_ground_truth"])
    if set(ground_truth) != set(indexed_gt):
        raise ValueError("Named/indexed ground-truth IDs differ")
    if set(labels) != set(metadata) or not set(ground_truth) <= set(metadata):
        raise ValueError("Pseudo-label, metadata and ground-truth membership disagree")
    train_ids, test_ids = sorted(set(metadata) - set(ground_truth)), sorted(ground_truth)
    if not train_ids:
        raise ValueError("The curated training pool is empty")
    source_doc, source_base = _document(source_inventory)
    feature_doc, feature_base = _document(feature_manifest)
    sources = _records(source_doc, source_base, "source inventory")
    features = _records(feature_doc, feature_base, "feature manifest")
    expected_ids = set(metadata)
    unexpected = [k for ds, k in features if ds == dataset and k not in expected_ids]
    if unexpected:
        raise ValueError(f"Unknown feature video IDs: {unexpected}")
    missing = [k for k in train_ids if (dataset, k) not in sources or (dataset, k) not in features]
    if missing:
        raise ValueError(f"Missing training sources/features: {missing}")
    output = Path(output_dir).resolve()
    final = output / "manifest.json"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {final}")
    videos, audits, common_provenance = {}, {}, None
    output.mkdir(parents=True, exist_ok=True)
    for video_id in train_ids:
        record, base = features[(dataset, video_id)]
        source = _source(sources[(dataset, video_id)][0], video_id)
        if _source(record, video_id) != source:
            raise ValueError(f"Feature source differs from verified R2 inventory: {video_id}")
        if record.get("annotation_revision") != annotation_revision:
            raise ValueError(f"Feature annotation revision mismatch: {video_id}")
        if Path(metadata[video_id].get("video_path", "")).stem != video_id:
            raise ValueError(f"Metadata video identity conflict remains: {video_id}")
        length = _timeline_length(labels[video_id], video_id)
        if metadata[video_id].get("feature_len") != length:
            raise ValueError(f"Pseudo-label length differs from metadata feature_len: {video_id}")
        if labels[video_id].get("action") != metadata[video_id].get("action_annts"):
            raise ValueError(f"Pseudo-label and metadata actions differ: {video_id}")
        provenance = record.get("feature_provenance", {})
        if (not isinstance(provenance, dict) or provenance.get("dimension") != 2048
                or provenance.get("layout") != "TD" or not provenance.get("model") or not provenance.get("timeline")):
            raise ValueError(f"Expected declared TD/2048 feature provenance: {video_id}")
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise ValueError(f"Feature extractor provenance differs across videos: {video_id}")
        feature_path = _artifact(record, base, "features")
        time_path = _artifact(record, base, "timestamps")
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        times = np.load(time_path, allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 2048 or array.shape[0] == 0 or array.dtype.kind != "f":
            raise ValueError(f"Features must be nonempty floating [T,2048]: {video_id}")
        frames = array.shape[0]
        if any(not np.isfinite(array[start:start + 1024]).all() for start in range(0, frames, 1024)):
            raise ValueError(f"Nonfinite features: {video_id}")
        if (times.shape != (frames,) or times.dtype.kind != "f" or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(frames, dtype=np.float64) / 10, rtol=0, atol=1e-7)):
            raise ValueError(f"Feature timestamps must be exact zero-origin 10 Hz anchors: {video_id}")
        if abs(frames - length) > 2:
            raise ValueError(f"Feature/pseudo-label length difference exceeds two-sample rounding allowance: {video_id}")
        support = {}
        for name in ("support_start", "support_end"):
            path = _artifact(record, base, name)
            values = np.load(path, allow_pickle=False)
            if (values.shape != (frames,) or values.dtype.kind != "f" or not np.isnan(values[0])
                    or not np.isfinite(values[1:]).all()):
                raise ValueError(f"Invalid trailing-window {name} timestamps: {video_id}")
            support[name] = values
        if (np.any(support["support_start"][1:] < 0)
                or np.any(support["support_start"][1:] > support["support_end"][1:])
                or np.any(support["support_end"][1:] >= times[1:])):
            raise ValueError(f"Feature support must precede its output anchor: {video_id}")
        retained = min(frames, length)
        if retained < 2:
            raise ValueError(f"At least two training samples are required: {video_id}")
        indices = np.rint(times[:retained] * 10).astype(np.int64)
        if frames > length:
            trimmed = output / f"{video_id}.features.npy"
            if trimmed.exists():
                raise FileExistsError(f"Refusing to overwrite trimmed feature artifact: {trimmed}")
            np.save(trimmed, np.asarray(array[:retained]))
            feature_path = trimmed
        videos[video_id] = {"feature_path": str(feature_path), "state_source": "pseudo_labels",
            "state_indices": indices.tolist(), "action_source": "pseudo_labels",
            "action_indices": indices.tolist(), "action_segment_end": action_segment_end}
        audits[video_id] = {"source": source, "source_feature_path": str(_path(base, record["features_path"], "features")),
            "feature_sha256": _sha256(feature_path), "timestamps_sha256": _sha256(time_path),
            "input_feature_samples": frames, "retained_samples": retained,
            "trimmed_feature_tail_samples": frames - retained, "unused_pseudo_tail_samples": length - retained,
            "alignment": "10 Hz output anchors select pseudo-state/action indices; trailing support does not shift labels"}
    evaluation_audit = {}
    for video_id in test_ids:
        count = _timeline_length(ground_truth[video_id], video_id)
        if _timeline_length(indexed_gt[video_id], video_id) != count:
            raise ValueError(f"Named/indexed ground-truth lengths differ: {video_id}")
        raw = _video_metadata(sources.get((dataset, video_id), ({}, root))[0])
        try:
            fps, raw_count = float(Fraction(raw["r_frame_rate"])), int(raw["nb_frames"])
            matches = fps > 0 and raw_count == count
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            fps, raw_count, matches = None, None, False
        evaluation_audit[video_id] = {"ground_truth_samples": count, "source_raw_frames": raw_count,
            "source_fps": fps, "length_matches_raw_frame_count": matches,
            "status": "raw-frame-length grid; exact temporal offset remains unverified" if matches else "evaluation alignment unverified",
            "used_as_training_target": False}
    manifest = {"schema_version": 1, "dataset": dataset,
        "split_provenance": {"description": "Current curated metadata/pseudo-label pool excluding every manual-GT ID; no official validation split",
            "paper_split_verified": False, "scope": "reimplementation/current curated split",
            "annotation_repository": "chefOST/chefOST-data", "annotation_revision": annotation_revision,
            "annotation_sha256": annotation_hashes, "extractor_assumption": "EgoPER RGB I3D extractor with explicit 10 Hz anchors and trailing support"},
        "vocabularies": {key: str(paths[key]) for key in ("objects", "states", "actions")},
        "annotations": {key: str(paths[key]) for key in ("pseudo_labels", "ground_truth")},
        "feature_provenance": common_provenance,
        "splits": {"train": train_ids, "validation": [], "test": test_ids}, "videos": videos,
        "input_audit": audits, "reserved_ground_truth_timeline_audit": evaluation_audit}
    temporary = output / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    # The existing adapter validates every actual training target/action/map.
    # Its implementation explicitly refuses validation/test target loading.
    validate_manifest(temporary, check_features=True)
    temporary.replace(final)
    return final
