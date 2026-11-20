"""Recover completed pipeline stages without starting duplicate compute."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_saved_call(path, from_id):
    """A from_id handle is lazy: its saved ID is usable before hydration."""
    call_id = json.loads(Path(path).read_text()).get("call_id")
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError("Invalid persisted Modal call ID")
    return from_id(call_id), call_id


def completed_feature_index(index_path, prepared_manifest, training_record, *,
                            dataset, source_records, feature_root, annotation_revision,
                            checkpoint_sha256):
    """Check durable publication records without re-running GPU extraction.

    Metadata hashes, array sizes and their published checksums must agree.
    Training/evaluation retain their full content-hash checks; this recovery
    check does not replace them or read/modify model weights.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        return None
    manifest = json.loads(Path(prepared_manifest).read_text())
    training = json.loads(Path(training_record).read_text())
    if _sha256(prepared_manifest) != training.get("manifest_sha256"):
        raise ValueError("Prepared manifest changed after training started")
    if manifest["split_provenance"]["annotation_revision"] != annotation_revision:
        raise ValueError("Prepared annotation revision differs")
    document = json.loads(index_path.read_text())
    rows = document.get("videos", [])
    ids = [row.get("video_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(source_records):
        raise ValueError("Completed feature index has missing/duplicate/unexpected IDs")
    for row in rows:
        video_id = row["video_id"]
        expected_metadata = Path(feature_root) / video_id / "metadata.json"
        path = Path(row["metadata_path"])
        if row.get("dataset") != dataset or path != expected_metadata:
            raise ValueError("Completed feature path/identity differs")
        published = {Path(item["key"]).name: item for item in row["artifacts"]}
        if (set(published) != {"features.npy", "timestamps.npy", "support_start.npy", "support_end.npy", "metadata.json"}
                or _sha256(path) != published["metadata.json"]["sha256"]):
            raise ValueError("Completed feature metadata checksum mismatch")
        metadata = json.loads(path.read_text())
        if (metadata.get("status") != "complete" or metadata.get("dataset") != dataset
                or metadata.get("video_id") != video_id
                or metadata.get("source") != source_records[video_id]["source"]
                or metadata.get("annotation_revision") != annotation_revision
                or metadata["feature_provenance"]["model_assets"]["checkpoint_sha256"] != checkpoint_sha256):
            raise ValueError("Completed feature source/provenance differs")
        for name in ("features", "timestamps", "support_start", "support_end"):
            filename = name + ".npy"
            if metadata.get(name + "_path") != filename:
                raise ValueError("Unexpected feature artifact path")
            artifact = path.parent / filename
            if (not artifact.is_file() or artifact.stat().st_size != published[filename]["size_bytes"]
                    or metadata["artifact_sha256"][name] != published[filename]["sha256"]):
                raise ValueError("Published feature artifact is missing or inconsistent")
    return document
