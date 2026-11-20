import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mosp.cloud_data import prepare_training_manifest
from mosp.data import PreparedDataset, compute_state_action_correlations


REVISION = "a" * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "MOSCATO"
    vocabulary = root / "vocabulary/CMU"
    actions = [[0, 2, "pour", "0"], [2, 4, "cut", "1"]]
    training = {"state": {"cup": [["empty"], ["empty"], ["full"], ["full"]]},
                "object": [["cup"]] * 4, "action": actions}
    # These unknown held-out state names must never become training targets.
    heldout = {"state": {"cup": [["held-out-only"]] * 6}, "object": [["cup"]] * 6}
    paths = {
        "pseudo_labels": root / "annotations/pseudo_labels/CMU/pseudo_annotations_poc_new_cmu.json",
        "ground_truth": root / "annotations/ground_truth/CMU/gt_annotations_cmu.json",
        "indexed_ground_truth": root / "annotations/ground_truth/CMU/gt_annotations_cmu_idx.json",
        "metadata": vocabulary / "files_dict.json",
        "objects": vocabulary / "object_dict.json",
        "states": vocabulary / "state_dict.json",
        "actions": vocabulary / "action_dict.json",
    }
    write_json(paths["pseudo_labels"], {"training": training, "test": heldout})
    write_json(paths["ground_truth"], {"test": heldout})
    write_json(paths["indexed_ground_truth"], {"test": {"state": {"0": [[999]] * 6}, "object": [[0]] * 6}})
    write_json(paths["metadata"], {
        "training": {"video_path": "/original/training.mp4", "feature_len": 4, "action_annts": actions},
        "test": {"video_path": "/original/test.mp4", "feature_len": 3, "action_annts": []}})
    write_json(paths["objects"], {"o2i": {"cup": 0}})
    write_json(paths["states"], {"s2i": {"empty": 0, "full": 1}})
    write_json(paths["actions"], {"a2i": {"pour": 0, "cut": 1}})
    artifact = tmp_path / "features/training"
    artifact.mkdir(parents=True)
    values = {"features": np.ones((4, 2048), dtype=np.float32),
              "timestamps": np.arange(4, dtype=np.float64) / 10,
              "support_start": np.array([np.nan, 0, .1, 0]),
              "support_end": np.array([np.nan, 0, .1, .2])}
    source = {"bucket": "fixture", "key": "videos/training.mp4", "sha256": "b" * 64, "size_bytes": 1000}
    record = {"dataset": "CMU", "video_id": "training", "annotation_revision": REVISION,
              "source": source, "artifact_root": str(artifact), "artifact_sha256": {},
              "feature_provenance": {"model": "EgoPER fixture", "dimension": 2048, "layout": "TD",
                                     "timeline": "output anchors i/10; trailing support recorded separately"}}
    for name, value in values.items():
        path = artifact / f"{name}.npy"
        np.save(path, value)
        record[f"{name}_path"] = path.name
        record["artifact_sha256"][name] = sha(path)
    return {"annotation_root": root, "dataset": "CMU", "annotation_revision": REVISION,
        "annotation_sha256": {key: sha(path) for key, path in paths.items()},
        "source_inventory": {"videos": [
            {"dataset": "CMU", "video_id": "training", "source": dict(source)},
            {"dataset": "CMU", "video_id": "test", "source": {**source, "key": "videos/test.mp4"},
             "video": {"r_frame_rate": "30/1", "nb_frames": "6"}}]},
        "feature_manifest": {"videos": [record]}, "output_dir": tmp_path / "prepared"}


def rewrite_artifact(inputs, name, value):
    record = inputs["feature_manifest"]["videos"][0]
    path = Path(record["artifact_root"]) / record[f"{name}_path"]
    np.save(path, value)
    record["artifact_sha256"][name] = sha(path)


def test_disjoint_curated_split_uses_anchors_and_training_only_correlations(inputs):
    path = prepare_training_manifest(**inputs)
    manifest = json.loads(path.read_text())
    assert manifest["splits"] == {"train": ["training"], "validation": [], "test": ["test"]}
    assert not manifest["split_provenance"]["paper_split_verified"]
    assert set(manifest["videos"]) == {"training"}
    # Do not shift labels backward to the trailing I3D support window.
    assert manifest["videos"]["training"]["state_indices"] == [0, 1, 2, 3]
    dataset = PreparedDataset(path, "train")
    sample = dataset[0]
    np.testing.assert_array_equal(sample.states, [[1, 1, 0, 0], [0, 0, 1, 1]])
    before, after = compute_state_action_correlations(dataset)
    np.testing.assert_array_equal(before, np.eye(2))
    np.testing.assert_array_equal(after, np.eye(2))
    assert "ground_truth" not in dataset._annotation_cache
    audit = manifest["reserved_ground_truth_timeline_audit"]["test"]
    assert audit["ground_truth_samples"] == 6
    assert audit["source_fps"] == 30
    assert audit["length_matches_raw_frame_count"]
    assert audit["used_as_training_target"] is False


@pytest.mark.parametrize("field,value", [("sha256", "c" * 64), ("size_bytes", 999), ("key", "videos/other.mp4")])
def test_rejects_feature_receipt_from_different_original_video(inputs, field, value):
    inputs["feature_manifest"]["videos"][0]["source"][field] = value
    with pytest.raises(ValueError, match="source differs|exact video"):
        prepare_training_manifest(**inputs)


def test_rejects_label_drift_from_pinned_snapshot(inputs):
    path = inputs["annotation_root"] / "vocabulary/CMU/files_dict.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Pinned annotation hash"):
        prepare_training_manifest(**inputs)


def test_rejects_corrupt_feature_artifact_before_use(inputs):
    record = inputs["feature_manifest"]["videos"][0]
    (Path(record["artifact_root"]) / record["features_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact checksum"):
        prepare_training_manifest(**inputs)


@pytest.mark.parametrize("times", [np.array([.1, .2, .3, .4]), np.array([0, .1, .1, .3]),
                                   np.array([0, .1, np.nan, .3]), np.array([0, .1, .25, .3])])
def test_rejects_shifted_duplicate_nonfinite_or_stretched_timestamp_grid(inputs, times):
    rewrite_artifact(inputs, "timestamps", times)
    with pytest.raises(ValueError, match="10 Hz anchors"):
        prepare_training_manifest(**inputs)


@pytest.mark.parametrize("shape", [(4, 2047), (4, 2048)])
def test_rejects_wrong_feature_dimension_or_nonfinite_values(inputs, shape):
    value = np.ones(shape, dtype=np.float32)
    value[0, 0] = np.nan
    rewrite_artifact(inputs, "features", value)
    with pytest.raises(ValueError, match="2048|Nonfinite features"):
        prepare_training_manifest(**inputs)


def test_trims_only_explicit_out_of_label_range_tail_without_modifying_source(inputs):
    rewrite_artifact(inputs, "features", np.ones((5, 2048), dtype=np.float32))
    rewrite_artifact(inputs, "timestamps", np.arange(5, dtype=np.float64) / 10)
    rewrite_artifact(inputs, "support_start", np.array([np.nan, 0, .1, 0, .1]))
    rewrite_artifact(inputs, "support_end", np.array([np.nan, 0, .1, .2, .3]))
    path = prepare_training_manifest(**inputs)
    manifest = json.loads(path.read_text())
    record = inputs["feature_manifest"]["videos"][0]
    assert np.load(Path(record["artifact_root"]) / record["features_path"]).shape == (5, 2048)
    assert PreparedDataset(path, "train")[0].features.shape == (2048, 4)
    assert manifest["input_audit"]["training"]["trimmed_feature_tail_samples"] == 1


def test_refuses_large_timeline_difference_instead_of_silent_truncation(inputs):
    rewrite_artifact(inputs, "features", np.ones((8, 2048), dtype=np.float32))
    rewrite_artifact(inputs, "timestamps", np.arange(8, dtype=np.float64) / 10)
    with pytest.raises(ValueError, match="rounding allowance"):
        prepare_training_manifest(**inputs)


def test_metadata_file_feature_rows_and_refuse_overwrite(inputs):
    record = inputs["feature_manifest"]["videos"][0]
    metadata_path = Path(record["artifact_root"]) / "metadata.json"
    write_json(metadata_path, record)
    inputs["feature_manifest"] = {"videos": [{"metadata_path": str(metadata_path)}]}
    prepare_training_manifest(**inputs)
    with pytest.raises(FileExistsError, match="overwrite prepared manifest"):
        prepare_training_manifest(**inputs)


def test_missing_training_feature_cannot_silently_reduce_pool(inputs):
    inputs["feature_manifest"]["videos"] = []
    with pytest.raises(ValueError, match="Missing training sources/features"):
        prepare_training_manifest(**inputs)


def test_rejects_unresolved_metadata_filename_alias(inputs):
    path = inputs["annotation_root"] / "vocabulary/CMU/files_dict.json"
    metadata = json.loads(path.read_text())
    metadata["training"]["video_path"] = "/original/other-training.mp4"
    write_json(path, metadata)
    inputs["annotation_sha256"]["metadata"] = sha(path)
    with pytest.raises(ValueError, match="Metadata video identity conflict"):
        prepare_training_manifest(**inputs)


def test_rejects_support_that_reaches_beyond_feature_anchor(inputs):
    rewrite_artifact(inputs, "support_end", np.array([np.nan, 0, .3, .2]))
    with pytest.raises(ValueError, match="support must precede"):
        prepare_training_manifest(**inputs)


def test_does_not_load_reserved_gt_feature_artifacts(inputs):
    inputs["feature_manifest"]["videos"].append({
        "dataset": "CMU", "video_id": "test", "features_path": "/must-not-read/test.npy"})
    path = prepare_training_manifest(**inputs)
    manifest = json.loads(path.read_text())
    assert set(manifest["input_audit"]) == {"training"}
    assert set(manifest["videos"]) == {"training"}
