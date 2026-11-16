import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosp import evaluate
from mosp.data import _vocabulary


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_micro_counts_are_pooled_not_average_of_frame_ratios():
    probabilities = np.array([[1., 0, 0], [0, 1, 1]])
    targets = np.array([[True, False, False], [False, False, False]])
    result = evaluate.frame_micro_scores(probabilities, targets)["fixed_threshold_0_5"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    assert result["f1"] == .5


def test_undefined_denominators_are_explicit_null():
    result = evaluate.frame_micro_scores(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool))
    assert result["f1_max"] is None
    assert result["thresholds_attaining_f1_max"] == []
    assert result["fixed_threshold_0_5"]["precision"] is None
    assert result["fixed_threshold_0_5"]["recall"] is None
    json.dumps(result, allow_nan=False)


def test_nearest_raw_24fps_indices_without_stretching():
    video = {"r_frame_rate": "24/1", "avg_frame_rate": "24/1", "nb_frames": "10", "duration": "0.416666667"}
    indices, audit = evaluate.align_raw_frames(np.array([0, .1, .2, .3]), video, 10)
    np.testing.assert_array_equal(indices, [0, 2, 5, 7])
    assert audit["max_nearest_frame_error_seconds"] <= .5 / 24
    assert audit["interpolation_or_rescaling"] is False


@pytest.mark.parametrize("change,match", [
    ({"nb_frames": "11"}, "raw-frame count"),
    ({"avg_frame_rate": "25/1"}, "constant-frame-rate"),
    ({"duration": "3.0"}, "duration disagrees"),
])
def test_rejects_unverified_raw_frame_alignment(change, match):
    video = {"r_frame_rate": "24/1", "avg_frame_rate": "24/1", "nb_frames": "10", "duration": "0.416666667", **change}
    with pytest.raises(ValueError, match=match):
        evaluate.align_raw_frames(np.array([0, .1, .2, .3]), video, 10)


@pytest.fixture
def evaluation_inputs(tmp_path, monkeypatch):
    root = tmp_path / "MOSCATO"
    vocab = root / "vocabulary/CMU"
    paths = {
        "objects": vocab / "object_dict.json", "states": vocab / "state_dict.json",
        "actions": vocab / "action_dict.json", "metadata": vocab / "files_dict.json",
        "pseudo_labels": root / "annotations/pseudo_labels/CMU/pseudo_annotations_poc_new_cmu.json",
        "ground_truth": root / "annotations/ground_truth/CMU/gt_annotations_cmu.json",
        "indexed_ground_truth": root / "annotations/ground_truth/CMU/gt_annotations_cmu_idx.json",
    }
    write_json(paths["objects"], {"o2i": {"cup": 0}})
    write_json(paths["states"], {"s2i": {"empty": 0, "full": 1}})
    write_json(paths["actions"], {"a2i": {"pour": 0}})
    write_json(paths["metadata"], {"training": {}, "test": {}})
    write_json(paths["pseudo_labels"], {"training": {}, "test": {}})
    write_json(paths["ground_truth"], {"test": {"state": {"cup": [["empty"]] * 3 + [["full"]] * 6}, "object": [["cup"]] * 9}})
    write_json(paths["indexed_ground_truth"], {"test": {"state": {"0": [[0]] * 3 + [[1]] * 6}, "object": [[0]] * 9}})
    provenance = {"model": "fixture", "dimension": 2, "layout": "TD", "timeline": "10 Hz output anchors"}
    metadata = {"dataset": "CMU", "training_video_ids": ["training"], "reserved_splits": {"validation": [], "test": ["test"]},
        "feature_provenance": provenance,
        "split_provenance": {"annotation_revision": "a" * 40, "annotation_sha256": {name: digest(path) for name, path in paths.items()}},
        "vocabulary_snapshot": {}}
    for name, prefix in (("objects", "o"), ("states", "s"), ("actions", "a")):
        vocabulary = _vocabulary(paths[name], prefix)
        metadata[name] = list(vocabulary.names)
        metadata["vocabulary_snapshot"][name] = vocabulary.snapshot

    class FeatureOnlyModel(torch.nn.Module):
        config = SimpleNamespace(feature_dim=2, num_state_classes=2)

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, features):
            # The input is exactly the extracted feature array, not GT labels.
            self.calls.append(features.detach().cpu().numpy())
            return {"probabilities": features}

    model = FeatureOnlyModel()
    saved = {"metadata": metadata, "epoch": 4}
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda path, device: (model, saved))
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"unit checkpoint")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    matrix = np.array([[.9, .1], [.1, .8], [.7, .2]], dtype=np.float32)
    np.save(feature_dir / "features.npy", matrix)
    np.save(feature_dir / "timestamps.npy", np.array([0, .1, .2]))
    source = {"bucket": "fixture", "key": "test/test.mp4", "sha256": "b" * 64, "size_bytes": 1000}
    record = {"dataset": "CMU", "video_id": "test", "source": source,
        "annotation_revision": "a" * 40, "feature_provenance": provenance, "artifact_root": str(feature_dir),
        "features_path": "features.npy", "timestamps_path": "timestamps.npy",
        "artifact_sha256": {name: digest(feature_dir / f"{name}.npy") for name in ("features", "timestamps")}}
    args = {"checkpoint": checkpoint, "annotation_root": root, "dataset": "CMU", "output_dir": tmp_path / "evaluation",
        "feature_manifest": {"videos": [record]}, "source_inventory": {"videos": [{"dataset": "CMU", "video_id": "test", "source": dict(source),
            "video": {"r_frame_rate": "30/1", "avg_frame_rate": "30/1", "duration": ".3", "nb_frames": "9"}}]}}
    return args, saved, model, paths, matrix


def test_checkpoint_only_feature_predictions_and_heldout_scores(evaluation_inputs):
    args, saved, model, paths, matrix = evaluation_inputs
    result = evaluate.evaluate_checkpoint(**args)
    assert len(model.calls) == 1
    np.testing.assert_array_equal(model.calls[0], matrix.T[None])
    score = result["frame_micro"]["fixed_threshold_0_5"]
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 1)
    assert score["f1"] == pytest.approx(2 / 3)
    assert result["frame_micro"]["f1_max"] == pytest.approx(6 / 7)
    assert result["frame_micro"]["thresholds_attaining_f1_max"] == [.2]
    assert result["training_ground_truth_overlap"] == []
    with np.load(args["output_dir"] / "predictions/test.npz") as prediction:
        assert set(prediction.files) == {"state_probabilities", "timestamps_seconds", "ground_truth_raw_frame_indices"}
        np.testing.assert_array_equal(prediction["ground_truth_raw_frame_indices"], [0, 3, 6])
    assert (args["output_dir"] / "metrics.json").is_file()
    assert (args["output_dir"] / "alignment_audit.json").is_file()


def test_ground_truth_leakage_in_checkpoint_is_rejected_before_prediction(evaluation_inputs):
    args, saved, model, _, _ = evaluation_inputs
    saved["metadata"]["training_video_ids"] = ["test"]
    with pytest.raises(ValueError, match="training IDs overlap"):
        evaluate.evaluate_checkpoint(**args)
    assert model.calls == []


def test_source_identity_mismatch_blocks_scoring(evaluation_inputs):
    args, _, model, _, _ = evaluation_inputs
    args["feature_manifest"]["videos"][0]["source"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source differs"):
        evaluate.evaluate_checkpoint(**args)
    assert model.calls == []


def test_model_feature_provenance_must_match(evaluation_inputs):
    args, _, model, _, _ = evaluation_inputs
    args["feature_manifest"]["videos"][0]["feature_provenance"] = {"model": "wrong extractor"}
    with pytest.raises(ValueError, match="extractor provenance"):
        evaluate.evaluate_checkpoint(**args)
    assert model.calls == []


def test_named_indexed_gt_disagreement_blocks_scoring(evaluation_inputs):
    args, saved, model, paths, _ = evaluation_inputs
    path = paths["indexed_ground_truth"]
    write_json(path, {"test": {"state": {"0": [[0]] * 9}, "object": [[0]] * 9}})
    saved["metadata"]["split_provenance"]["annotation_sha256"]["indexed_ground_truth"] = digest(path)
    with pytest.raises(ValueError, match="targets disagree"):
        evaluate.evaluate_checkpoint(**args)
    assert model.calls == []
