import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosp import batch_predict as batch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.fixture"
    checkpoint.write_bytes(b"synthetic checkpoint identity, no model weights")
    provenance = {"model": "fixture", "dimension": 2, "layout": "TD", "timeline": "10 Hz output anchors"}
    saved = {"epoch": 100, "model_config": {"feature_dim": 2, "num_state_classes": 2, "num_actions": 2},
             "metadata": {"dataset": "EGTEA", "objects": ["cup"], "states": ["empty", "full"],
                          "actions": ["pour", "cut"], "feature_provenance": provenance,
                          "reserved_splits": {"test": ["video-b", "video-a"]}}}

    class FeatureOnlyModel(torch.nn.Module):
        config = SimpleNamespace(feature_dim=2, num_state_classes=2, num_actions=2)

        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, features):
            assert not self.training
            assert not torch.is_grad_enabled()
            self.inputs.append(features.detach().cpu().numpy().copy())
            return {"probabilities": features.sigmoid(), "action_logits": torch.stack([features, features + 1])}

    model = FeatureOnlyModel()
    loads = []

    def load_once(path, device):
        loads.append((path, device))
        return model, saved

    monkeypatch.setattr(batch, "load_checkpoint", load_once)
    source_records, feature_records, arrays = [], [], {}
    for number, video_id in enumerate(["video-a", "video-b"]):
        directory = tmp_path / "features" / video_id
        directory.mkdir(parents=True)
        matrix = np.arange((3 + number) * 2, dtype=np.float32).reshape(3 + number, 2) / 10
        arrays[video_id] = matrix
        np.save(directory / "features.npy", matrix)
        np.save(directory / "timestamps.npy", np.arange(len(matrix), dtype=np.float64) / 10)
        source = {"bucket": "fixture", "key": f"videos/{video_id}.mp4", "sha256": str(number + 1) * 64,
                  "size_bytes": 1000 + number}
        source_records.append({"dataset": "EGTEA", "video_id": video_id, "source": source})
        feature_records.append({"schema_version": 1, "status": "complete", "dataset": "EGTEA", "video_id": video_id,
            "source": dict(source), "feature_provenance": dict(provenance), "frame_count": len(matrix),
            "artifact_root": str(directory), "features_path": "features.npy", "timestamps_path": "timestamps.npy",
            "artifact_sha256": {name: digest(directory / f"{name}.npy") for name in ("features", "timestamps")}})
    # Additional rows have no usable artifacts and must not be read by default.
    feature_records.append({"dataset": "EGTEA", "video_id": "unrequested", "features_path": "/must-not-read.npy"})
    return {"checkpoint": checkpoint, "feature_manifest": {"videos": feature_records},
            "source_inventory": {"videos": source_records}, "output_dir": tmp_path / "outputs"}, model, loads, saved, arrays


def assert_no_result_scores(value):
    if isinstance(value, dict):
        assert not {"precision", "recall", "f1", "f1_max", "frame_micro"} & set(value)
        for child in value.values():
            assert_no_result_scores(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_result_scores(child)


def test_loads_once_preserves_every_timepoint_and_emits_only_probabilities(inputs):
    args, model, loads, saved, arrays = inputs
    notifications = []

    def progress(record, completed, total):
        assert Path(record["prediction_path"]).is_file()
        assert Path(record["receipt_path"]).is_file()
        index = json.loads((args["output_dir"] / "index.json").read_text())
        assert record["video_id"] in index["per_video"]
        notifications.append((record["video_id"], completed, total))

    result = batch.batch_predict(**args, progress_callback=progress)
    assert len(loads) == 1
    assert len(model.inputs) == 2
    assert result["status"] == "complete"
    assert result["dataset"] == "EGTEA"
    assert result["requested_video_ids"] == ["video-a", "video-b"]
    assert notifications == [("video-a", 1, 2), ("video-b", 2, 2)]
    assert_no_result_scores(result)
    for video_id, entry in result["per_video"].items():
        with np.load(entry["prediction_path"], allow_pickle=False) as prediction:
            assert set(prediction.files) == {"state_probabilities", "action_probabilities", "timestamps_seconds"}
            np.testing.assert_array_equal(prediction["timestamps_seconds"], np.arange(len(arrays[video_id])) / 10)
            np.testing.assert_allclose(prediction["state_probabilities"], 1 / (1 + np.exp(-arrays[video_id])), rtol=1e-6)
            np.testing.assert_allclose(prediction["action_probabilities"].sum(axis=1), 1)
            assert prediction["state_probabilities"].shape == arrays[video_id].shape
        receipt = json.loads(Path(entry["receipt_path"]).read_text())
        assert receipt["checkpoint_sha256"] == digest(args["checkpoint"])
        assert receipt["prediction_sha256"] == digest(Path(entry["prediction_path"]))
        assert receipt["vocabularies"]["states"] == ["empty", "full"]
        assert receipt["probabilities_thresholded"] is False
        assert_no_result_scores(receipt)


def test_completed_batch_resume_does_not_run_model_or_change_prediction_bytes(inputs):
    args, model, loads, _, _ = inputs
    first = batch.batch_predict(**args)
    original = {k: (Path(v["prediction_path"]).stat().st_mtime_ns, v["prediction_sha256"])
                for k, v in first["per_video"].items()}
    model.inputs.clear()
    second = batch.batch_predict(**args)
    assert len(loads) == 2  # Once in each invocation, never once per video.
    assert model.inputs == []
    assert {k: (Path(v["prediction_path"]).stat().st_mtime_ns, v["prediction_sha256"])
            for k, v in second["per_video"].items()} == original


def test_callback_failure_resumes_after_committed_video(inputs):
    args, model, _, _, _ = inputs

    def interrupted(record, completed, total):
        raise RuntimeError("volume callback interrupted")

    with pytest.raises(RuntimeError, match="callback interrupted"):
        batch.batch_predict(**args, progress_callback=interrupted)
    first_path = args["output_dir"] / "predictions/video-a/predictions.npz"
    original = first_path.read_bytes()
    assert not (args["output_dir"] / "predictions/video-b").exists()
    model.inputs.clear()
    result = batch.batch_predict(**args)
    assert len(model.inputs) == 1
    assert first_path.read_bytes() == original
    assert result["status"] == "complete"


def test_recovers_directory_committed_before_index_update(inputs, monkeypatch):
    args, model, _, _, _ = inputs
    original_write = batch.write_json
    calls = []

    def interrupted(path, value):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("index write interrupted")
        return original_write(path, value)

    monkeypatch.setattr(batch, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="index write interrupted"):
        batch.batch_predict(**args)
    assert (args["output_dir"] / "predictions/video-a/metadata.json").is_file()
    monkeypatch.setattr(batch, "write_json", original_write)
    model.inputs.clear()
    result = batch.batch_predict(**args)
    assert len(model.inputs) == 1
    assert result["completed_video_ids"] == ["video-a", "video-b"]


def test_incomplete_staging_is_never_published(inputs, monkeypatch):
    args, model, _, _, _ = inputs
    original_save = batch.np.savez_compressed

    def interrupted(handle, **arrays):
        handle.write(b"partial zip")
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(batch.np, "savez_compressed", interrupted)
    with pytest.raises(RuntimeError, match="write interrupted"):
        batch.batch_predict(**args)
    assert list((args["output_dir"] / "predictions").iterdir()) == []
    monkeypatch.setattr(batch.np, "savez_compressed", original_save)
    assert batch.batch_predict(**args)["status"] == "complete"


def test_different_checkpoint_cannot_overwrite_existing_predictions(inputs):
    args, model, _, _, _ = inputs
    result = batch.batch_predict(**args)
    original = Path(result["per_video"]["video-a"]["prediction_path"]).read_bytes()
    args["checkpoint"].write_bytes(b"different synthetic checkpoint")
    model.inputs.clear()
    with pytest.raises(ValueError, match="different checkpoint"):
        batch.batch_predict(**args)
    assert model.inputs == []
    assert Path(result["per_video"]["video-a"]["prediction_path"]).read_bytes() == original


def test_different_features_cannot_overwrite_completed_video(inputs):
    args, _, _, _, _ = inputs
    result = batch.batch_predict(**args)
    record = args["feature_manifest"]["videos"][0]
    path = Path(record["artifact_root"]) / record["features_path"]
    np.save(path, np.full((3, 2), .25, dtype=np.float32))
    record["artifact_sha256"]["features"] = digest(path)
    with pytest.raises(ValueError, match="different inputs/provenance"):
        batch.batch_predict(**args)
    assert digest(Path(result["per_video"]["video-a"]["prediction_path"])) == result["per_video"]["video-a"]["prediction_sha256"]


def test_corrupt_completed_prediction_is_not_silently_regenerated(inputs):
    args, model, _, _, _ = inputs
    result = batch.batch_predict(**args)
    Path(result["per_video"]["video-a"]["prediction_path"]).write_bytes(b"damaged")
    model.inputs.clear()
    with pytest.raises(ValueError, match="checksum mismatch"):
        batch.batch_predict(**args)
    assert model.inputs == []


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("key", "videos/different.mp4"), ("size_bytes", 1)])
def test_source_identity_mismatch_blocks_all_inference(inputs, field, value):
    args, model, _, _, _ = inputs
    args["feature_manifest"]["videos"][1]["source"][field] = value
    with pytest.raises(ValueError, match="source differs|exact video"):
        batch.batch_predict(**args)
    assert model.inputs == []


def test_provenance_mismatch_blocks_all_inference(inputs):
    args, model, _, _, _ = inputs
    args["feature_manifest"]["videos"][0]["feature_provenance"]["model"] = "different extractor"
    with pytest.raises(ValueError, match="extractor provenance"):
        batch.batch_predict(**args)
    assert model.inputs == []


@pytest.mark.parametrize("invalid", ["nonfinite", "dimension", "timestamps"])
def test_invalid_feature_artifacts_are_rejected_before_model(inputs, invalid):
    args, model, _, _, _ = inputs
    record = args["feature_manifest"]["videos"][0]
    name = "timestamps" if invalid == "timestamps" else "features"
    path = Path(record["artifact_root"]) / record[f"{name}_path"]
    if invalid == "timestamps":
        value = np.array([0, .1, .25])
    else:
        value = np.ones((3, 3 if invalid == "dimension" else 2), dtype=np.float32)
        if invalid == "nonfinite":
            value[0, 0] = np.nan
    np.save(path, value)
    record["artifact_sha256"][name] = digest(path)
    with pytest.raises(ValueError, match="Nonfinite|dimensions|10 Hz"):
        batch.batch_predict(**args)
    assert model.inputs == []


def test_different_request_membership_is_rejected_before_forward(inputs):
    args, model, _, _, _ = inputs
    first = batch.batch_predict(**args, video_ids=["video-a"])
    original = first["per_video"]["video-a"]["prediction_sha256"]
    model.inputs.clear()
    with pytest.raises(ValueError, match="different requested video IDs"):
        batch.batch_predict(**args, video_ids=["video-b"])
    assert model.inputs == []
    assert digest(Path(first["per_video"]["video-a"]["prediction_path"])) == original


def test_module_has_no_target_or_scoring_api():
    assert "annotation_root" not in inspect.signature(batch.batch_predict).parameters
    tree = ast.parse(inspect.getsource(batch))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.rsplit(".", 1)[-1] in {"evaluate", "metrics"} for name in imports)
    functions = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "evaluate_checkpoint" not in functions
    assert "frame_micro_scores" not in functions
