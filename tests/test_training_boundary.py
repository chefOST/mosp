"""Unit checks for the training/prediction boundary; no benchmark scoring."""

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mosp import predict, train
from mosp.runtime import load_checkpoint


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    sample = SimpleNamespace(
        video_id="training-input",
        features=np.ones((2, 3), dtype=np.float32),
        states=np.array([[1, 1, 0], [0, 0, 1]], dtype=np.float32),
        actions=np.array([0, 0, 1], dtype=np.int64),
    )

    class TrainingInput:
        video_ids = [sample.video_id]
        manifest = {"splits": {"train": video_ids, "validation": ["reserved-without-files"], "test": []}}
        metadata = {
            "objects": ["cup"], "states": ["empty", "full"], "actions": ["take", "pour"],
            "feature_dimension": 2,
            "feature_provenance": {"model": "unit fixture", "dimension": 2, "layout": "DT", "timeline": "three samples"},
        }

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return sample

    requested_splits = []

    def load_training_only(path, split):
        requested_splits.append(split)
        assert split == "train", "Trainer must never request held-out samples"
        return TrainingInput()

    monkeypatch.setattr(train, "PreparedDataset", load_training_only)
    monkeypatch.setattr(train, "compute_state_action_correlations", lambda _: (np.eye(2), np.eye(2)))
    args = train.parser().parse_args([
        "--manifest", str(manifest), "--out", str(tmp_path / "training"),
        "--epochs", "2", "--beta", "0.3", "--layers", "1", "--refinements", "0",
        "--channels", "2", "--dropout", "0",
    ])
    run = train.train(args)
    assert requested_splits == ["train"]
    assert run["last_completed_epoch"] == 2
    assert "checkpoint_selection" not in run
    assert all("validation_state_bce" not in row for row in run["history"])
    assert {path.name for path in args.out.glob("*.pt")} == {"last.pt"}
    return args.out / "last.pt"


def test_latest_epoch_checkpoint_without_validation_or_best_selection(checkpoint):
    model, saved = load_checkpoint(checkpoint)
    assert saved["epoch"] == 2
    assert "selection" not in saved
    assert saved["metadata"]["training_video_ids"] == ["training-input"]
    assert saved["metadata"]["reserved_splits"]["validation"] == ["reserved-without-files"]
    assert not model.training  # Inference mode, not a score calculation.


def test_prediction_exports_probabilities_without_annotations_or_scores(checkpoint, tmp_path):
    _, saved = load_checkpoint(checkpoint)
    feature_path = tmp_path / "input.npy"
    np.save(feature_path, np.ones((2, 3), dtype=np.float32))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(saved["metadata"]["feature_provenance"]))
    output = tmp_path / "predictions.npz"
    args = predict.parser().parse_args([
        "--checkpoint", str(checkpoint), "--features", str(feature_path), "--layout", "DT",
        "--feature-provenance", str(provenance), "--out", str(output),
    ])
    predict.predict(args)
    with np.load(output) as arrays:
        assert set(arrays.files) == {"state_probabilities", "action_probabilities"}
        assert arrays["state_probabilities"].shape == (3, 2)
        assert arrays["action_probabilities"].shape == (3, 2)
        assert np.isfinite(arrays["state_probabilities"]).all()


def test_scoring_modules_are_not_part_of_package():
    assert importlib.util.find_spec("mosp.evaluate") is None
    assert importlib.util.find_spec("mosp.metrics") is None
