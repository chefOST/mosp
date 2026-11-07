import json

import numpy as np
import pytest

from mosp import evaluate, predict, train
from mosp.runtime import load_checkpoint
from mosp.smoke import make_fixture


@pytest.fixture
def trained(tmp_path):
    manifest = make_fixture(tmp_path / "fixture")
    output = tmp_path / "training"
    args = train.parser().parse_args([
        "--manifest", str(manifest), "--out", str(output), "--epochs", "1", "--beta", "0.3",
        "--layers", "2", "--refinements", "1", "--channels", "4", "--dropout", "0"])
    run = train.train(args)
    assert run["status"] == "completed"
    return manifest, output


def test_checkpoint_roundtrip_and_annotation_free_prediction(trained, tmp_path):
    manifest, output = trained
    checkpoint = output / "selected.pt"
    model, saved = load_checkpoint(checkpoint)
    assert saved["model_config"]["beta"] == 0.3
    assert saved["metadata"]["training_video_ids"] == ["train_a", "train_b"]
    assert saved["metadata"]["validation_video_ids"] == ["validation"]
    destination = tmp_path / "prediction.npz"
    # Prediction remains possible with annotation files physically unavailable.
    (manifest.parent / "pseudo.json").unlink()
    (manifest.parent / "ground_truth.json").unlink()
    predict.predict(predict.parser().parse_args([
        "--checkpoint", str(checkpoint), "--features", str(manifest.parent / "test.npy"),
        "--layout", "DT", "--feature-provenance", str(manifest.parent / "feature_provenance.json"),
        "--out", str(destination)]))
    with np.load(destination) as result:
        assert result["state_probabilities"].shape == (16, 2)
        assert np.isfinite(result["state_probabilities"]).all()


def test_evaluation_and_changed_vocabulary_rejection(trained, tmp_path):
    manifest, output = trained
    args = evaluate.parser().parse_args([
        "--manifest", str(manifest), "--checkpoint", str(output / "selected.pt"),
        "--evaluation-timeline", "annotation", "--segment-threshold", "0.5", "--gap-frames", "0",
        "--map-empty-classes", "exclude"])
    assert evaluate.evaluate(args)["protocol"]["num_videos"] == 1
    (manifest.parent / "states.json").write_text(json.dumps({"s2i": {"after": 0, "before": 1}}))
    with pytest.raises(ValueError, match="Checkpoint/data mismatch"):
        evaluate.evaluate(args)


def test_training_does_not_overwrite_completed_run(trained):
    manifest, output = trained
    args = train.parser().parse_args([
        "--manifest", str(manifest), "--out", str(output), "--epochs", "1", "--beta", "0.3"])
    with pytest.raises(FileExistsError):
        train.train(args)


def test_prediction_rejects_wrong_extractor_even_with_matching_dimension(trained, tmp_path):
    manifest, output = trained
    source = manifest.parent / "feature_provenance.json"
    provenance = json.loads(source.read_text())
    provenance["model"] = "an incompatible backbone with four output features"
    source.write_text(json.dumps(provenance))
    args = predict.parser().parse_args([
        "--checkpoint", str(output / "selected.pt"), "--features", str(manifest.parent / "test.npy"),
        "--layout", "DT", "--feature-provenance", str(source), "--out", str(tmp_path / "bad.npz")])
    with pytest.raises(ValueError, match="feature provenance"):
        predict.predict(args)


def test_full_frame_evaluation_requires_explicit_prediction_mapping(trained):
    manifest, output = trained
    data = json.loads(manifest.read_text())
    del data["videos"]["test"]["evaluation_indices"]
    manifest.write_text(json.dumps(data))
    args = evaluate.parser().parse_args([
        "--manifest", str(manifest), "--checkpoint", str(output / "selected.pt"),
        "--evaluation-timeline", "annotation", "--segment-threshold", "0.5", "--gap-frames", "0",
        "--map-empty-classes", "exclude"])
    with pytest.raises(ValueError, match="evaluation_indices"):
        evaluate.evaluate(args)


@pytest.mark.parametrize("protected_id", ["train_a", "validation"])
def test_new_manifest_cannot_relabel_checkpoint_training_or_selection_videos_as_test(trained, protected_id):
    manifest, output = trained
    data = json.loads(manifest.read_text())
    pseudo = json.loads((manifest.parent / "pseudo.json").read_text())
    ground_truth = json.loads((manifest.parent / "ground_truth.json").read_text())
    ground_truth[protected_id] = {"state": pseudo[protected_id]["state"]}
    (manifest.parent / "ground_truth.json").write_text(json.dumps(ground_truth))
    data["splits"] = {"train": ["train_b"], "validation": [], "test": [protected_id]}
    data["videos"] = {key: data["videos"][key] for key in ("train_b", protected_id)}
    data["videos"][protected_id].update({"state_source": "ground_truth", "action_source": "none",
                                         "evaluation_indices": list(range(16))})
    manifest.write_text(json.dumps(data))
    args = evaluate.parser().parse_args([
        "--manifest", str(manifest), "--checkpoint", str(output / "selected.pt"),
        "--evaluation-timeline", "annotation", "--segment-threshold", "0.5", "--gap-frames", "0",
        "--map-empty-classes", "exclude"])
    with pytest.raises(ValueError, match="checkpoint training/selection data"):
        evaluate.evaluate(args)
