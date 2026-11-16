"""Score/output logging separation with no cloud calls or model inference."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("modal")
import modal_pipeline as pipeline
from mosp import evaluate


def has_score_payload(value):
    if isinstance(value, dict):
        return bool(set(value) & {"metrics", "precision", "recall", "f1", "frame_micro", "per_video"}) or any(
            has_score_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(has_score_payload(item) for item in value)
    return False


def test_output_artifacts_and_runs_exclude_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "WORK", tmp_path)
    monkeypatch.setattr(pipeline, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    config = {"group": "training-group", "annotation_revision": "a" * 40}
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"existing model; never loaded by this logging check")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = tmp_path / "runs/training-group/CMU"
    output = root / "evaluation"
    (output / "predictions").mkdir(parents=True)
    np.savez(output / "predictions/video.npz", state_probabilities=np.zeros((2, 2)),
             timestamps_seconds=np.array([0.0, 0.1]), ground_truth_raw_frame_indices=np.array([0, 3]))
    alignment = output / "alignment_audit.json"
    alignment.write_text(json.dumps({"annotation_revision": config["annotation_revision"], "videos": {}}))
    metrics = {"dataset": "CMU", "checkpoint_sha256": checkpoint_sha,
               "checkpoint_epoch": 100, "scope": "fixture held-out metrics",
               "annotation_revision": config["annotation_revision"],
               "frame_micro": {"precision": 0.4, "recall": 0.5, "f1": 0.44}}
    (output / "metrics.json").write_text(json.dumps(metrics))
    original = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    # Even a stray score file in the prediction directory must not be swept in.
    (output / "predictions/scores.json").write_text(json.dumps(metrics))
    runs = []

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.files = name, type, metadata, []

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append((path, name))

        def add_dir(self, *args, **kwargs):
            pytest.fail("Directory uploads could leak metrics into saved outputs")

    class Run:
        def __init__(self, stage, group):
            self.stage, self.group, self.summary, self.artifacts = stage, group, {}, []
            self.url = f"https://wandb.example.test/{stage}"

        def log_artifact(self, artifact, aliases):
            self.artifacts.append(artifact)
            return SimpleNamespace(wait=lambda: None)

        def finish(self, exit_code):
            assert exit_code == 0

    def start(config, stage, dataset, *, group=None):
        run = Run(stage, group or config["group"])
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    monkeypatch.setattr(pipeline, "_wandb_run", start)
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args, **kwargs: pytest.fail("No inference requested"))
    result = pipeline.evaluate_dataset.local("CMU", str(checkpoint), {}, {}, config)
    predictions, scores = runs
    assert (predictions.stage, predictions.group) == ("predictions", "training-group")
    assert (scores.stage, scores.group) == ("metrics", "training-group-metrics")
    assert not has_score_payload(predictions.summary)
    assert not has_score_payload(predictions.artifacts[0].metadata)
    assert predictions.artifacts[0].type == "predictions"
    assert [name for _, name in predictions.artifacts[0].files] == [
        "predictions/video.npz", "alignment_audit.json"]
    assert scores.summary["metrics"] == metrics
    assert scores.artifacts[0].type == "evaluation-metrics"
    assert scores.artifacts[0].name != predictions.artifacts[0].name
    assert [name for _, name in scores.artifacts[0].files] == ["metrics.json"]
    assert not has_score_payload(result)
    assert not has_score_payload(json.loads((root / "predictions-complete.json").read_text()))
    assert not has_score_payload(json.loads((root / "evaluation-complete.json").read_text()))
    assert result["checkpoint_sha256"] == checkpoint_sha
    assert result["annotation_revision"] == config["annotation_revision"]
    assert result["checkpoint_epoch"] == 100
    assert result["scope"] == metrics["scope"]
    assert all(path.read_bytes() == content for path, content in original.items())


def test_legacy_nested_metrics_do_not_escape_into_parent_results():
    legacy = {"dataset": "CMU", "wandb_url": "https://wandb.example.test/old-output-run",
              "metrics": {"checkpoint_sha256": "a" * 64, "checkpoint_epoch": 100,
                          "annotation_revision": "b" * 40, "scope": "original scope",
                          "frame_micro": {"precision": 0.2, "recall": 0.3, "f1": 0.24}}}
    results = {"CMU": {"training": {"dataset": "CMU", "checkpoint_path": "/work/last.pt",
                                      "wandb_url": "https://wandb.example.test/train",
                                      "metrics": {"precision": 0.9}}, "evaluation": legacy}}
    safe = pipeline._pipeline_references(results)
    assert not has_score_payload(safe)
    assert safe["CMU"]["evaluation"]["checkpoint_sha256"] == "a" * 64
    assert safe["CMU"]["evaluation"]["annotation_revision"] == "b" * 40
    assert safe["CMU"]["evaluation"]["checkpoint_epoch"] == 100
    assert safe["CMU"]["evaluation"]["scope"] == "original scope"
    assert safe["CMU"]["evaluation"]["predictions_run_url"] == legacy["wandb_url"]
    assert safe["CMU"]["training"]["checkpoint_path"] == "/work/last.pt"


def test_reference_fields_cannot_hide_nested_score_payloads():
    with pytest.raises(ValueError, match="strings"):
        pipeline._evaluation_references({"dataset": "CMU", "metrics_run_url": {"f1": 0.9}})


def test_metrics_group_override_preserves_exact_project_and_training_config(monkeypatch):
    calls = []
    config = {"group": "training-group", "epochs": 100}
    fake = SimpleNamespace(Settings=lambda **kwargs: kwargs,
                           init=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(url="test"))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    pipeline._wandb_run(config, "metrics", "CMU", group="training-group-metrics")
    assert calls[0]["project"] == "MOSP - full evals"
    assert calls[0]["entity"] == "chefOST"
    assert calls[0]["group"] == "training-group-metrics"
    assert calls[0]["job_type"] == "metrics"
    assert config == {"group": "training-group", "epochs": 100}
