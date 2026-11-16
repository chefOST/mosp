"""Output-only cloud publication checks; no cloud calls or model inference."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")
import modal_predictions as job


def test_score_payload_guard_allows_model_precision_but_rejects_scores():
    job._assert_no_scores({"feature_provenance": {"mixed_precision": True}, "per_video": {"id": {"samples": 12}}})
    for key in ("metrics", "frame_micro", "precision", "recall", "f1"):
        with pytest.raises(ValueError, match="score"):
            job._assert_no_scores({"per_video": {"id": {key: 0.5}}})


def test_output_membership_is_explicit():
    files = job._output_files({"per_video": {"video": {"prediction_path": "/out/video.npz",
                                                           "receipt_path": "/out/video.json",
                                                           "unrelated": "/out/metrics.json"}}})
    assert files == [("/out/video.npz", "predictions/video.npz"), ("/out/video.json", "receipts/video.json")]


def test_cloud_job_only_publishes_predictions_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "WORK", tmp_path)
    monkeypatch.setattr(job, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    group = "fixture"
    root = tmp_path / "runs" / group
    checkpoint = root / "EGTEA/training/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model fixture, not deserialized")
    (root / "EGTEA/training-complete.json").write_text(json.dumps({
        "checkpoint_path": str(checkpoint), "run_metadata": {"status": "completed", "last_completed_epoch": 100,
                                                            "wandb": {"id": "training-run"}}}))
    artifacts, calls = [], []
    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.files = name, type, metadata, []
        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append(name)
        def add_dir(self, *args, **kwargs):
            pytest.fail("No directory uploads")
    class Run:
        url = "https://wandb.example/predictions"
        def __init__(self): self.summary = {}
        def log(self, values):
            assert set(values) == {"predictions/videos_complete", "predictions/videos_total"}
        def log_artifact(self, artifact, aliases):
            artifacts.append(artifact)
            return SimpleNamespace(wait=lambda: SimpleNamespace(source_name=artifact.name + ":v0"))
        def finish(self, exit_code): assert exit_code == 0
    run = Run()
    def init(**kwargs):
        calls.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Artifact=Artifact, Settings=lambda **kwargs: kwargs))
    def predict(checkpoint_path, features, source, output, *, device, progress_callback):
        assert checkpoint_path == checkpoint and device == "cuda"
        assert features.name == "feature-index.json" and source.name == "source-inventory.json"
        assert not any("annotation" in str(x) for x in (features, source))
        output.mkdir(parents=True)
        index = {"dataset": "EGTEA", "per_video": {}}
        for i in range(12):
            video_id = f"video{i:02}"
            npz, receipt = output / f"{video_id}.npz", output / f"{video_id}.json"
            npz.write_bytes(b"saved prediction fixture")
            receipt.write_text(json.dumps({"video_id": video_id, "source": {"key": video_id}}))
            row = {"prediction_path": str(npz), "receipt_path": str(receipt), "video_id": video_id}
            index["per_video"][video_id] = row
            progress_callback(row, i + 1, 12)
        (output / "index.json").write_text(json.dumps(index))
        (output / "metrics.json").write_text('{"f1": 0.9}')
        return index
    monkeypatch.setitem(sys.modules, "mosp.batch_predict", SimpleNamespace(batch_predict=predict))
    result = job.predict_egtea.local(group, code_revision="test")
    assert calls[0]["project"] == "MOSP - full evals" and calls[0]["job_type"] == "predictions"
    assert len(artifacts) == 1 and artifacts[0].type == "predictions"
    assert len(artifacts[0].files) == 25
    assert "metrics.json" not in artifacts[0].files
    job._assert_no_scores(run.summary)
    assert result["prediction_videos"] == 12 and result["status"] == "complete"
