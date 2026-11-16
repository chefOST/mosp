"""Exercise orchestration locally with every cloud/job operation replaced."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")
import modal_pipeline as pipeline
from mosp import cloud_data, data, pipeline_state


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "WORK", tmp_path)
    revision = "a" * 40
    config = {"group": "training-only-test", "annotation_revision": revision,
              "annotation_sha256": {}, "epochs": 100}
    train_ids = [f"train-{i:03d}" for i in range(134)]
    test_ids = ["held-out"]
    annotations = tmp_path / "annotations" / revision
    gt = annotations / "annotations/ground_truth/CMU/gt_annotations_cmu.json"
    gt.parent.mkdir(parents=True)
    gt.write_text(json.dumps({v: {} for v in test_ids}))
    metadata = annotations / "vocabulary/CMU/files_dict.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({v: {} for v in train_ids + test_ids}))
    inventory = {"videos": [{"dataset": "CMU", "video_id": v} for v in train_ids + test_ids]}
    assets = {"recipe_id": "recipe", "checkpoint_sha256": "b" * 64}
    root = tmp_path / "runs" / config["group"] / "CMU"
    calls = {"maps": [], "spawns": [], "evaluations": [], "cached": []}
    result = {"dataset": "CMU", "checkpoint_path": "preserved-model.pt", "wandb_url": "test"}
    remote_call = SimpleNamespace(object_id="fc-new123", get=lambda: result)

    class Run:
        def __init__(self):
            self.summary = {}

        def log(self, values):
            pass

        def log_artifact(self, artifact, aliases):
            return SimpleNamespace(wait=lambda: None)

        def finish(self, exit_code):
            pass

    class Artifact:
        def __init__(self, *args, **kwargs):
            pass

        def add_file(self, path, name):
            assert pipeline.Path(path).is_file()

    def extract(rows, **kwargs):
        calls["maps"].append([row["video_id"] for row in rows])
        return iter(rows)

    def spawn(*args):
        calls["spawns"].append(args)
        return remote_call

    def evaluate(*args):
        calls["evaluations"].append(args)
        return {"dataset": "CMU", "metrics": {"test": True}}

    def prepare(annotation_root, dataset, annotation_revision, source, features, output, **kwargs):
        assert {r["video_id"] for r in features["videos"]} == set(train_ids)
        output.mkdir(parents=True)
        manifest = output / "manifest.json"
        manifest.write_text(json.dumps({"splits": {"train": train_ids, "validation": [], "test": test_ids}}))
        return manifest

    def cached(index, manifest, evidence, **kwargs):
        calls["cached"].append((index.name, set(kwargs["source_records"]), evidence.name))
        if not index.exists():
            return None
        document = json.loads(index.read_text())
        # The real completed_feature_index enforces this exact membership rule;
        # its checksum/provenance failure cases are covered in test_pipeline_state.
        if {r["video_id"] for r in document["videos"]} != set(kwargs["source_records"]):
            raise ValueError("Completed feature index has unexpected IDs")
        assert evidence.is_file()
        return document

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    monkeypatch.setattr(pipeline, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(pipeline, "_wandb_run", lambda *args: Run())
    monkeypatch.setattr(pipeline, "extract_one", SimpleNamespace(map=extract))
    monkeypatch.setattr(pipeline, "train_dataset", SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(pipeline, "evaluate_dataset", SimpleNamespace(remote=evaluate))
    monkeypatch.setattr(pipeline.modal.FunctionCall, "from_id", lambda _: remote_call)
    monkeypatch.setattr(cloud_data, "prepare_training_manifest", prepare)
    monkeypatch.setattr(data, "validate_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_state, "completed_feature_index", cached)
    return SimpleNamespace(config=config, inventory=inventory, assets=assets, root=root, calls=calls,
                           train_ids=train_ids, test_ids=test_ids, result=result)


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_legacy_configs_stop_after_training(stage, explicit):
    if explicit:
        stage.config["evaluate_after_training"] = False
    result = pipeline.dataset_pipeline.local("CMU", stage.inventory, stage.assets, stage.config)
    assert result == {"training": stage.result}
    assert stage.calls["maps"] == [stage.train_ids]
    assert len(stage.calls["spawns"]) == 1
    assert stage.calls["evaluations"] == []
    assert not (stage.root / "feature-index.json").exists()
    handoff = json.loads((stage.root / "training-call.json").read_text())
    manifest = stage.root / "prepared/manifest.json"
    assert handoff["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_explicit_opt_in_extracts_held_out_features_and_evaluates(stage):
    stage.config["evaluate_after_training"] = True
    result = pipeline.dataset_pipeline.local("CMU", stage.inventory, stage.assets, stage.config)
    assert stage.calls["maps"] == [stage.train_ids, stage.test_ids]
    assert len(stage.calls["evaluations"]) == 1
    assert "evaluation" in result


@pytest.mark.parametrize("queued", [False, True])
def test_training_only_recovery_uses_train_index_and_retains_full_index(stage, queued):
    stage.root.mkdir(parents=True)
    train_rows = [r for r in stage.inventory["videos"] if r["video_id"] in stage.train_ids]
    (stage.root / "train-feature-index.json").write_text(json.dumps({"videos": train_rows}))
    full = stage.root / "feature-index.json"
    full.write_text(json.dumps(stage.inventory))
    original_full = full.read_bytes()
    (stage.root / "prepared").mkdir()
    manifest = stage.root / "prepared/manifest.json"
    manifest.write_text("{}")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    handoff = {"call_id": "fc-existing123", "manifest_sha256": sha}
    (stage.root / "training-call.json").write_text(json.dumps(handoff))
    if not queued:
        (stage.root / "training").mkdir()
        (stage.root / "training/run.json").write_text(json.dumps({"manifest_sha256": sha}))
    result = pipeline.dataset_pipeline.local("CMU", stage.inventory, stage.assets, stage.config)
    assert result == {"training": stage.result}
    assert not stage.calls["maps"] and not stage.calls["spawns"] and not stage.calls["evaluations"]
    assert stage.calls["cached"] == [("train-feature-index.json", set(stage.train_ids),
                                      "training-call.json" if queued else "run.json")]
    assert full.read_bytes() == original_full


@pytest.mark.parametrize("bad_value", ["false", 1, None])
def test_non_boolean_evaluation_opt_in_is_rejected(bad_value):
    with pytest.raises(ValueError, match="explicit boolean"):
        pipeline._evaluation_requested({"evaluate_after_training": bad_value})


def test_new_run_config_requires_explicit_evaluation_opt_in(monkeypatch):
    monkeypatch.setattr(pipeline, "annotation_relative_paths", lambda _: [])
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    assert pipeline.build_run_config("default")["evaluate_after_training"] is False
    assert pipeline.build_run_config("opt-in", evaluate=True)["evaluate_after_training"] is True
