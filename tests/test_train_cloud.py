"""Recovery and logging checks using tiny CPU data, never benchmark labels."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosp import runtime, train
from mosp.model import MOSPConfig


@pytest.fixture
def manifest(tmp_path):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    write("objects.json", {"o2i": {"cup": 0}})
    write("states.json", {"s2i": {"empty": 0, "full": 1}})
    write("actions.json", {"a2i": {"fill": 0, "empty": 1}})
    ids = ["train-a", "train-b"]
    labels = {"state": {"cup": [["empty"], ["empty"], ["full"], ["full"], ["empty"], ["empty"]]},
              "action": [[0, 3, "fill", 0], [3, 6, "empty", 1]]}
    write("pseudo.json", {video_id: labels for video_id in ids})
    write("gt.json", {"test": {"not_a_training_target": True}})
    for index, video_id in enumerate(ids):
        np.save(tmp_path / f"{video_id}.npy", np.arange(12, dtype=np.float32).reshape(2, 6) / 10 + index)
    value = {
        "schema_version": 1, "dataset": "CMU",
        "split_provenance": {"description": "unit fixture", "paper_split_verified": False},
        "vocabularies": {"objects": "objects.json", "states": "states.json", "actions": "actions.json"},
        "annotations": {"pseudo_labels": "pseudo.json", "ground_truth": "gt.json"},
        "feature_provenance": {"model": "fixture", "dimension": 2, "layout": "DT", "timeline": "six samples"},
        "splits": {"train": ids, "validation": ["reserved-without-files"], "test": ["test"]},
        "videos": {video_id: {"feature_path": f"{video_id}.npy", "state_source": "pseudo_labels",
                                "state_indices": list(range(6)), "action_source": "pseudo_labels",
                                "action_indices": list(range(6)), "action_segment_end": "exclusive"}
                   for video_id in ids},
    }
    write("manifest.json", value)
    return tmp_path / "manifest.json"


def arguments(manifest, name="run", *extra):
    return train.parser().parse_args([
        "--manifest", str(manifest), "--out", str(manifest.parent / name), "--epochs", "3",
        "--beta", "0.3", "--layers", "1", "--refinements", "0", "--channels", "3",
        "--dropout", "0.25", "--seed", "42", *extra,
    ])


def interrupt_after_first_epoch(path):
    if runtime.read_checkpoint(path)["epoch"] == 1:
        raise RuntimeError("simulated worker interruption")


def test_resume_matches_uninterrupted_weights_losses_and_optimizer(manifest):
    whole_args = arguments(manifest, "whole")
    whole = train.train(whole_args)
    resumed_args = arguments(manifest, "resumed", "--checkpoint-every", "1")
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train.train(resumed_args, checkpoint_callback=interrupt_after_first_epoch)
    assert runtime.read_checkpoint(resumed_args.out / "last.pt")["epoch"] == 1
    resumed_args.resume = True
    resumed = train.train(resumed_args)
    assert whole["history"] == resumed["history"]
    assert resumed["resume_count"] == 1
    assert all(not any("validation" in key for key in row) for row in resumed["history"])
    expected = runtime.read_checkpoint(whole_args.out / "last.pt")
    actual = runtime.read_checkpoint(resumed_args.out / "last.pt")
    for key, tensor in expected["state_dict"].items():
        assert torch.equal(tensor, actual["state_dict"][key]), key
    assert expected["optimizer"]["param_groups"] == actual["optimizer"]["param_groups"]
    for parameter, values in expected["optimizer"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, actual["optimizer"]["state"][parameter][key])
    assert {p.name for p in resumed_args.out.glob("*.pt")} == {
        "last.pt", "epoch_000001.pt", "epoch_000002.pt", "epoch_000003.pt"}


@pytest.mark.parametrize("field,value", [("learning_rate", 0.01), ("beta", 0.7), ("epochs", 4)])
def test_resume_rejects_changed_training_configuration(manifest, field, value):
    args = arguments(manifest)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train.train(args, checkpoint_callback=interrupt_after_first_epoch)
    before = runtime.file_sha256(args.out / "last.pt")
    args.resume = True
    setattr(args, field, value)
    with pytest.raises(ValueError, match="Resume training contract changed"):
        train.train(args)
    assert runtime.file_sha256(args.out / "last.pt") == before


def test_resume_rejects_changed_feature_bytes_without_manifest_change(manifest):
    args = arguments(manifest)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train.train(args, checkpoint_callback=interrupt_after_first_epoch)
    np.save(manifest.parent / "train-a.npy", np.zeros((2, 6), dtype=np.float32))
    args.resume = True
    with pytest.raises(ValueError, match="training_data"):
        train.train(args)


def test_resume_rejects_optimizer_settings_tampered_outside_contract(manifest):
    args = arguments(manifest)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train.train(args, checkpoint_callback=interrupt_after_first_epoch)
    path = args.out / "last.pt"
    saved = runtime.read_checkpoint(path)
    saved["optimizer"]["param_groups"][0]["lr"] = 0.1
    torch.save(saved, path)
    args.resume = True
    with pytest.raises(ValueError, match="optimizer settings"):
        train.train(args)


def test_initial_checkpoint_survives_first_epoch_interruption(manifest):
    args = arguments(manifest)

    def stop(path):
        assert runtime.read_checkpoint(path)["epoch"] == 0
        assert (path.parent / "run.json").is_file()
        raise RuntimeError("stop before first epoch")

    with pytest.raises(RuntimeError, match="stop before first epoch"):
        train.train(args, checkpoint_callback=stop)
    args.resume = True
    assert train.train(args)["last_completed_epoch"] == 3


def test_checkpoint_write_failure_preserves_previous_checkpoint(manifest, monkeypatch):
    path = manifest.parent / "last.pt"
    path.write_bytes(b"previous valid checkpoint bytes")

    def fail(_, stream):
        stream.write(b"incomplete checkpoint")
        raise OSError("disk interrupted")

    monkeypatch.setattr(runtime.torch, "save", fail)
    model = SimpleNamespace(config=MOSPConfig(2, 2, 0.3), state_dict=lambda: {})
    optimizer = SimpleNamespace(state_dict=lambda: {})
    with pytest.raises(OSError, match="disk interrupted"):
        runtime.save_checkpoint(path, model, optimizer, epoch=1, metadata={}, manifest=manifest,
                                correlations=(torch.eye(2), torch.eye(2)))
    assert path.read_bytes() == b"previous valid checkpoint bytes"


def test_optional_wandb_exact_destination_resume_and_no_system_metrics(manifest, monkeypatch):
    initializations, logged_runs, artifacts = [], [], []

    class Artifact:
        def __init__(self, name, **kwargs):
            self.name, self.kwargs, self.files = name, kwargs, []

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append(name)

    class Run:
        url = "https://wandb.example.test/run"

        def __init__(self):
            self.summary, self.logs, self.exit_code = {}, [], None

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, step):
            self.logs.append((step, values))

        def log_artifact(self, artifact, aliases):
            artifacts.append((artifact, aliases))
            return SimpleNamespace(wait=lambda: None)

        def finish(self, exit_code):
            self.exit_code = exit_code

    def init(**kwargs):
        initializations.append(kwargs)
        result = Run()
        logged_runs.append(result)
        return result

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, Settings=lambda **kwargs: kwargs, Artifact=Artifact))
    args = arguments(manifest, "logged", "--wandb-project", "MOSP - full evals",
                     "--wandb-group", "reproduction-coffee", "--wandb-job-type", "train",
                     "--wandb-name", "coffee-seed42")
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train.train(args, checkpoint_callback=interrupt_after_first_epoch)
    args.resume = True
    train.train(args)
    assert initializations[0]["id"] == initializations[1]["id"]
    for call in initializations:
        assert call["project"] == "MOSP - full evals"
        assert call["entity"] == "chefOST"
        assert call["group"] == "reproduction-coffee"
        assert call["job_type"] == "train"
        assert call["settings"]["x_disable_stats"] is True
        assert call["settings"]["x_disable_meta"] is True
        assert call["settings"]["x_disable_machine_info"] is True
    assert logged_runs[0].exit_code == 1 and logged_runs[1].exit_code == 0
    assert [step for run in logged_runs for step, _ in run.logs] == [1, 2, 3]
    assert all(set(values) == {"epoch", "train/total", "train/mosp", "train/sai", "train/tas",
                               "train/tas_ce", "train/tas_smoothing"}
               for run in logged_runs for _, values in run.logs)
    assert artifacts[-1][1] == ["latest", "epoch-000003", "final"]
    assert artifacts[-1][0].files == ["last.pt", "run.json", "manifest.snapshot.json"]
