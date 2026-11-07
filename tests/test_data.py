import json
from types import SimpleNamespace

import numpy as np
import pytest

from mosp.data import (
    IGNORE_ACTION,
    PreparedDataset,
    Sample,
    compute_state_action_correlations,
    validate_manifest,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def manifest(tmp_path):
    write_json(tmp_path / "objects.json", {"o2i": {"cup": 7}, "all2one": {"mug": "cup"}})
    write_json(tmp_path / "states.json", {"s2i": {"full": 12, "empty": 3}, "all2one": {"filled": "full"}})
    write_json(tmp_path / "actions.json", {"a2i": {"pour": "8", "cut": "20"}})
    labels = {"state": {"mug": [["empty"], ["filled"], ["full"]]}, "action": [[0, 3, "pour", "8"]]}
    write_json(tmp_path / "pseudo.json", {"train": labels, "heldout": labels, "unselected_gt": labels})
    write_json(tmp_path / "gt.json", {"heldout": {"state": {"cup": [["empty"], ["full"], ["full"]]}}, "unselected_gt": labels})
    np.save(tmp_path / "train.npy", np.arange(6, dtype=np.float32).reshape(3, 2))
    data = {
        "schema_version": 1,
        "dataset": "CMU",
        "split_provenance": {"description": "Loader unit-test fixture; not a real video split", "paper_split_verified": False},
        "vocabularies": {"objects": "objects.json", "states": "states.json", "actions": "actions.json"},
        "annotations": {"pseudo_labels": "pseudo.json", "ground_truth": "gt.json"},
        "feature_provenance": {"model": "fixture", "dimension": 2, "layout": "TD", "timeline": "3 explicit samples"},
        "splits": {"train": ["train"], "validation": [], "test": ["heldout"]},
        "videos": {
            "train": {"feature_path": "train.npy", "state_source": "pseudo_labels", "state_indices": [0, 1, 2], "action_source": "pseudo_labels", "action_indices": [0, 1, 2], "action_segment_end": "exclusive"},
        },
    }
    path = tmp_path / "manifest.json"
    write_json(path, data)
    return path


def edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    write_json(path, data)


def test_sparse_vocab_aliases_and_explicit_feature_layout(manifest):
    dataset = PreparedDataset(manifest, "train")
    sample = dataset[0]
    assert sample.features.shape == (2, 3)
    assert dataset.metadata["states"] == ["empty", "full"]
    assert dataset.metadata["actions"] == ["pour", "cut"]
    assert dataset.metadata["vocabulary_snapshot"]["states"]["source_ids"] == [3, 12]
    np.testing.assert_array_equal(sample.states, [[1, 0, 0], [0, 1, 1]])
    np.testing.assert_array_equal(sample.actions, [0, 0, 0])
    assert sample.action_segments == ((0, 3, 0),)
    assert "ground_truth" not in dataset._annotation_cache


def test_rejects_gt_training_even_when_id_not_in_selected_test(manifest):
    def contaminate(data):
        data["splits"]["train"] = ["unselected_gt"]
        data["videos"]["unselected_gt"] = data["videos"].pop("train")
    edit(manifest, contaminate)
    with pytest.raises(ValueError, match="manually annotated ground-truth"):
        PreparedDataset(manifest, "train")


@pytest.mark.parametrize("other", ["validation", "test"])
def test_rejects_split_overlap(manifest, other):
    edit(manifest, lambda data: data["splits"][other].append("train"))
    with pytest.raises(ValueError, match="Split overlap"):
        PreparedDataset(manifest, "train")


def test_rejects_implicit_splits_and_alignment(manifest):
    edit(manifest, lambda data: data["splits"].pop("validation"))
    with pytest.raises(ValueError, match="explicitly contain"):
        PreparedDataset(manifest, "train")
    edit(manifest, lambda data: data["splits"].update(validation=[]))
    edit(manifest, lambda data: data["videos"]["train"].pop("state_indices"))
    with pytest.raises(ValueError, match="Explicit state_indices"):
        PreparedDataset(manifest, "train")


def test_state_and_action_maps_use_separate_timelines(manifest):
    pseudo = json.loads((manifest.parent / "pseudo.json").read_text())
    pseudo["train"]["state"]["mug"] = [["empty"], [], ["full"], [], ["empty"]]
    pseudo["train"]["action"] = [[0, 2, "pour", "8"], [2, 3, "cut", "20"]]
    write_json(manifest.parent / "pseudo.json", pseudo)
    np.save(manifest.parent / "state_index.npy", np.array([0, 2, 4], dtype=np.int64))
    edit(manifest, lambda data: data["videos"]["train"].update(state_indices="state_index.npy"))
    sample = PreparedDataset(manifest, "train")[0]
    np.testing.assert_array_equal(sample.states, [[1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(sample.actions, [0, 0, 1])


@pytest.mark.parametrize("indices, message", [([0, 1], "3 entries"), ([0, 1, 99], "exceed"), ([0, 2, 1], "chronological"), ([0, 1, 1.5], "integer indices")])
def test_bad_mapping_fails_without_stretching(manifest, indices, message):
    edit(manifest, lambda data: data["videos"]["train"].update(state_indices=indices))
    with pytest.raises(ValueError, match=message):
        PreparedDataset(manifest, "train")[0]


def test_missing_features_and_incorrect_layout_fail(manifest):
    edit(manifest, lambda data: data["feature_provenance"].update(layout="DT"))
    with pytest.raises(ValueError, match="shape does not match"):
        PreparedDataset(manifest, "train")[0]
    edit(manifest, lambda data: data["videos"]["train"].update(feature_path="absent.npy"))
    with pytest.raises(FileNotFoundError, match="Missing features"):
        PreparedDataset(manifest, "train")


def test_unknown_actions_ignored_and_inclusive_boundaries_explicit(manifest):
    pseudo = json.loads((manifest.parent / "pseudo.json").read_text())
    pseudo["train"]["action"] = [[0, 0, "pour", "8"], [1, 1, "unknown", "999"], [2, 2, "cut", "20"]]
    write_json(manifest.parent / "pseudo.json", pseudo)
    edit(manifest, lambda data: data["videos"]["train"].update(action_segment_end="inclusive"))
    sample = PreparedDataset(manifest, "train")[0]
    assert sample.actions.tolist() == [0, IGNORE_ACTION, 1]
    assert sample.action_segments == ((0, 1, 0), (2, 3, 1))


def test_correlations_keep_same_label_action_boundaries_and_train_only(manifest):
    pseudo = json.loads((manifest.parent / "pseudo.json").read_text())
    pseudo["train"] = {"state": {"cup": [["empty"], ["empty"], ["full"], ["full"], ["full"], ["full"]]}, "action": [[0, 2, "pour", "8"], [2, 6, "pour", "8"]]}
    write_json(manifest.parent / "pseudo.json", pseudo)
    np.save(manifest.parent / "train.npy", np.ones((6, 2), dtype=np.float32))
    edit(manifest, lambda data: data["videos"]["train"].update(state_indices=list(range(6)), action_indices=list(range(6))))
    dataset = PreparedDataset(manifest, "train")
    before, after = compute_state_action_correlations(dataset)
    np.testing.assert_allclose(before, [[1 / 3, 2 / 3], [0, 0]])
    np.testing.assert_allclose(after, [[1 / 3, 2 / 3], [0, 0]])
    assert dataset[0].action_segments == ((0, 2, 0), (2, 6, 0))
    with pytest.raises(ValueError, match="train split only"):
        compute_state_action_correlations(SimpleNamespace(split="test"))


def test_correlation_midpoint_is_on_original_action_timeline(manifest):
    pseudo = json.loads((manifest.parent / "pseudo.json").read_text())
    pseudo["train"] = {"state": {"cup": [["empty"], ["empty"], ["full"]]}, "action": [[0, 10, "pour", "8"]]}
    write_json(manifest.parent / "pseudo.json", pseudo)
    edit(manifest, lambda data: data["videos"]["train"].update(action_indices=[0, 1, 9]))
    before, after = compute_state_action_correlations(PreparedDataset(manifest, "train"))
    np.testing.assert_array_equal(before[0], [1, 0])
    np.testing.assert_array_equal(after[0], [0, 1])


@pytest.mark.parametrize("split", ["validation", "test", "other"])
def test_nontraining_loaders_are_rejected_before_accessing_sources(manifest, split):
    with pytest.raises(ValueError, match="training data only"):
        PreparedDataset(manifest.parent / "absent.json", split)


def test_validate_checks_training_only_and_counts_reserved_ids(manifest):
    edit(manifest, lambda data: data["splits"].update(validation=["reserved_without_files"]))
    result = validate_manifest(manifest)
    assert result["training"] == {"videos": 1, "frames": 3}
    assert result["reserved_ids"] == {"validation": 1, "test": 1}
    assert result["ground_truth_exclusion_ids"] == 2
    assert result["features_checked"] is True
    assert set(json.loads(manifest.read_text())["videos"]) == {"train"}


def test_ground_truth_is_membership_only_not_target_data(manifest):
    write_json(manifest.parent / "gt.json", {"heldout": {"state": "not a target array"}, "unselected_gt": None})
    dataset = PreparedDataset(manifest, "train")
    np.testing.assert_array_equal(dataset[0].states, [[1, 0, 0], [0, 1, 1]])
    assert dataset.ground_truth_ids == {"heldout", "unselected_gt"}
    assert "ground_truth" not in dataset._annotation_cache
    with pytest.raises(ValueError, match="exclusion IDs only"):
        dataset._source("ground_truth")
    assert not any(name.startswith("evaluation_") for name in Sample.__dataclass_fields__)


@pytest.mark.parametrize("field", ["evaluation_indices", "evaluation_states"])
def test_training_records_reject_removed_evaluation_fields(manifest, field):
    edit(manifest, lambda data: data["videos"]["train"].update({field: [0, 1, 2]}))
    with pytest.raises(ValueError, match="Evaluation fields are not supported"):
        PreparedDataset(manifest, "train")


def test_reserved_video_records_are_rejected(manifest):
    edit(manifest, lambda data: data["videos"].update(heldout={"feature_path": "should_not_be_read.npy"}))
    with pytest.raises(ValueError, match="exactly the train IDs"):
        PreparedDataset(manifest, "train")


def test_training_cannot_select_ground_truth_as_state_source(manifest):
    edit(manifest, lambda data: data["videos"]["train"].update(state_source="ground_truth"))
    with pytest.raises(ValueError, match="Training labels must come from pseudo_labels"):
        PreparedDataset(manifest, "train")
