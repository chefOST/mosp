import hashlib
import json

import pytest

from mosp.pipeline_state import attach_saved_call, completed_feature_index


def test_saved_call_id_does_not_require_a_hydrated_handle(tmp_path):
    path = tmp_path / "training-call.json"
    path.write_text(json.dumps({"call_id": "fc-existing123"}))
    class LazyCall:
        @property
        def object_id(self):
            raise AttributeError("unhydrated")
    lazy = LazyCall()
    seen = []
    def restore(call_id):
        seen.append(call_id)
        return lazy
    call, call_id = attach_saved_call(path, restore)
    assert call is lazy and call_id == "fc-existing123" and seen == [call_id]


def test_invalid_saved_call_is_not_restored(tmp_path):
    path = tmp_path / "call.json"
    path.write_text(json.dumps({"call_id": None}))
    with pytest.raises(ValueError, match="call ID"):
        attach_saved_call(path, lambda _: pytest.fail("must not restore"))


def completed_fixture(tmp_path):
    source = {"bucket": "test", "key": "v.mp4", "sha256": "a" * 64, "size_bytes": 10}
    feature_root = tmp_path / "features"
    folder = feature_root / "v"
    folder.mkdir(parents=True)
    metadata = {"status": "complete", "dataset": "CMU", "video_id": "v", "source": source,
                "annotation_revision": "b" * 40,
                "feature_provenance": {"model_assets": {"checkpoint_sha256": "c" * 64}},
                "artifact_sha256": {}}
    artifacts = []
    for name in ("features", "timestamps", "support_start", "support_end"):
        path = folder / (name + ".npy")
        path.write_bytes(name.encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata[name + "_path"] = path.name
        metadata["artifact_sha256"][name] = sha
        artifacts.append({"key": "prefix/" + path.name, "sha256": sha, "size_bytes": path.stat().st_size})
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata))
    artifacts.append({"key": "prefix/metadata.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "size_bytes": path.stat().st_size})
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"videos": [{"dataset": "CMU", "video_id": "v",
                                            "metadata_path": str(path), "artifacts": artifacts}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split_provenance": {"annotation_revision": "b" * 40}}))
    training = tmp_path / "run.json"
    training.write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    return index, manifest, training, {
        "dataset": "CMU", "source_records": {"v": {"source": source}}, "feature_root": feature_root,
        "annotation_revision": "b" * 40, "checkpoint_sha256": "c" * 64}


def test_completed_publication_can_be_reused_without_gpu_calls(tmp_path):
    index, manifest, training, kwargs = completed_fixture(tmp_path)
    assert completed_feature_index(index, manifest, training, **kwargs)["videos"][0]["video_id"] == "v"


@pytest.mark.parametrize("damage", ["manifest", "metadata", "array", "membership"])
def test_inconsistent_cached_stage_is_not_reused(tmp_path, damage):
    index, manifest, training, kwargs = completed_fixture(tmp_path)
    if damage == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif damage == "metadata":
        (kwargs["feature_root"] / "v/metadata.json").write_text("{}")
    elif damage == "array":
        (kwargs["feature_root"] / "v/features.npy").unlink()
    else:
        kwargs["source_records"]["unexpected"] = {}
    with pytest.raises(ValueError):
        completed_feature_index(index, manifest, training, **kwargs)
