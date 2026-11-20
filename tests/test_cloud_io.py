import pytest

from mosp import cloud_io


def test_inventory_keeps_exact_ids_and_ground_truth_prefix(monkeypatch):
    documents = {
        cloud_io.VIDEO_MANIFESTS[0]: {"entries": [
            {"dataset": "CMU", "ground_truth_video_id": "S01_Eggs", "local_path": "_copies/MP4/CMU/S01_Eggs.mp4",
             "sha256": "a" * 64, "bytes": 10, "media": {"streams": [{"codec_type": "video", "nb_frames": "30"}]}},
            {"dataset": "EgoPER", "ground_truth_video_id": "not-available"}]},
        cloud_io.VIDEO_MANIFESTS[1]: {"entries": [
            {"dataset": "EGTEA", "video_id": "P01", "object_key": "moscato/non_ground_truth/EGTEA/P01.mp4",
             "sha256": "b" * 64, "size_bytes": 20, "video": {"nb_frames": "24"}}]},
    }
    monkeypatch.setattr(cloud_io, "read_json_object", lambda _, key: documents[key])
    rows = cloud_io.video_inventory(None)["videos"]
    assert len(rows) == 2
    assert rows[0]["source"]["key"] == "moscato/releases/564f350e536734db/videos/_copies/MP4/CMU/S01_Eggs.mp4"
    assert rows[0]["video"]["nb_frames"] == "30"
    documents[cloud_io.VIDEO_MANIFESTS[1]]["entries"][0]["object_key"] = "wrong-recording.mp4"
    with pytest.raises(ValueError, match="identity"):
        cloud_io.video_inventory(None)


def test_download_rejects_wrong_video_bytes(tmp_path):
    body = b"wrong content"
    class Client:
        def head_object(self, **kwargs):
            return {"ContentLength": len(body)}
        def download_file(self, bucket, key, filename):
            from pathlib import Path
            Path(filename).write_bytes(body)
    row = {"video_id": "sample", "source": {"bucket": cloud_io.BUCKET, "key": "sample.mp4",
                                            "size_bytes": len(body), "sha256": "a" * 64}}
    with pytest.raises(ValueError, match="checksum"):
        cloud_io.download_verified_video(Client(), row, tmp_path / "sample.mp4")


def test_upload_refuses_existing_different_object(tmp_path):
    path = tmp_path / "features.npy"
    path.write_bytes(b"features")
    class Client:
        def head_object(self, **kwargs):
            return {"ContentLength": 8, "Metadata": {"sha256": "different"}, "ETag": '"etag"'}
        def put_object(self, **kwargs):
            raise AssertionError("Existing object must not be overwritten")
    with pytest.raises(ValueError, match="replace"):
        cloud_io.put_verified_file(Client(), path, "features.npy")


def test_feature_publication_marker_follows_arrays(monkeypatch, tmp_path):
    record = {"dataset": "CMU", "video_id": "sample", "frame_count": 3}
    for name in ("features", "timestamps", "support_start", "support_end"):
        record[name + "_path"] = str(tmp_path / (name + ".npy"))
    monkeypatch.setattr(cloud_io, "load_feature_artifact", lambda _: record)
    calls = []
    monkeypatch.setattr(cloud_io, "put_verified_file", lambda client, path, key, **kwargs:
                        calls.append(key) or {"key": key})
    cloud_io.publish_features(None, tmp_path, "recipe")
    assert len(calls) == 5
    assert calls[-1].endswith("/metadata.json")
    assert all(not key.endswith("metadata.json") for key in calls[:-1])
