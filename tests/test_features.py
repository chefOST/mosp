import hashlib
import json

import numpy as np
import pytest

from mosp import features


def test_ring_window_matches_upstream_roll_and_excludes_newest():
    """Exercise initialization, all padding, wraparound, and the one-frame lag."""
    actual = features.TemporalWindow()
    reference = np.zeros((64, 3, 2, 2), dtype=np.float32)
    for index in range(130):
        frame = np.full((3, 2, 2), index + 1, dtype=np.float32)
        reference = np.roll(reference, -1, axis=0)
        reference[-1] = frame
        result = actual.append(frame)
        np.testing.assert_array_equal(result, reference[np.arange(32) * 2].transpose(1, 0, 2, 3))
        assert result.max() < index + 1


def test_selected_context_support_is_not_the_output_anchor():
    starts, ends = features.support_timestamps(66, 10)
    assert np.isnan(starts[0]) and np.isnan(ends[0])
    np.testing.assert_allclose(starts[[1, 2, 3, 63, 64, 65]], [0, .1, 0, 0, .1, .2])
    np.testing.assert_allclose(ends[[1, 2, 3, 63, 64, 65]], [0, .1, .2, 6.2, 6.3, 6.4])


def test_rgb_processing_keeps_rgb_order_and_upstream_normalization():
    pytest.importorskip("cv2")
    frame = np.empty((4, 6, 3), dtype=np.uint8)
    frame[:] = [255, 128, 0]
    result = features.preprocess_rgb(frame)
    expected = (np.array([1, 128 / 255, 0], dtype=np.float32) - features.RGB_MEAN[:, 0, 0]) / features.RGB_STD[:, 0, 0]
    assert result.shape == (3, 256, 256)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, 50, 50], expected)


@pytest.mark.parametrize("sample_fps", [0, 24, 30, float("nan")])
def test_rejects_silent_sampling_recipe_changes(sample_fps):
    with pytest.raises(ValueError, match="10 Hz"):
        features.FeatureConfig(sample_fps=sample_fps).validate()


def _fake_extraction(monkeypatch):
    pytest.importorskip("cv2")
    frames = [np.full((2, 3, 3), 100 + i, np.uint8) for i in range(5)]
    monkeypatch.setattr(features, "probe_video", lambda _: {
        "streams": [{"width": 3, "height": 2, "start_time": "0", "avg_frame_rate": "24/1"}],
        "format": {"duration": "0.5"},
    })
    monkeypatch.setattr(features, "sampled_rgb_frames", lambda *_: (frame for frame in frames))
    monkeypatch.setattr(features.subprocess, "check_output", lambda *a, **k: "ffmpeg test fixture\n")

    def predictor(clips):
        values = clips[:, :, -1].mean(axis=(1, 2, 3))
        return np.repeat(values[:, None], 2048, axis=1)

    return predictor


def test_atomic_artifact_and_explicit_timestamps(monkeypatch, tmp_path):
    predictor = _fake_extraction(monkeypatch)
    source = tmp_path / "input.bin"
    source.write_bytes(b"unit-test placeholder; not a media artifact")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "features"
    result = features.extract_video_features(
        source, output, model=predictor, config=features.FeatureConfig(batch_size=2),
        provenance={"dataset": "EGTEA", "video_id": "fixture", "annotation_revision": "fixture",
                    "source": {"bucket": "fixture", "key": "fixture", "sha256": sha, "size_bytes": source.stat().st_size}},
    )
    assert result["status"] == "complete"
    assert result["frame_count"] == 5
    matrix = np.load(result["features_path"], allow_pickle=False)
    assert matrix.shape == (5, 2048)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], np.zeros(2048))
    np.testing.assert_allclose(np.load(result["timestamps_path"]), [0, .1, .2, .3, .4])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["features_path"] == "features.npy"
    assert metadata["source"]["sha256"] == sha
    assert metadata["native_video_modified"] is False
    assert metadata["feature_provenance"]["context_grid_indices"] == "i-63, i-61, ..., i-1"
    assert not list(tmp_path.glob(".features.partial-*"))
    with pytest.raises(FileExistsError):
        features.extract_video_features(source, output, model=predictor)
    with (output / "features.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        features.load_feature_artifact(output)


def test_failed_inference_publishes_no_partial_artifact(monkeypatch, tmp_path):
    _fake_extraction(monkeypatch)
    source = tmp_path / "input.bin"
    source.write_bytes(b"test fixture")

    def broken(clips):
        return np.full((len(clips), 2048), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        features.extract_video_features(source, tmp_path / "failed", model=broken)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.partial-*"))


def test_source_checksum_mismatch_fails_before_decoding(monkeypatch, tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"not a video")
    monkeypatch.setattr(features, "probe_video", lambda _: pytest.fail("decoder must not run"))
    with pytest.raises(ValueError, match="declared source"):
        features.extract_video_features(source, tmp_path / "features", model=lambda _: None,
                                        provenance={"source": {"sha256": "0" * 64}})
