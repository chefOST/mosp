import numpy as np
import pytest

from mosp.metrics import average_precision, close_short_gaps, f1_max, mean_ap, state_metrics


def test_ap_includes_first_recall_increment_and_handles_ties():
    assert average_precision([0.9, 0.1], [1, 0]) == 1.0
    assert average_precision([0.5, 0.5], [1, 0]) == 0.5
    assert average_precision([0.5, 0.5], [0, 1]) == 0.5
    assert average_precision([0.9, 0.7, 0.2], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)


def test_empty_ap_class_policy_is_explicit():
    p, y = np.array([[0.9, 0.1]]), np.array([[1, 0]])
    assert mean_ap(p, y, empty_class_policy="exclude")["value"] == 100
    assert mean_ap(p, y, empty_class_policy="zero")["value"] == 50


def test_gap_merging_strict_threshold_and_boundaries():
    assert close_short_gaps([0, 1, 0, 0, 1, 0], 3).tolist() == [False, True, True, True, True, False]
    assert close_short_gaps([1, 0, 0, 1], 2).tolist() == [True, False, False, True]


def test_segments_preserve_video_edges_and_count_false_positives():
    videos = [(np.ones((2, 1)), np.ones((2, 1))), (np.ones((2, 1)), np.zeros((2, 1)))]
    result = state_metrics(videos, segment_threshold=0.5, gap_frames=10, empty_class_policy="exclude")
    assert result["segment_F1"]["0.5"]["tp"] == 1
    assert result["segment_F1"]["0.5"]["fp"] == 1
    assert result["segment_F1"]["0.5"]["value"] == pytest.approx(200 / 3)


def test_f1_is_pooled_not_mean_of_framewise_scores():
    result = f1_max([[0.9, 0.9], [0.9, 0.1]], [[1, 0], [1, 0]])
    assert result["value"] == 80


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError, match="probabilities"):
        f1_max([[float("nan")]], [[1]])
