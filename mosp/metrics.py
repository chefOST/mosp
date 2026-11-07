"""Paper-style state metrics; these are distinct from ChefOST's project metrics.

F1 counts are pooled over the test set (§5.1). AP handles tied scores as a group.
The paper does not specify treatment of classes with no positives, so callers
must choose and record that policy. Inputs are probabilities, never logits.
"""

from __future__ import annotations

import numpy as np


def _validate(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("Expected matching [frames, object-state classes] arrays")
    if 0 in probabilities.shape:
        raise ValueError("Cannot score empty predictions")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("State targets must be binary")
    return probabilities, targets.astype(bool)


def f1_max(probabilities, targets):
    probabilities, targets = _validate(probabilities, targets)
    curve = []
    for threshold in np.arange(1, 10) / 10:
        predictions = probabilities >= threshold
        tp = int(np.count_nonzero(predictions & targets))
        fp = int(np.count_nonzero(predictions & ~targets))
        fn = int(np.count_nonzero(~predictions & targets))
        denom = 2 * tp + fp + fn
        curve.append({"threshold": float(threshold), "f1": 100 * 2 * tp / denom if denom else 0.0,
                      "tp": tp, "fp": fp, "fn": fn})
    best = max(curve, key=lambda point: point["f1"])
    return {"value": best["f1"], "threshold": best["threshold"], "curve": curve}


def average_precision(scores, targets):
    """Step-integrated PR area, with tied scores evaluated simultaneously."""
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=bool)
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_targets = scores[order], targets[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_targets)[group_ends]
    precision = tp / (group_ends + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def mean_ap(probabilities, targets, *, empty_class_policy):
    probabilities, targets = _validate(probabilities, targets)
    if empty_class_policy not in {"exclude", "zero"}:
        raise ValueError("empty_class_policy must explicitly be 'exclude' or 'zero'")
    included = targets.any(axis=0) if empty_class_policy == "exclude" else np.ones(targets.shape[1], dtype=bool)
    scores = [average_precision(probabilities[:, c], targets[:, c]) for c in np.flatnonzero(included)]
    return {"value": 100 * float(np.mean(scores)) if scores else 0.0,
            "classes_scored": len(scores), "classes_total": targets.shape[1],
            "empty_class_policy": empty_class_policy}


def segments(binary):
    binary = np.asarray(binary, dtype=bool)
    edges = np.diff(np.r_[False, binary, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def close_short_gaps(binary, maximum_gap_exclusive):
    """Merge positive runs separated by fewer than the specified background frames."""
    if maximum_gap_exclusive < 0:
        raise ValueError("Gap limit cannot be negative")
    result = np.asarray(binary, dtype=bool).copy()
    runs = segments(result)
    for (_, end), (start, _) in zip(runs, runs[1:]):
        if start - end < maximum_gap_exclusive:
            result[end:start] = True
    return result


def _segment_counts(predicted, target, overlap):
    preds, truth = segments(predicted), segments(target)
    matched = set()
    tp = 0
    for left, right in preds:
        ious = [max(0, min(right, end) - max(left, start)) / (max(right, end) - min(left, start))
                for start, end in truth]
        if ious:
            best = int(np.argmax(ious))
            if ious[best] >= overlap and best not in matched:
                matched.add(best)
                tp += 1
    return tp, len(preds) - tp, len(truth) - tp


def state_metrics(videos, *, segment_threshold, gap_frames, empty_class_policy):
    """Score a list of (probabilities[T,C], targets[T,C]), preserving video edges.

    gap_frames is measured in the *evaluated timeline*, not raw-video frames.
    No resampling or cross-video concatenation is used for segment scoring.
    """
    if not 0 <= segment_threshold <= 1 or gap_frames < 0:
        raise ValueError("Invalid segment threshold or gap length")
    if not videos:
        raise ValueError("No videos to evaluate")
    checked = [_validate(p, y) for p, y in videos]
    if len({p.shape[1] for p, _ in checked}) != 1:
        raise ValueError("All videos must use the same object-state vocabulary")
    probabilities = np.concatenate([p for p, _ in checked])
    targets = np.concatenate([y for _, y in checked])
    segment_scores = {}
    for overlap in (0.1, 0.25, 0.5):
        totals = np.zeros(3, dtype=np.int64)
        for probs, labels in checked:
            for c in range(probs.shape[1]):
                prediction = close_short_gaps(probs[:, c] >= segment_threshold, gap_frames)
                totals += _segment_counts(prediction, labels[:, c], overlap)
        tp, fp, fn = map(int, totals)
        denominator = 2 * tp + fp + fn
        segment_scores[str(overlap)] = {"value": 100 * 2 * tp / denominator if denominator else 0.0,
                                         "tp": tp, "fp": fp, "fn": fn}
    return {"F1-max": f1_max(probabilities, targets),
            "mAP": mean_ap(probabilities, targets, empty_class_policy=empty_class_policy),
            "segment_F1": segment_scores,
            "protocol": {"segment_threshold": segment_threshold, "gap_frames": gap_frames,
                         "gap_unit": "evaluation timeline samples", "num_videos": len(checked),
                         "num_frames": len(targets), "score_unit": "percent",
                         "aggregation": "pooled TP/FP/FN across videos and object-state classes",
                         "warning": "F1-max selects a test-set threshold for this reported metric only"}}
