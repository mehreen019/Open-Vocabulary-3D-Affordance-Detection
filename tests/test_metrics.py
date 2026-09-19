"""Tests for segmentation metrics, checked against hand-computed values."""

import numpy as np

from src.evaluation.metrics import SegmentationMetrics, compute_metrics


def test_perfect_prediction_scores_one():
    gt = np.array([0, 1, 2, 1, 0])
    summary = compute_metrics(gt.copy(), gt, num_classes=3)
    # ~1.0 up to the 1e-6 smoothing epsilon (kept to match OpenAD's evaluator).
    assert np.isclose(summary["miou"], 1.0, atol=1e-5)
    assert np.isclose(summary["accuracy"], 1.0, atol=1e-5)
    assert np.isclose(summary["mean_class_accuracy"], 1.0, atol=1e-5)


def test_known_confusion_matches_hand_computation():
    gt = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 2])
    summary = compute_metrics(pred, gt, num_classes=3)
    # class0: IoU 1/2, class1: IoU 2/3, class2: IoU 1/1
    assert np.isclose(summary["miou"], (0.5 + 2 / 3 + 1.0) / 3, atol=1e-4)
    assert np.isclose(summary["accuracy"], 4 / 5, atol=1e-4)
    # recalls: class0 1/2, class1 2/2, class2 1/1
    assert np.isclose(summary["mean_class_accuracy"], (0.5 + 1.0 + 1.0) / 3, atol=1e-4)


def test_streaming_update_matches_single_shot():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 4, size=200)
    pred = rng.integers(0, 4, size=200)

    one_shot = compute_metrics(pred, gt, num_classes=4)

    streamed = SegmentationMetrics(num_classes=4)
    for start in range(0, 200, 37):  # uneven batches
        chunk = slice(start, start + 37)
        streamed.update(pred[chunk], gt[chunk])

    assert np.isclose(streamed.miou(), one_shot["miou"], atol=1e-6)
    assert np.isclose(streamed.accuracy(), one_shot["accuracy"], atol=1e-6)
