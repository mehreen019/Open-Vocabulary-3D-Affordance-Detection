"""Tests for the trivial-floor baselines (random / majority-per-point).

Uses a small synthetic in-memory dataset shaped like OpenAD's AffordNetDataset
batches, so these run without the real dataset, OpenAD, or a GPU -- only torch
(for the DataLoader) is required.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.baseline import TrivialBaseline, compute_class_frequencies, evaluate_baseline
from src.prompts import Vocabulary


class _FakeDataset(torch.utils.data.Dataset):
    """Yields (data, _, target, shape_id, category) like AffordNetDataset."""

    def __init__(self, labels_per_sample):
        self.labels_per_sample = labels_per_sample

    def __len__(self):
        return len(self.labels_per_sample)

    def __getitem__(self, idx):
        labels = self.labels_per_sample[idx]
        num_points = len(labels)
        data = torch.zeros(num_points, 3)  # points are irrelevant to a trivial baseline
        # AffordNetDataset returns float32 labels (not integer), so predictors
        # and frequency counting must tolerate that -- mirror it here.
        target = torch.tensor(labels, dtype=torch.float32)
        return data, 0, target, f"shape_{idx}", "category"


def test_random_baseline_predicts_in_range():
    baseline = TrivialBaseline(name="random")
    rng = np.random.default_rng(0)
    pred = baseline.predict(num_points=100, num_classes=5, rng=rng, majority_class=2)
    assert pred.shape == (100,)
    assert pred.min() >= 0 and pred.max() < 5


def test_majority_baseline_predicts_constant_class():
    baseline = TrivialBaseline(name="majority")
    rng = np.random.default_rng(0)
    pred = baseline.predict(num_points=50, num_classes=5, rng=rng, majority_class=3)
    assert (pred == 3).all()


def test_unknown_baseline_name_raises():
    baseline = TrivialBaseline(name="bogus")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        baseline.predict(num_points=10, num_classes=3, rng=rng, majority_class=0)


def test_compute_class_frequencies_matches_hand_count():
    # class 0 appears 3 times, class 1 appears 5 times, class 2 appears 2 times.
    dataset = _FakeDataset([[0, 0, 1, 1, 1], [1, 1, 2, 2, 0]])
    counts = compute_class_frequencies(dataset, num_classes=3, batch_size=2)
    assert counts.tolist() == [3, 5, 2]


def test_majority_baseline_picks_the_most_frequent_class():
    # class 1 is overwhelmingly the most frequent ground-truth class.
    labels = [[1] * 90 + [0] * 5 + [2] * 5]
    dataset = _FakeDataset(labels)
    vocabulary = Vocabulary(canonical=("a", "b", "c"))

    result = evaluate_baseline(dataset, vocabulary, name="majority", batch_size=1, per_sample=False)

    # Predicting all-class-1 gets every class-1 point right and nothing else,
    # so IoU is high for class 1 and zero for the others.
    assert result.summary["per_class_iou"]["b"] > 0.8
    assert result.summary["per_class_iou"]["a"] == 0.0
    assert result.summary["per_class_iou"]["c"] == 0.0


def test_random_baseline_runs_and_produces_bounded_miou():
    # Equal-length label arrays: real OpenAD samples are always fixed at 2048
    # points, so the default DataLoader collate can stack them directly.
    labels = [[0, 1, 2] * 20, [0, 1, 2] * 20]
    dataset = _FakeDataset(labels)
    vocabulary = Vocabulary(canonical=("a", "b", "c"))

    result = evaluate_baseline(dataset, vocabulary, name="random", batch_size=2, seed=1)

    assert 0.0 <= result.summary["miou"] <= 1.0
    assert len(result.per_sample) == 2
    assert set(result.per_sample[0].keys()) == {"shape_id", "category", "miou", "accuracy"}
