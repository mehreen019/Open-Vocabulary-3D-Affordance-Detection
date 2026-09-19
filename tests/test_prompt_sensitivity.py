"""Tests for heatmap-consistency plumbing (no GPU/model needed)."""

from dataclasses import dataclass

import numpy as np

from src.evaluation.prompt_sensitivity import CONSISTENCY_FORMS, heatmap_consistency
from src.prompts import load_prompt_sets


@dataclass
class FakeSample:
    points: np.ndarray


class ConstantPredictor:
    """Returns the same score map for every phrase -> perfect consistency."""

    def __init__(self, scores):
        self._scores = scores

    def predict_affordance(self, points, query, background="none"):
        return self._scores


def test_identical_heatmaps_are_perfectly_consistent():
    prompt_sets = load_prompt_sets()
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    predictor = ConstantPredictor(scores)
    samples = [FakeSample(np.zeros((5, 3), dtype=np.float32))]

    rows = heatmap_consistency(predictor, samples, prompt_sets)

    # One row per studied affordance per form pair (3 forms -> 3 pairs).
    expected_pairs = len(prompt_sets.studied) * 3
    assert len(rows) == expected_pairs
    for row in rows:
        assert row.form_a in CONSISTENCY_FORMS and row.form_b in CONSISTENCY_FORMS
        assert np.isclose(row.correlation, 1.0, atol=1e-6)
        assert np.isclose(row.positive_iou, 1.0, atol=1e-6)
