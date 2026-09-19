"""Tests for the demo backend logic that needs no model/GPU."""

import numpy as np

from src.app.backend import (
    STATE_INVALID,
    STATE_LOW_CONFIDENCE,
    STATE_SUCCESS,
    DemoBackend,
)


def _backend() -> DemoBackend:
    return DemoBackend(config_path="unused", checkpoints={"pretrained": "unused"})


def test_empty_query_is_invalid_without_loading():
    # The empty-query guard returns before touching any model or sample.
    pred = _backend().predict(sample_index=0, query="   ", model_name="pretrained")
    assert pred.state == STATE_INVALID
    assert pred.points.shape == (0, 3)


def test_state_thresholding():
    backend = _backend()
    assert backend._state(np.array([0.1, 0.2, 0.49])) == STATE_LOW_CONFIDENCE
    assert backend._state(np.array([0.1, 0.9])) == STATE_SUCCESS


def test_summary_flags_low_confidence():
    backend = _backend()
    low = backend._summarize("grasp", np.array([0.1, 0.2, 0.3]))
    high = backend._summarize("grasp", np.array([0.1, 0.9]))
    assert "Low confidence" in low
    assert "Low confidence" not in high
