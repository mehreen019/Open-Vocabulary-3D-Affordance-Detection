"""Backend for the interactive demo: samples, models, and cached predictions.

Kept UI-free so it can be unit-tested and reused. Predictions are cached per
(sample, query, model) so repeated or compared queries are instant — the basis of
the "build around cached predictions" strategy that keeps the demo responsive even
without fast GPU access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..prompts import load_vocabulary

#: Prediction states the interface must surface distinctly.
STATE_SUCCESS = "success"
STATE_LOW_CONFIDENCE = "low_confidence"
STATE_INVALID = "invalid"
STATE_ERROR = "error"


@dataclass
class Prediction:
    points: np.ndarray  # (N, 3)
    scores: np.ndarray  # (N,) affordance probability
    summary: str  # human-readable markdown summary
    state: str


class DemoBackend:
    """Holds loaded models and samples and answers affordance queries."""

    def __init__(
        self,
        config_path: str,
        checkpoints: dict[str, str],
        *,
        split: str = "val",
        num_samples: int = 20,
        data_root: str | None = None,
        device: str = "cuda",
        low_confidence: float = 0.5,
    ):
        self.config_path = config_path
        self.checkpoints = checkpoints
        self.split = split
        self.num_samples = num_samples
        self.data_root = data_root
        self.device = device
        self.low_confidence = low_confidence

        self.vocabulary = load_vocabulary()
        self._predictors: dict = {}
        self._samples: list = []
        self._cache: dict[tuple, Prediction] = {}

    def load(self) -> "DemoBackend":
        """Load the model(s) and the sample pool (requires OpenAD + GPU + data)."""
        from ..data import load_dataset, load_sample
        from ..inference.predictor import AffordancePredictor
        from ..openad_bridge import build_model, load_config

        cfg = load_config(self.config_path, data_root=self.data_root)
        for name, checkpoint in self.checkpoints.items():
            model = build_model(cfg, checkpoint_path=checkpoint, device=self.device)
            self._predictors[name] = AffordancePredictor(model, self.vocabulary, self.device)

        dataset = load_dataset(self.split, data_root=self.data_root)
        count = min(self.num_samples, len(dataset))
        self._samples = [load_sample(dataset, i) for i in range(count)]
        return self

    @property
    def sample_labels(self) -> list[str]:
        return [f"{i}: {s.category} ({s.shape_id})" for i, s in enumerate(self._samples)]

    @property
    def model_names(self) -> list[str]:
        return list(self._predictors)

    def predict(self, sample_index: int, query: str, model_name: str) -> Prediction:
        """Return a cached or freshly computed prediction, with an explicit state."""
        query = (query or "").strip()
        if not query:
            return Prediction(
                np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
                "Enter an affordance query to begin (e.g. `grasp`, `where can I sit?`).",
                STATE_INVALID,
            )

        key = (sample_index, query.lower(), model_name)
        if key in self._cache:
            return self._cache[key]

        try:
            sample = self._samples[sample_index]
            scores = self._predictors[model_name].predict_affordance(sample.points, query)
            prediction = Prediction(
                sample.points, scores, self._summarize(query, scores), self._state(scores)
            )
        except Exception as exc:  # surfaced to the UI, never crashes the app
            prediction = Prediction(
                np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
                f"Inference failed: {exc}", STATE_ERROR,
            )

        self._cache[key] = prediction
        return prediction

    def _state(self, scores: np.ndarray) -> str:
        return STATE_SUCCESS if float(scores.max()) >= self.low_confidence else STATE_LOW_CONFIDENCE

    def _summarize(self, query: str, scores: np.ndarray) -> str:
        highlighted = float((scores >= self.low_confidence).mean()) * 100
        peak, mean = float(scores.max()), float(scores.mean())
        note = ""
        if peak < self.low_confidence:
            note = "\n\n⚠️ Low confidence everywhere — the model is unsure; treat this as unreliable."
        return (
            f"**Query:** {query}\n\n"
            f"- Highlighted points (score ≥ {self.low_confidence:g}): **{highlighted:.1f}%**\n"
            f"- Peak score: {peak:.2f} · mean: {mean:.2f}"
            f"{note}"
        )
