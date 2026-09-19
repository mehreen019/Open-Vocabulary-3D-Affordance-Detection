"""Turn a point cloud + natural-language queries into per-point affordance scores.

The OpenAD model has no fixed classifier head: it aligns each point to whatever
query strings it is given (via the frozen CLIP text encoder) and returns a
log-softmax over exactly those queries. So prediction is simply "pass the points
and the query list, exponentiate the output."
"""

from __future__ import annotations

import numpy as np

from ..prompts import Vocabulary, load_vocabulary


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center to the origin and scale to the unit sphere (OpenAD's normalization)."""
    points = np.asarray(points, dtype=np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
    return centered / scale


class AffordancePredictor:
    """Run affordance queries against a loaded OpenAD model."""

    def __init__(self, model, vocabulary: Vocabulary | None = None, device: str = "cuda"):
        self.model = model
        self.vocabulary = vocabulary or load_vocabulary()
        self.device = device

    def predict(self, points: np.ndarray, queries: list[str]) -> np.ndarray:
        """Per-point probabilities over ``queries``.

        Args:
            points: raw ``(N, 3)`` coordinates (normalized internally).
            queries: the query strings; softmax is taken over exactly these.

        Returns:
            ``(N, len(queries))`` array of probabilities that sum to 1 per point.
        """
        import torch

        normalized = normalize_points(points)
        # Model expects (B, 3, N).
        tensor = (
            torch.from_numpy(normalized).float().to(self.device).unsqueeze(0).permute(0, 2, 1)
        )
        with torch.no_grad():
            log_probs = self.model(tensor, list(queries))  # (1, K, N)
        probs = log_probs.exp().squeeze(0).permute(1, 0)  # (N, K)
        return probs.cpu().numpy()

    def predict_affordance(
        self, points: np.ndarray, query: str, background: str = "none"
    ) -> np.ndarray:
        """Per-point probability of a single affordance versus a background query."""
        probs = self.predict(points, [query, background])
        return probs[:, 0]

    def predict_full_vocabulary(self, points: np.ndarray) -> np.ndarray:
        """Per-point probabilities over the full canonical vocabulary.

        ``argmax`` over the returned columns is the predicted class and matches the
        evaluation protocol.
        """
        return self.predict(points, list(self.vocabulary.canonical))
