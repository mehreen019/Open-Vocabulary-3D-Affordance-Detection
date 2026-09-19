"""Trivial floor baselines: random and majority-class-per-point.

These anchor the metric scale (named explicitly in the PR proposal, slide 5:
"Trivial floor: random or majority-affordance per point, used to anchor the
metric scale."). Neither touches the model, CLIP, or GPU -- they only need
ground-truth labels, so they can run on any machine with the dataset present.

Random and majority predictions are made per point independently over the
full canonical vocabulary (including the background class `none`), matching
how ``evaluate_model`` scores the real model: a single argmax-style class
index per point against the same ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..prompts import Vocabulary, load_vocabulary
from .evaluate import EvaluationResult, build_eval_loader
from .metrics import SegmentationMetrics, compute_metrics


def compute_class_frequencies(dataset, num_classes: int, batch_size: int = 16) -> np.ndarray:
    """Count ground-truth points per class over a dataset split (one pass, no model)."""
    counts = np.zeros(num_classes, dtype=np.int64)
    loader = build_eval_loader(dataset, batch_size=batch_size, num_workers=0)
    for _, _, target, _, _ in loader:
        gt = target.reshape(-1).cpu().numpy().astype(np.int64)
        counts += np.bincount(gt, minlength=num_classes)
    return counts


@dataclass(frozen=True)
class TrivialBaseline:
    """A per-point prediction rule that ignores points/queries entirely."""

    name: str  # "random" or "majority"

    def predict(self, num_points: int, num_classes: int, rng: np.random.Generator, majority_class: int) -> np.ndarray:
        if self.name == "random":
            return rng.integers(0, num_classes, size=num_points, dtype=np.int64)
        if self.name == "majority":
            return np.full(num_points, majority_class, dtype=np.int64)
        raise ValueError(f"unknown baseline {self.name!r}; expected 'random' or 'majority'")


def evaluate_baseline(
    dataset,
    vocabulary: Vocabulary,
    *,
    name: str = "random",
    batch_size: int = 16,
    seed: int = 0,
    per_sample: bool = True,
) -> EvaluationResult:
    """Score a trivial baseline over ``dataset`` using the same metrics as the real model."""
    class_names = list(vocabulary.canonical)
    num_classes = len(class_names)
    baseline = TrivialBaseline(name=name)
    rng = np.random.default_rng(seed)

    frequencies = compute_class_frequencies(dataset, num_classes, batch_size=batch_size)
    majority_class = int(frequencies.argmax())

    metrics = SegmentationMetrics(num_classes, class_names=class_names)
    rows: list[dict] = []

    loader = build_eval_loader(dataset, batch_size=batch_size, num_workers=0)
    for _, _, target, shape_ids, categories in loader:
        gt_batch = target.reshape(target.shape[0], -1).cpu().numpy().astype(np.int64)

        for b in range(gt_batch.shape[0]):
            gt = gt_batch[b]
            pred = baseline.predict(gt.shape[0], num_classes, rng, majority_class)

            metrics.update(pred, gt)
            if per_sample:
                sample_summary = compute_metrics(pred, gt, num_classes, class_names)
                rows.append(
                    {
                        "shape_id": str(shape_ids[b]),
                        "category": str(categories[b]),
                        "miou": sample_summary["miou"],
                        "accuracy": sample_summary["accuracy"],
                    }
                )

    return EvaluationResult(
        condition=f"baseline_{name}",
        queries=class_names,
        summary=metrics.summary(),
        per_sample=rows,
    )
