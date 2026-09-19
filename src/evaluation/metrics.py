"""Point-level segmentation metrics.

Definitions match OpenAD's evaluator (``_ref_openad/utils/eval.py``) so our numbers
are comparable to theirs:

* per-class IoU  = intersection / union over all points, per class
* mIoU           = mean of per-class IoU
* accuracy       = fraction of points predicted correctly
* mean class acc = mean per-class recall

The accumulator supports streaming over batches; :func:`compute_metrics` is a
one-shot helper for arrays already in memory (used for per-sample scores).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-6


@dataclass
class SegmentationMetrics:
    """Streaming accumulator for point-level segmentation metrics."""

    num_classes: int
    class_names: list[str] | None = None
    intersection: np.ndarray = field(init=False)
    union: np.ndarray = field(init=False)
    support: np.ndarray = field(init=False)  # ground-truth points per class
    total_correct: int = field(init=False, default=0)
    total_points: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.intersection = np.zeros(self.num_classes, dtype=np.int64)
        self.union = np.zeros(self.num_classes, dtype=np.int64)
        self.support = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Accumulate one batch of predicted and ground-truth class indices."""
        pred = np.asarray(pred).reshape(-1)
        gt = np.asarray(gt).reshape(-1)
        self.total_correct += int((pred == gt).sum())
        self.total_points += gt.size
        for c in range(self.num_classes):
            predicted_c = pred == c
            actual_c = gt == c
            self.intersection[c] += int((predicted_c & actual_c).sum())
            self.union[c] += int((predicted_c | actual_c).sum())
            self.support[c] += int(actual_c.sum())

    def per_class_iou(self) -> np.ndarray:
        return self.intersection / (self.union + _EPS)

    def miou(self) -> float:
        return float(self.per_class_iou().mean())

    def accuracy(self) -> float:
        return self.total_correct / (self.total_points + _EPS)

    def mean_class_accuracy(self) -> float:
        return float((self.intersection / (self.support + _EPS)).mean())

    def summary(self) -> dict:
        """A JSON-friendly summary of all metrics."""
        iou = self.per_class_iou()
        names = self.class_names or [str(i) for i in range(self.num_classes)]
        return {
            "miou": self.miou(),
            "accuracy": self.accuracy(),
            "mean_class_accuracy": self.mean_class_accuracy(),
            "per_class_iou": {name: float(v) for name, v in zip(names, iou)},
        }


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, num_classes: int, class_names: list[str] | None = None
) -> dict:
    """Compute the metric summary for a single prediction/ground-truth pair."""
    metrics = SegmentationMetrics(num_classes, class_names)
    metrics.update(pred, gt)
    return metrics.summary()
