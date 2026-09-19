"""Evaluate a model over a dataset split under a fixed query (affordance) list.

The query list must be index-aligned with the per-point labels (i.e. the same
length and order as the canonical vocabulary); ``argmax`` over the model's output
then indexes directly into it. This lets us score any phrasing condition
(canonical labels, OpenAD synonyms, question/description variants) with identical
data, metrics, and thresholds — only the strings change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..prompts import Vocabulary, load_vocabulary
from .metrics import SegmentationMetrics, compute_metrics


@dataclass
class EvaluationResult:
    """Aggregate metrics plus optional per-sample rows for one condition."""

    condition: str
    queries: list[str]
    summary: dict
    per_sample: list[dict]


def build_eval_loader(dataset, batch_size: int = 16, num_workers: int = 0):
    """Windows-safe validation loader (default ``num_workers=0``)."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


def evaluate_model(
    model,
    dataset,
    queries: list[str],
    *,
    condition: str = "canonical",
    class_names: list[str] | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cuda",
    per_sample: bool = True,
) -> EvaluationResult:
    """Run ``model`` over ``dataset`` with ``queries`` and return metrics."""
    import torch

    num_classes = len(queries)
    class_names = class_names or list(queries)
    metrics = SegmentationMetrics(num_classes, class_names=class_names)
    rows: list[dict] = []

    loader = build_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    with torch.no_grad():
        for data, _, target, shape_ids, categories in loader:
            data = data.float().to(device).permute(0, 2, 1)  # (B, 3, N)
            logits = model(data, list(queries))  # (B, K, N) log-softmax over queries
            pred = logits.argmax(dim=1).cpu().numpy()  # (B, N)
            gt = target.reshape(target.shape[0], -1).cpu().numpy().astype(np.int64)

            metrics.update(pred, gt)
            if per_sample:
                for b in range(pred.shape[0]):
                    sample_summary = compute_metrics(pred[b], gt[b], num_classes, class_names)
                    rows.append(
                        {
                            "shape_id": str(shape_ids[b]),
                            "category": str(categories[b]),
                            "miou": sample_summary["miou"],
                            "accuracy": sample_summary["accuracy"],
                        }
                    )

    return EvaluationResult(
        condition=condition,
        queries=list(queries),
        summary=metrics.summary(),
        per_sample=rows,
    )


def save_result(result: EvaluationResult, out_dir: Path, vocabulary: Vocabulary | None = None) -> None:
    """Write aggregate JSON, per-class IoU CSV, and per-sample CSV for a condition."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = vocabulary or load_vocabulary()

    aggregate = {
        "condition": result.condition,
        "queries": result.queries,
        **result.summary,
    }
    (out_dir / f"{result.condition}_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    with (out_dir / f"{result.condition}_per_class_iou.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "query", "iou"])
        for name, query, iou in zip(
            vocabulary.canonical, result.queries, result.summary["per_class_iou"].values()
        ):
            writer.writerow([name, query, f"{iou:.6f}"])

    if result.per_sample:
        keys = list(result.per_sample[0].keys())
        with (out_dir / f"{result.condition}_per_sample.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result.per_sample)
