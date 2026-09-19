"""Seen/unseen affordance-class aggregation (the class-disjoint eval).

This does not re-run inference. It re-aggregates the per-class IoU that
``evaluate_model``/``save_result`` already computes (see ``evaluate.py``) into
a seen-class mean and an unseen-class mean, using the split defined in
``configs/experiments/class_split.yaml``.

"Unseen" means: an affordance never referenced by name (canonical or
paraphrase) during fine-tuning's prompt augmentation -- a zero-shot probe of
the frozen CLIP text encoder's transfer, not a geometry holdout. See the
comment at the top of class_split.yaml before quoting this in a report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..paths import EXPERIMENTS_DIR
from ..prompts import Vocabulary, load_vocabulary


@dataclass(frozen=True)
class ClassSplit:
    """Seen/unseen affordance names, both disjoint and both subsets of the vocabulary."""

    vocabulary: Vocabulary
    unseen: tuple[str, ...]
    seen: tuple[str, ...]


def load_class_split(
    path: Path | None = None, vocabulary: Vocabulary | None = None
) -> ClassSplit:
    """Load the class-disjoint split (defaults to configs/experiments/class_split.yaml)."""
    path = Path(path) if path else EXPERIMENTS_DIR / "class_split.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    vocabulary = vocabulary or load_vocabulary()

    unseen = tuple(data.get("unseen", []))
    seen = tuple(name for name in vocabulary.canonical if name != "none" and name not in unseen)
    return ClassSplit(vocabulary=vocabulary, unseen=unseen, seen=seen)


def validate_class_split(split: ClassSplit, studied_affordances: tuple[str, ...] = ()) -> None:
    """Raise ``ValueError`` if the split is malformed or collides with the prompt-sensitivity study."""
    for name in split.unseen:
        if name not in split.vocabulary:
            raise ValueError(f"unseen affordance {name!r} is not in the canonical vocabulary")
    if len(set(split.unseen)) != len(split.unseen):
        raise ValueError(f"duplicate entries in unseen: {split.unseen}")
    if "none" in split.unseen:
        raise ValueError("the background class 'none' cannot be marked unseen")

    overlap = set(split.unseen) & set(studied_affordances)
    if overlap:
        raise ValueError(
            f"unseen affordances overlap prompt_sets.yaml studied_affordances: {sorted(overlap)} "
            "(the two splits must stay disjoint so fine-tuning never prompts an 'unseen' class)"
        )


def seen_unseen_miou(per_class_iou: dict[str, float], split: ClassSplit) -> dict:
    """Aggregate a per-class IoU dict (from an evaluation summary) into seen/unseen means.

    ``per_class_iou`` keys are class names, e.g. ``result.summary["per_class_iou"]``
    as saved by ``evaluate.save_result`` (or loaded back from a ``*_summary.json``).
    """
    missing = [name for name in (*split.seen, *split.unseen) if name not in per_class_iou]
    if missing:
        raise KeyError(f"per_class_iou is missing classes required by the split: {missing}")

    seen_ious = [per_class_iou[name] for name in split.seen]
    unseen_ious = [per_class_iou[name] for name in split.unseen]
    seen_miou = sum(seen_ious) / len(seen_ious)
    unseen_miou = sum(unseen_ious) / len(unseen_ious)

    return {
        "seen_classes": list(split.seen),
        "unseen_classes": list(split.unseen),
        "seen_miou": seen_miou,
        "unseen_miou": unseen_miou,
        "gap": seen_miou - unseen_miou,
        "per_class_iou": {
            "seen": {name: per_class_iou[name] for name in split.seen},
            "unseen": {name: per_class_iou[name] for name in split.unseen},
        },
    }


def aggregate_summary_file(summary_path: Path, split: ClassSplit | None = None) -> dict:
    """Load a saved ``*_summary.json`` (from ``run_eval.py``) and compute the seen/unseen gap."""
    split = split or load_class_split()
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    result = seen_unseen_miou(summary["per_class_iou"], split)
    result["condition"] = summary.get("condition")
    result["queries"] = summary.get("queries")
    return result
