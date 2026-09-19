"""Measure how sensitive the model is to prompt phrasing (plan Step 7).

Two complementary views, all with the same checkpoint, samples, and thresholds —
only the wording changes:

1. **Accuracy by phrasing** — mIoU and per-studied-affordance IoU under the
   canonical labels versus question and description phrasings.
2. **Heatmap consistency** — for each studied affordance, how similar are the
   per-point score maps produced by equivalent prompts (correlation + overlap of
   the positive regions).

Low consistency is exactly what motivates the interface's prompt-comparison and
query-refinement features, and what the prompt-augmented fine-tune aims to reduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inference.predictor import AffordancePredictor
from ..prompts import PromptSets, Vocabulary

#: Phrasing conditions compared for consistency (canonical label vs. natural forms).
CONSISTENCY_FORMS = ("canonical", "question", "description")


def representative_phrase(
    prompt_sets: PromptSets, affordance: str, form: str, split: str
) -> str:
    """One phrase for an (affordance, form): the canonical label, or the first
    phrase of that form in ``split`` (falls back to the canonical label)."""
    if form == "canonical":
        return affordance
    options = prompt_sets.phrases(affordance, split, form)
    return options[0] if options else affordance


@dataclass
class ConsistencyRow:
    affordance: str
    form_a: str
    form_b: str
    correlation: float  # mean Pearson correlation of score maps
    positive_iou: float  # mean IoU of thresholded positive regions


def _score_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _positive_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    pa, pb = a >= threshold, b >= threshold
    union = np.logical_or(pa, pb).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pa, pb).sum() / union)


def heatmap_consistency(
    predictor: AffordancePredictor,
    samples,
    prompt_sets: PromptSets,
    prompt_split: str = "heldout",
    threshold: float = 0.5,
) -> list[ConsistencyRow]:
    """Average heatmap agreement between phrasing forms over ``samples``.

    ``samples`` is an iterable of :class:`~src.data.samples.Sample`.
    """
    samples = list(samples)
    form_pairs = [
        (CONSISTENCY_FORMS[i], CONSISTENCY_FORMS[j])
        for i in range(len(CONSISTENCY_FORMS))
        for j in range(i + 1, len(CONSISTENCY_FORMS))
    ]

    rows: list[ConsistencyRow] = []
    for affordance in prompt_sets.studied:
        # Cache one score map per form per sample.
        score_maps = {form: [] for form in CONSISTENCY_FORMS}
        for sample in samples:
            for form in CONSISTENCY_FORMS:
                phrase = representative_phrase(prompt_sets, affordance, form, prompt_split)
                score_maps[form].append(predictor.predict_affordance(sample.points, phrase))

        for form_a, form_b in form_pairs:
            corrs, ious = [], []
            for a, b in zip(score_maps[form_a], score_maps[form_b]):
                corrs.append(_score_correlation(a, b))
                ious.append(_positive_iou(a, b, threshold))
            rows.append(
                ConsistencyRow(
                    affordance=affordance,
                    form_a=form_a,
                    form_b=form_b,
                    correlation=float(np.nanmean(corrs)),
                    positive_iou=float(np.nanmean(ious)),
                )
            )
    return rows


def studied_affordance_iou(
    results_by_condition: dict, prompt_sets: PromptSets, vocabulary: Vocabulary
) -> list[dict]:
    """Per-studied-affordance IoU across conditions, as table rows.

    ``results_by_condition`` maps a condition name to an
    :class:`~src.evaluation.evaluate.EvaluationResult`.
    """
    rows = []
    for affordance in prompt_sets.studied:
        row = {"affordance": affordance}
        for condition, result in results_by_condition.items():
            row[condition] = result.summary["per_class_iou"].get(affordance, float("nan"))
        rows.append(row)
    return rows
