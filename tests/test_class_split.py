"""Tests for the class-disjoint (seen/unseen) evaluation split.

No GPU, data, or OpenAD required -- seen_unseen_miou operates on a per-class
IoU dict, the same shape ``evaluate.save_result`` writes to a *_summary.json.
"""

import pytest

from src.evaluation.class_split import (
    load_class_split,
    seen_unseen_miou,
    validate_class_split,
)
from src.prompts import load_prompt_sets, load_vocabulary


def test_split_covers_vocabulary_without_overlap():
    vocab = load_vocabulary()
    split = load_class_split(vocabulary=vocab)

    assert set(split.unseen) <= set(vocab.canonical)
    assert set(split.seen) <= set(vocab.canonical)
    assert not (set(split.seen) & set(split.unseen))  # disjoint
    assert "none" not in split.unseen and "none" not in split.seen  # background excluded
    assert set(split.seen) | set(split.unseen) == set(vocab.canonical) - {"none"}


def test_split_disjoint_from_studied_affordances():
    vocab = load_vocabulary()
    split = load_class_split(vocabulary=vocab)
    prompt_sets = load_prompt_sets(vocabulary=vocab)

    validate_class_split(split, studied_affordances=prompt_sets.studied)  # must not raise


def test_validate_rejects_studied_overlap():
    vocab = load_vocabulary()
    split = load_class_split(vocabulary=vocab)
    colliding_studied = (split.unseen[0],)

    with pytest.raises(ValueError):
        validate_class_split(split, studied_affordances=colliding_studied)


def test_seen_unseen_miou_aggregates_correctly():
    vocab = load_vocabulary()
    split = load_class_split(vocabulary=vocab)

    # Synthetic per-class IoU: seen classes at 0.8, unseen at 0.2, so the
    # aggregation is trivially checkable by hand.
    per_class_iou = {name: (0.8 if name in split.seen else 0.2) for name in vocab.canonical}

    result = seen_unseen_miou(per_class_iou, split)

    assert result["seen_miou"] == pytest.approx(0.8)
    assert result["unseen_miou"] == pytest.approx(0.2)
    assert result["gap"] == pytest.approx(0.6)
    assert set(result["per_class_iou"]["seen"]) == set(split.seen)
    assert set(result["per_class_iou"]["unseen"]) == set(split.unseen)


def test_seen_unseen_miou_raises_on_missing_classes():
    vocab = load_vocabulary()
    split = load_class_split(vocabulary=vocab)
    incomplete = {name: 0.5 for name in split.seen}  # unseen classes missing

    with pytest.raises(KeyError):
        seen_unseen_miou(incomplete, split)
