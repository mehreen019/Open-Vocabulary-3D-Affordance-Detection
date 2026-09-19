"""Tests for the vocabulary and prompt sets. No GPU, data, or OpenAD required."""

import random

from src.prompts import (
    PROMPT_FORMS,
    build_vocab_variant,
    load_prompt_sets,
    load_vocabulary,
    sample_training_vocab,
    validate_prompt_sets,
)


def test_vocabulary_is_index_aligned():
    vocab = load_vocabulary()
    assert vocab.canonical[0] == "grasp"
    assert vocab.canonical[-1] == "none"  # background is the last class
    assert len(vocab) == 19
    assert len(set(vocab.canonical)) == len(vocab)  # no duplicates


def test_prompt_sets_are_valid_and_frozen():
    prompt_sets = load_prompt_sets()
    validate_prompt_sets(prompt_sets)  # raises if finetune/heldout overlap, etc.
    assert set(prompt_sets.studied) <= set(prompt_sets.vocabulary.canonical)


def test_vocab_variant_keeps_length_and_indices():
    prompt_sets = load_prompt_sets()
    canonical = prompt_sets.vocabulary.canonical
    for form in PROMPT_FORMS:
        variant = build_vocab_variant(prompt_sets, form=form)
        assert len(variant) == len(canonical)
        # Non-studied classes and `none` are never rephrased.
        for i, name in enumerate(canonical):
            if name not in prompt_sets.studied:
                assert variant[i] == name


def test_training_vocab_only_uses_finetune_phrases():
    prompt_sets = load_prompt_sets()
    rng = random.Random(0)
    heldout = {
        phrase
        for aff in prompt_sets.studied
        for phrase in prompt_sets.phrases_for_split(aff, "heldout")
    }
    for _ in range(50):
        variant = sample_training_vocab(prompt_sets, rng)
        assert len(variant) == len(prompt_sets.vocabulary)
        assert not (set(variant) & heldout)  # never leak held-out phrases
