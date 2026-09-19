"""Affordance vocabulary and prompt sets.

This is the single source of truth for:

* the canonical, index-aligned class vocabulary (must match the dataset labels), and
* the frozen prompt sets used by the prompt-sensitivity experiment and by
  prompt-augmented fine-tuning.

The model aligns points to whatever query strings it is given, so a "prompt
variant" is simply the canonical vocabulary with some studied classes' strings
replaced by a paraphrase. The per-point target indices never change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import PROMPTS_DIR

#: The three phrasing forms studied.
PROMPT_FORMS = ("label", "question", "description")

#: The two disjoint phrase groups.
PROMPT_SPLITS = ("finetune", "heldout")


@dataclass(frozen=True)
class Vocabulary:
    """Ordered, index-aligned affordance class names."""

    canonical: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.canonical)

    def index(self, affordance: str) -> int:
        return self.canonical.index(affordance)

    def __contains__(self, affordance: str) -> bool:
        return affordance in self.canonical


@dataclass(frozen=True)
class PromptSets:
    """Frozen paraphrases for the studied affordances.

    ``phrasings[affordance][split][form]`` is a tuple of phrases.
    """

    vocabulary: Vocabulary
    studied: tuple[str, ...]
    phrasings: dict[str, dict[str, dict[str, tuple[str, ...]]]]

    def phrases(self, affordance: str, split: str, form: str) -> tuple[str, ...]:
        """Phrases for one (affordance, split, form); empty tuple if none."""
        return self.phrasings.get(affordance, {}).get(split, {}).get(form, ())

    def phrases_for_split(self, affordance: str, split: str) -> tuple[str, ...]:
        """All phrases for an affordance in a split, across every form."""
        by_form = self.phrasings.get(affordance, {}).get(split, {})
        return tuple(p for form in PROMPT_FORMS for p in by_form.get(form, ()))


def load_vocabulary(path: Path | None = None) -> Vocabulary:
    """Load the canonical vocabulary (defaults to configs/prompts/vocabulary.yaml)."""
    path = Path(path) if path else PROMPTS_DIR / "vocabulary.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Vocabulary(canonical=tuple(data["canonical"]))


def load_openad_synonyms(path: Path | None = None) -> tuple[str, ...]:
    """Load OpenAD's index-aligned open-vocab test synonyms (for reproduction)."""
    path = Path(path) if path else PROMPTS_DIR / "vocabulary.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(data.get("openad_val_synonyms", []))


def load_prompt_sets(
    path: Path | None = None, vocabulary: Vocabulary | None = None
) -> PromptSets:
    """Load the frozen prompt sets (defaults to configs/prompts/prompt_sets.yaml)."""
    path = Path(path) if path else PROMPTS_DIR / "prompt_sets.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    vocabulary = vocabulary or load_vocabulary()

    phrasings: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for affordance, splits in (data.get("prompts") or {}).items():
        phrasings[affordance] = {
            split: {
                form: tuple(splits.get(split, {}).get(form, []) or [])
                for form in PROMPT_FORMS
            }
            for split in PROMPT_SPLITS
        }

    return PromptSets(
        vocabulary=vocabulary,
        studied=tuple(data.get("studied_affordances", [])),
        phrasings=phrasings,
    )


def build_vocab_variant(
    prompt_sets: PromptSets,
    form: str,
    split: str = "finetune",
    rng: random.Random | None = None,
) -> list[str]:
    """Return a full index-aligned vocabulary with studied classes rephrased.

    Studied affordances are replaced by a phrase of ``form`` drawn from ``split``
    (randomly if ``rng`` is given, otherwise the first phrase). Non-studied
    classes keep their canonical label. This is used by the prompt-sensitivity
    experiment (e.g. ``form="question"``) with the same target indices.
    """
    if form not in PROMPT_FORMS:
        raise ValueError(f"unknown form {form!r}; expected one of {PROMPT_FORMS}")

    variant = list(prompt_sets.vocabulary.canonical)
    for affordance in prompt_sets.studied:
        options = prompt_sets.phrases(affordance, split, form)
        if not options:
            continue  # keep canonical if this class has no phrase of that form
        choice = rng.choice(options) if rng is not None else options[0]
        variant[prompt_sets.vocabulary.index(affordance)] = choice
    return variant


def sample_training_vocab(
    prompt_sets: PromptSets, rng: random.Random
) -> list[str]:
    """Return a randomly paraphrased vocabulary for one training step.

    For each studied affordance, sample one phrase from any *finetune* form; other
    classes keep their canonical label. Held-out phrases are never used here.
    """
    variant = list(prompt_sets.vocabulary.canonical)
    for affordance in prompt_sets.studied:
        options = prompt_sets.phrases_for_split(affordance, "finetune")
        if options:
            variant[prompt_sets.vocabulary.index(affordance)] = rng.choice(options)
    return variant


def validate_prompt_sets(prompt_sets: PromptSets) -> None:
    """Raise ``ValueError`` if the frozen prompt sets violate the design rules."""
    vocab = prompt_sets.vocabulary

    for affordance in prompt_sets.studied:
        if affordance not in vocab:
            raise ValueError(
                f"studied affordance {affordance!r} is not in the canonical vocabulary"
            )
        finetune = set(prompt_sets.phrases_for_split(affordance, "finetune"))
        heldout = set(prompt_sets.phrases_for_split(affordance, "heldout"))
        overlap = finetune & heldout
        if overlap:
            raise ValueError(
                f"{affordance!r}: phrases appear in both finetune and heldout "
                f"(evaluation leakage): {sorted(overlap)}"
            )
        if not finetune:
            raise ValueError(f"{affordance!r}: no finetune phrases defined")
        if not heldout:
            raise ValueError(f"{affordance!r}: no heldout phrases defined")
