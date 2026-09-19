"""Evaluate a checkpoint on a split under one phrasing condition and save metrics.

Example (reproduce OpenAD's open-vocab number on the pretrained checkpoint):
    python -m src.evaluation.run_eval \
        --checkpoint results/checkpoints/openad_pn2.t7 \
        --condition openad_synonyms

Conditions:
    canonical         the canonical affordance labels (index-aligned)
    openad_synonyms   OpenAD's open-vocab test synonyms (their reported setting)
    label|question|description   studied affordances rephrased in that form
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..openad_bridge import build_model, load_config
from ..paths import DEFAULT_CHECKPOINT, METRICS_DIR, REF_OPENAD, ensure_dir
from ..prompts import (
    PROMPT_FORMS,
    build_vocab_variant,
    load_openad_synonyms,
    load_prompt_sets,
    load_vocabulary,
)
from .evaluate import evaluate_model, save_result

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"
OPENAD_TARGET_MIOU = 14.37  # full-shape open-vocab, from arXiv:2303.02401 Table I


def build_queries(condition: str, prompt_split: str) -> tuple[list[str], list[str]]:
    """Return (queries, canonical_class_names) for a phrasing condition."""
    vocabulary = load_vocabulary()
    canonical = list(vocabulary.canonical)

    if condition == "canonical":
        return canonical, canonical
    if condition == "openad_synonyms":
        return list(load_openad_synonyms()), canonical
    if condition in PROMPT_FORMS:
        prompt_sets = load_prompt_sets(vocabulary=vocabulary)
        queries = build_vocab_variant(prompt_sets, form=condition, split=prompt_split)
        return queries, canonical
    raise ValueError(f"unknown condition {condition!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument(
        "--condition",
        default="canonical",
        choices=["canonical", "openad_synonyms", *PROMPT_FORMS],
    )
    parser.add_argument(
        "--prompt-split", default="heldout", choices=["finetune", "heldout"],
        help="which phrase group to use for label/question/description conditions",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(METRICS_DIR))
    parser.add_argument("--no-per-sample", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("no checkpoint given (pass --checkpoint or set OPENAD_CHECKPOINT)")
    return args


def main() -> None:
    args = parse_args()

    queries, class_names = build_queries(args.condition, args.prompt_split)

    cfg = load_config(args.config, data_root=args.data_root)
    model = build_model(cfg, checkpoint_path=args.checkpoint, device=args.device)

    # Import here so the module stays importable without OpenAD/torch present.
    from ..data import load_dataset

    dataset = load_dataset(args.split, data_root=args.data_root)

    result = evaluate_model(
        model,
        dataset,
        queries,
        condition=args.condition,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        per_sample=not args.no_per_sample,
    )

    out_dir = ensure_dir(Path(args.output_dir))
    save_result(result, out_dir)

    miou = result.summary["miou"] * 100
    print(f"condition={args.condition} split={args.split}")
    print(f"  mIoU={miou:.2f}  acc={result.summary['accuracy']:.4f}")
    if args.condition in ("canonical", "openad_synonyms"):
        print(
            f"  (OpenAD full-shape open-vocab target ~{OPENAD_TARGET_MIOU}; "
            f"~12-15 is a successful reproduction — do not compare to 40+ closed-set)"
        )
    print(f"  saved to {out_dir}")


if __name__ == "__main__":
    main()
