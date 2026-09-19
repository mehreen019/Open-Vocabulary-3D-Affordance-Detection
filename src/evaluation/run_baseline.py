"""Evaluate the trivial-floor baselines (random / majority-per-point).

No model, checkpoint, or GPU needed -- only the dataset. Anchors the metric
scale, as named in the PR proposal (slide 5).

Example:
    python -m src.evaluation.run_baseline --name majority
    python -m src.evaluation.run_baseline --name random
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import METRICS_DIR, ensure_dir
from ..prompts import load_vocabulary
from .baseline import evaluate_baseline
from .evaluate import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="random", choices=["random", "majority"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(METRICS_DIR))
    parser.add_argument("--no-per-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vocabulary = load_vocabulary()

    # Import here so the module stays importable without OpenAD present.
    from ..data import load_dataset

    dataset = load_dataset(args.split, data_root=args.data_root)

    result = evaluate_baseline(
        dataset,
        vocabulary,
        name=args.name,
        batch_size=args.batch_size,
        seed=args.seed,
        per_sample=not args.no_per_sample,
    )

    out_dir = ensure_dir(Path(args.output_dir))
    save_result(result, out_dir, vocabulary=vocabulary)

    miou = result.summary["miou"] * 100
    print(f"baseline={args.name} split={args.split}")
    print(f"  mIoU={miou:.2f}  acc={result.summary['accuracy']:.4f}")
    print(f"  saved to {out_dir}")


if __name__ == "__main__":
    main()
