"""Report the seen/unseen mIoU gap from an already-saved evaluation summary.

Does not run inference or touch the model/data/GPU -- it re-aggregates the
per-class IoU that ``run_eval.py`` already saved. Run ``run_eval.py`` first.

Example:
    python -m src.evaluation.run_eval --checkpoint results/checkpoints/openad_pn2.t7 \
        --condition canonical
    python -m src.evaluation.run_class_split \
        --summary results/metrics/canonical_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import METRICS_DIR
from .class_split import aggregate_summary_file, load_class_split, validate_class_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", required=True, help="path to a *_summary.json saved by run_eval.py"
    )
    parser.add_argument("--split", default=None, help="path to class_split.yaml (default: repo config)")
    parser.add_argument("--output-dir", default=str(METRICS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split = load_class_split(Path(args.split) if args.split else None)
    validate_class_split(split)

    result = aggregate_summary_file(Path(args.summary), split=split)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = result.get("condition") or "eval"
    out_path = out_dir / f"{condition}_seen_unseen.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"condition={result.get('condition')}")
    print(f"  seen mIoU   ({len(result['seen_classes'])} classes) = {result['seen_miou'] * 100:.2f}")
    print(f"  unseen mIoU ({len(result['unseen_classes'])} classes) = {result['unseen_miou'] * 100:.2f}")
    print(f"  gap = {result['gap'] * 100:.2f}")
    print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
