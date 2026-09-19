"""Fair before/after comparison of two checkpoints (plan Step 9).

Evaluates each model under identical conditions (same data, metrics, thresholds;
only the prompt wording changes) and writes the comparison table plus the
per-studied-affordance change on held-out phrasings. This is the PR project's
main quantitative result: does prompt-augmented fine-tuning improve held-out
phrasing without hurting canonical labels?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..openad_bridge import build_model, load_config
from ..paths import METRICS_DIR, REF_OPENAD, ensure_dir
from ..prompts import load_prompt_sets, load_vocabulary
from .evaluate import evaluate_model
from .run_eval import build_queries

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"

#: (name, form, split); form "canonical" ignores split.
DEFAULT_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("canonical", "canonical", "heldout"),
    ("question_finetune", "question", "finetune"),
    ("question_heldout", "question", "heldout"),
    ("description_finetune", "description", "finetune"),
    ("description_heldout", "description", "heldout"),
)

#: The held-out natural-phrasing conditions that define the headline improvement.
HEADLINE_HELDOUT = ("question_heldout", "description_heldout")


def compare(
    cfg,
    checkpoints: dict[str, str],
    *,
    conditions=DEFAULT_CONDITIONS,
    data_root=None,
    split: str = "val",
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cuda",
) -> dict:
    """Return ``{model_name: {condition_name: EvaluationResult}}``."""
    from ..data import load_dataset

    vocabulary = load_vocabulary()
    dataset = load_dataset(split, data_root=data_root)

    results: dict[str, dict] = {}
    for model_name, checkpoint in checkpoints.items():
        model = build_model(cfg, checkpoint_path=checkpoint, device=device)
        results[model_name] = {}
        for cond_name, form, prompt_split in conditions:
            queries, class_names = build_queries(form, prompt_split)
            results[model_name][cond_name] = evaluate_model(
                model, dataset, queries,
                condition=cond_name, class_names=class_names,
                batch_size=batch_size, num_workers=num_workers, device=device,
                per_sample=False,
            )
        del model  # free GPU memory before loading the next checkpoint
    return results


def write_comparison(results: dict, conditions, out_dir: Path) -> None:
    """Write the mIoU comparison table and the held-out per-affordance change."""
    out_dir = ensure_dir(out_dir)
    model_names = list(results)
    cond_names = [c[0] for c in conditions]

    # mIoU table: rows = condition, columns = model.
    with (out_dir / "comparison_miou.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", *model_names])
        for cond in cond_names:
            writer.writerow(
                [cond, *[f"{results[m][cond].summary['miou'] * 100:.2f}" for m in model_names]]
            )

    (out_dir / "comparison_miou.json").write_text(
        json.dumps(
            {m: {c: results[m][c].summary["miou"] for c in cond_names} for m in model_names},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Per-studied-affordance IoU on held-out phrasings, per model (for delta analysis).
    prompt_sets = load_prompt_sets()
    with (out_dir / "heldout_affordance_iou.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["affordance", "condition", *model_names])
        for affordance in prompt_sets.studied:
            for cond in HEADLINE_HELDOUT:
                if cond not in cond_names:
                    continue
                writer.writerow(
                    [affordance, cond,
                     *[f"{results[m][cond].summary['per_class_iou'].get(affordance, float('nan')):.4f}"
                       for m in model_names]]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pretrained", required=True, help="pretrained checkpoint path")
    parser.add_argument("--finetuned", required=True, help="fine-tuned checkpoint path")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(METRICS_DIR / "comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, data_root=args.data_root)
    checkpoints = {"pretrained": args.pretrained, "finetuned": args.finetuned}

    results = compare(
        cfg, checkpoints, data_root=args.data_root, split=args.split,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
    )
    write_comparison(results, DEFAULT_CONDITIONS, Path(args.output_dir))

    print("mIoU (%)  condition            pretrained  finetuned  delta")
    for cond, *_ in DEFAULT_CONDITIONS:
        pre = results["pretrained"][cond].summary["miou"] * 100
        fin = results["finetuned"][cond].summary["miou"] * 100
        print(f"          {cond:20s} {pre:9.2f}  {fin:8.2f}  {fin - pre:+.2f}")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
