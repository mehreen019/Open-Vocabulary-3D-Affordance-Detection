"""Run the prompt-sensitivity experiment on a checkpoint and save the tables.

Example:
    python -m src.evaluation.run_prompt_sensitivity \
        --checkpoint results/checkpoints/openad_pn2.t7 --num-samples 40
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..openad_bridge import build_model, load_config
from ..paths import DEFAULT_CHECKPOINT, METRICS_DIR, REF_OPENAD, ensure_dir
from ..prompts import load_prompt_sets, load_vocabulary
from .evaluate import evaluate_model
from .prompt_sensitivity import heatmap_consistency, studied_affordance_iou
from .run_eval import build_queries

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"
CONDITIONS = ("canonical", "question", "description")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--prompt-split", default="heldout", choices=["finetune", "heldout"])
    parser.add_argument("--num-samples", type=int, default=40, help="samples for heatmap consistency")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(METRICS_DIR / "prompt_sensitivity"))
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("no checkpoint given (pass --checkpoint or set OPENAD_CHECKPOINT)")
    return args


def main() -> None:
    args = parse_args()
    vocabulary = load_vocabulary()
    prompt_sets = load_prompt_sets(vocabulary=vocabulary)

    cfg = load_config(args.config, data_root=args.data_root)
    model = build_model(cfg, checkpoint_path=args.checkpoint, device=args.device)

    from ..data import load_dataset, load_sample
    from ..inference.predictor import AffordancePredictor

    dataset = load_dataset(args.split, data_root=args.data_root)
    predictor = AffordancePredictor(model, vocabulary, device=args.device)

    # 1) Accuracy by phrasing condition.
    results_by_condition = {}
    for condition in CONDITIONS:
        queries, class_names = build_queries(condition, args.prompt_split)
        results_by_condition[condition] = evaluate_model(
            model,
            dataset,
            queries,
            condition=condition,
            class_names=class_names,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            per_sample=False,
        )

    # 2) Heatmap consistency between equivalent prompts.
    num = min(args.num_samples, len(dataset))
    samples = [load_sample(dataset, i) for i in range(num)]
    consistency = heatmap_consistency(predictor, samples, prompt_sets, args.prompt_split)

    out_dir = ensure_dir(Path(args.output_dir))
    _write_outputs(out_dir, results_by_condition, prompt_sets, vocabulary, consistency)

    print("mIoU by phrasing condition:")
    for condition, result in results_by_condition.items():
        print(f"  {condition:12s} {result.summary['miou'] * 100:.2f}")
    print(f"heatmap consistency over {num} samples -> {out_dir}")


def _write_outputs(out_dir, results_by_condition, prompt_sets, vocabulary, consistency) -> None:
    overall = {c: r.summary["miou"] for c, r in results_by_condition.items()}
    (out_dir / "miou_by_condition.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    per_aff = studied_affordance_iou(results_by_condition, prompt_sets, vocabulary)
    if per_aff:
        with (out_dir / "studied_affordance_iou.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_aff[0].keys()))
            writer.writeheader()
            writer.writerows(per_aff)

    with (out_dir / "heatmap_consistency.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["affordance", "form_a", "form_b", "correlation", "positive_iou"])
        for row in consistency:
            writer.writerow(
                [row.affordance, row.form_a, row.form_b, f"{row.correlation:.4f}", f"{row.positive_iou:.4f}"]
            )


if __name__ == "__main__":
    main()
