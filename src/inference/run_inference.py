"""Run one affordance query on one object and save a reusable prediction artifact.

Example:
    python -m src.inference.run_inference \
        --checkpoint results/checkpoints/openad_pn2.t7 \
        --split val --index 0 --query "grasp"

Outputs an ``.npz`` (raw points, labels, per-point scores) and a ``.json`` sidecar
(metadata) under ``results/predictions/``. Colored ``.ply`` export is added by the
visualization module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import find_index_by_shape_id, load_dataset, load_sample
from ..openad_bridge import build_model, load_config
from ..paths import DEFAULT_CHECKPOINT, PREDICTIONS_DIR, REF_OPENAD, ensure_dir
from ..prompts import load_vocabulary
from .predictor import AffordancePredictor

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="OpenAD config file")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="path to a model checkpoint"
    )
    parser.add_argument("--data-root", default=None, help="override dataset directory")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=None, help="dataset index")
    parser.add_argument("--shape-id", default=None, help="dataset shape id")
    parser.add_argument("--query", required=True, help="natural-language affordance query")
    parser.add_argument("--background", default="none", help="background/contrast query")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(PREDICTIONS_DIR))
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("no checkpoint given (pass --checkpoint or set OPENAD_CHECKPOINT)")
    if args.index is None and args.shape_id is None:
        parser.error("select a sample with --index or --shape-id")
    return args


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, data_root=args.data_root)
    model = build_model(cfg, checkpoint_path=args.checkpoint, device=args.device)
    predictor = AffordancePredictor(model, load_vocabulary(), device=args.device)

    dataset = load_dataset(args.split, data_root=args.data_root)
    index = args.index if args.index is not None else find_index_by_shape_id(dataset, args.shape_id)
    sample = load_sample(dataset, index)

    scores = predictor.predict_affordance(sample.points, args.query, args.background)

    out_dir = ensure_dir(Path(args.output_dir))
    stem = f"{sample.shape_id}_{_slug(args.query)}"
    np.savez(
        out_dir / f"{stem}.npz",
        points=sample.points,
        labels=sample.labels,
        scores=scores.astype(np.float32),
    )
    meta = {
        "shape_id": sample.shape_id,
        "category": sample.category,
        "split": args.split,
        "index": index,
        "query": args.query,
        "background": args.background,
        "checkpoint": str(args.checkpoint),
        "num_points": int(sample.points.shape[0]),
        "positive_fraction": float((scores >= 0.5).mean()),
        "score_mean": float(scores.mean()),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[{sample.category} / {sample.shape_id}] query={args.query!r}")
    print(f"  positive fraction (>=0.5): {meta['positive_fraction']:.3f}")
    print(f"  saved: {out_dir / f'{stem}.npz'}")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


if __name__ == "__main__":
    main()
