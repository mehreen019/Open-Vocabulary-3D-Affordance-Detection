"""Render before/after qualitative figures for curated (object, affordance) cases.

For each case we use a *held-out* natural-phrasing query (one the fine-tune never
saw) and draw three panels with a shared color scale:

    ground truth | pretrained heatmap | fine-tuned heatmap

This makes the collapse-under-natural-phrasing (pretrained) and its recovery
(fine-tuned) visible at a glance. Figures go to results/figures/qualitative/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..openad_bridge import build_model, load_config
from ..paths import FIGURES_DIR, REF_OPENAD, ensure_dir
from ..prompts import load_prompt_sets, load_vocabulary
from .pointcloud import render_comparison

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_open_vocab_cfg.py"

#: Curated (object category, studied affordance) cases spanning success and rarity.
DEFAULT_CASES: tuple[tuple[str, str], ...] = (
    ("Mug", "pourable"),
    ("Chair", "sittable"),
    ("Knife", "cut"),
    ("Bottle", "grasp"),
    ("Bowl", "contain"),
    ("Bag", "grasp"),
)


def find_sample_with_affordance(dataset, category: str, affordance_index: int):
    """First record of ``category`` whose ground truth contains the affordance."""
    from ..data import load_sample

    for i, record in enumerate(dataset.all_data):
        if str(record["semantic class"]).lower() != category.lower():
            continue
        sample = load_sample(dataset, i)
        if np.any(sample.labels == affordance_index):
            return sample
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(FIGURES_DIR / "qualitative"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocabulary = load_vocabulary()
    prompt_sets = load_prompt_sets(vocabulary=vocabulary)

    from ..data import load_dataset
    from ..inference.predictor import AffordancePredictor

    cfg = load_config(args.config, data_root=args.data_root)
    predictors = {
        "pretrained": AffordancePredictor(build_model(cfg, args.pretrained, args.device), vocabulary, args.device),
        "finetuned": AffordancePredictor(build_model(cfg, args.finetuned, args.device), vocabulary, args.device),
    }
    dataset = load_dataset(args.split, data_root=args.data_root)
    out_dir = ensure_dir(Path(args.output_dir))

    for category, affordance in DEFAULT_CASES:
        # Use a held-out question phrasing so the figure shows unseen-phrasing behavior.
        heldout = prompt_sets.phrases(affordance, "heldout", "question")
        query = heldout[0] if heldout else affordance
        index = vocabulary.index(affordance)

        sample = find_sample_with_affordance(dataset, category, index)
        if sample is None:
            print(f"skip {category}/{affordance}: no sample with that affordance")
            continue

        gt = (sample.labels == index).astype(np.float32)
        panels = [
            (f"ground truth ({affordance})", gt),
            ("pretrained", predictors["pretrained"].predict_affordance(sample.points, query)),
            ("fine-tuned", predictors["finetuned"].predict_affordance(sample.points, query)),
        ]
        path = out_dir / f"{category.lower()}_{affordance}_{sample.shape_id}.png"
        render_comparison(
            sample.points, panels, path=path, suptitle=f"{category} — held-out query: “{query}”"
        )
        print(f"saved {path.name}")

    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()
