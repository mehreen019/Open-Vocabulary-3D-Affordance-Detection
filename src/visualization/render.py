"""Turn a saved prediction (.npz from run_inference) into a colored .ply and figure.

Example:
    python -m src.visualization.render \
        --prediction results/predictions/Mug_12345_grasp.npz \
        --gt-affordance grasp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..paths import FIGURES_DIR, PREDICTIONS_DIR, ensure_dir
from ..prompts import load_vocabulary
from .pointcloud import render_comparison, render_scatter, save_colored_ply, score_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="path to a prediction .npz")
    parser.add_argument(
        "--gt-affordance", default=None,
        help="canonical affordance name to also render as ground truth (optional)",
    )
    parser.add_argument("--ply-dir", default=str(PREDICTIONS_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.prediction)
    data = np.load(npz_path)
    points, scores = data["points"], data["scores"]
    labels = data["labels"] if "labels" in data.files else None

    meta_path = npz_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    query = meta.get("query", npz_path.stem)
    stem = npz_path.stem

    # Colored point cloud (.ply) of the predicted heatmap.
    ply_path = save_colored_ply(points, score_to_rgb(scores), Path(args.ply_dir) / f"{stem}.ply")

    # Figure: prediction alone, or prediction next to ground truth.
    panels: list[tuple[str, np.ndarray]] = [(f"prediction: {query}", scores)]
    if args.gt_affordance and labels is not None:
        gt_index = load_vocabulary().index(args.gt_affordance)
        gt_mask = (labels == gt_index).astype(np.float32)
        panels.insert(0, (f"ground truth: {args.gt_affordance}", gt_mask))

    figure_dir = ensure_dir(Path(args.figure_dir))
    fig_path = figure_dir / f"{stem}.png"
    if len(panels) == 1:
        render_scatter(points, scores, title=panels[0][0], path=fig_path)
    else:
        render_comparison(points, panels, path=fig_path)

    print(f"saved ply:    {ply_path}")
    print(f"saved figure: {fig_path}")


if __name__ == "__main__":
    main()
