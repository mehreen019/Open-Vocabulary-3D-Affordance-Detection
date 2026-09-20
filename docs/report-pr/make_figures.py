"""Build the report figures from the saved metric files in ``data/``.

Run from ``docs/report-pr``:

    python make_figures.py

Inputs (copied from the experiment outputs, see README of this folder):
    data/headfp1/comparison_miou.csv, data/headfp1/heldout_affordance_iou.csv,
    data/canonical_per_class_iou.csv, data/openad_synonyms_per_class_iou.csv
    figures/raw/*.png  (before/after renderings from src.visualization.qualitative)

Outputs (in ``figures/``): fig_results.pdf, fig_perclass.pdf,
fig_qual_a.png, fig_qual_b.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
RAW = FIG / "raw"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

# Two-series palette (colour-blind safe): muted blue for pretrained, orange for fine-tuned.
C_PRE = "#6d8fb3"
C_FT = "#d9822b"
C_CANON = "#6d8fb3"
C_SYN = "#b0503f"


def _label_bars(ax, bars, fmt="{:.1f}", dy=0.6, rotation=0):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=5.6,
            rotation=rotation,
        )


def fig_results() -> None:
    """(a) mIoU per phrasing condition; (b, c) per-affordance IoU on held-out phrasing."""
    cmp_ = pd.read_csv(DATA / "headfp1" / "comparison_miou.csv").set_index("condition")
    held = pd.read_csv(DATA / "headfp1" / "heldout_affordance_iou.csv")

    fig = plt.figure(figsize=(7.16, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.26)

    # (a) mIoU by condition
    ax = fig.add_subplot(gs[0])
    labels = [
        "Canonical\nlabels",
        "Question\n(train)",
        "Question\n(held-out)",
        "Descr.\n(train)",
        "Descr.\n(held-out)",
    ]
    x = np.arange(len(cmp_))
    w = 0.38
    b1 = ax.bar(x - w / 2, cmp_["pretrained"], w, color=C_PRE, label="Pretrained OpenAD")
    b2 = ax.bar(x + w / 2, cmp_["finetuned"], w, color=C_FT, label="Prompt-augmented fine-tune (head + fp1)")
    _label_bars(ax, b1, rotation=90)
    _label_bars(ax, b2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(0, 54)
    ax.set_ylabel("mIoU (%), 19 classes")
    ax.set_title("(a) Overall mIoU by phrasing condition", loc="left")
    fig.legend(
        *ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )

    # (b), (c) per-affordance held-out IoU
    order = ["grasp", "contain", "sittable", "pourable", "cut"]
    for k, (cond, title) in enumerate(
        [("question_heldout", "(b) Held-out question phrasing"),
         ("description_heldout", "(c) Held-out description phrasing")]
    ):
        ax = fig.add_subplot(gs[k + 1])
        sub = held[held["condition"] == cond].set_index("affordance").loc[order]
        x = np.arange(len(order))
        bb1 = ax.bar(x - w / 2, sub["pretrained"] * 100, w, color=C_PRE)
        bb2 = ax.bar(x + w / 2, sub["finetuned"] * 100, w, color=C_FT)
        _label_bars(ax, bb1, "{:.0f}", 1.2)
        _label_bars(ax, bb2, "{:.0f}", 1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylim(0, 72)
        ax.set_title(title, loc="left")
        if k == 0:
            ax.set_ylabel("IoU of the rephrased class (%)")

    fig.savefig(FIG / "fig_results.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_perclass() -> None:
    """Per-class IoU, canonical words versus OpenAD's synonym queries."""
    can = pd.read_csv(DATA / "canonical_per_class_iou.csv")
    syn = pd.read_csv(DATA / "openad_synonyms_per_class_iou.csv")
    df = can.merge(syn, on="class", suffixes=("_can", "_syn"))
    df["class"] = df["class"].replace({"displaY": "display"})

    fig, ax = plt.subplots(figsize=(7.16, 2.45))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w / 2, df["iou_can"] * 100, w, color=C_CANON, label="Canonical words (in-vocabulary)")
    ax.bar(x + w / 2, df["iou_syn"] * 100, w, color=C_SYN, label="OpenAD synonyms (open-vocabulary)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c} ({q})" for c, q in zip(df["class"], df["query_syn"])],
        fontsize=6.2,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Per-class IoU (%)")
    ax.set_ylim(0, 68)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.savefig(FIG / "fig_perclass.pdf", bbox_inches="tight")
    plt.close(fig)


def _crop_panels(path: Path) -> Image.Image:
    """Keep the three panel titles and point clouds; drop the suptitle and colour bar."""
    im = Image.open(path).convert("RGB").crop((30, 112, 1170, 470))
    arr = np.asarray(im).astype(int)
    ink = (arr.sum(axis=2) < 3 * 245)
    ys, xs = np.where(ink)
    pad = 6
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    return im.crop(box)


def _qual_figure(cases: list[tuple[str, str, str]], out: Path) -> None:
    """``cases`` = (raw filename prefix, row label, query text)."""
    crops = [_crop_panels(next(RAW.glob(f"{prefix}_*.png"))) for prefix, _, _ in cases]
    width_in = 5.9
    ratios = [c.height / c.width for c in crops]
    heights = [width_in * r for r in ratios]
    gap = 0.30
    fig_h = sum(heights) + gap * len(cases)
    fig = plt.figure(figsize=(7.16, fig_h))
    gs = fig.add_gridspec(
        len(cases), 2, width_ratios=[width_in, 0.16], height_ratios=heights,
        wspace=0.03, hspace=gap / (sum(heights) / len(cases)),
    )
    for i, ((prefix, label, query), crop) in enumerate(zip(cases, crops)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(crop, interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.0, 1.03, label, transform=ax.transAxes, fontsize=7.6,
                fontweight="bold", ha="left", va="bottom")
        ax.text(1.0, 1.03, f"held-out query: “{query}”", transform=ax.transAxes,
                fontsize=7.4, style="italic", ha="right", va="bottom")
    # Shared colour bar (same 0..1 viridis scale used by every panel).
    cax = fig.add_subplot(gs[:, 1])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("affordance score (ground truth is 0 or 1)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_qualitative() -> None:
    _qual_figure(
        [
            ("chair_sittable", "Chair, sittable", "Where would a person sit down here?"),
            ("bowl_contain", "Bowl, contain", "Where does it keep things inside?"),
            ("knife_cut", "Knife, cut", "Where is the part that cuts things?"),
        ],
        FIG / "fig_qual_a.png",
    )
    _qual_figure(
        [
            ("mug_pourable", "Mug, pourable", "Where does the liquid come out when tipped?"),
            ("bottle_grasp", "Bottle, grasp", "Where should I take hold of this?"),
            ("bag_grasp", "Bag, grasp", "Where should I take hold of this?"),
        ],
        FIG / "fig_qual_b.png",
    )


if __name__ == "__main__":
    fig_results()
    fig_perclass()
    fig_qualitative()
    print("figures written to", FIG)
