"""Color point clouds by affordance score and render figures.

A fixed score range (0..1) is used everywhere so colors are comparable across
figures — a requirement for the qualitative analysis. Heavy imports (matplotlib,
trimesh) are done lazily so importing this module never requires them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CMAP = "viridis"


def score_to_rgb(
    scores: np.ndarray, cmap: str = DEFAULT_CMAP, vmin: float = 0.0, vmax: float = 1.0
) -> np.ndarray:
    """Map per-point scores to ``(N, 3)`` uint8 RGB using a fixed color scale."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    normed = Normalize(vmin=vmin, vmax=vmax, clip=True)(np.asarray(scores, dtype=np.float32))
    rgba = colormaps[cmap](normed)
    return (rgba[:, :3] * 255).astype(np.uint8)


def save_colored_ply(points: np.ndarray, colors: np.ndarray, path: str | Path) -> Path:
    """Write a colored point cloud to a ``.ply`` file."""
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(vertices=np.asarray(points), colors=np.asarray(colors))
    cloud.export(str(path))
    return path


def _scatter(ax, points: np.ndarray, scores: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    points = np.asarray(points)
    handle = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.asarray(scores), cmap=cmap, vmin=vmin, vmax=vmax, s=3, linewidths=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    return handle


def render_scatter(
    points: np.ndarray,
    scores: np.ndarray,
    title: str | None = None,
    path: str | Path | None = None,
    cmap: str = DEFAULT_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Path | None:
    """Render a single score heatmap over a point cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    handle = _scatter(ax, points, scores, title or "", cmap, vmin, vmax)
    fig.colorbar(handle, ax=ax, shrink=0.6, pad=0.0)
    return _save_or_show(fig, path)


def render_comparison(
    points: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    path: str | Path | None = None,
    cmap: str = DEFAULT_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    suptitle: str | None = None,
) -> Path | None:
    """Render several score maps of the same object side by side (shared scale).

    ``panels`` is a list of ``(title, scores)`` — e.g. ground truth, pretrained,
    fine-tuned, or different prompt phrasings.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4 * len(panels), 4.2))
    handle = None
    for i, (title, scores) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), i, projection="3d")
        handle = _scatter(ax, points, scores, title, cmap, vmin, vmax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    if handle is not None:
        fig.colorbar(handle, ax=fig.axes, shrink=0.6, pad=0.02)
    return _save_or_show(fig, path)


def _save_or_show(fig, path: str | Path | None) -> Path | None:
    import matplotlib.pyplot as plt

    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
