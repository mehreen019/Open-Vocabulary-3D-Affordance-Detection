"""Load individual point-cloud samples with raw coordinates and labels.

OpenAD's ``AffordNetDataset`` normalizes coordinates and returns tensors ready for
the model. For inference and visualization we also want the *raw* coordinates and
per-point labels plus metadata, so these helpers read them straight from the
loaded records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..openad_bridge import ensure_openad_importable
from ..paths import DATA_DIR


@dataclass
class Sample:
    """One object: raw points, per-point labels, and identifiers."""

    shape_id: str
    category: str
    points: np.ndarray  # (N, 3) raw coordinates
    labels: np.ndarray  # (N,) integer per-point class index


def load_dataset(split: str = "val", data_root: str | Path | None = None, partial: bool = False):
    """Return an OpenAD ``AffordNetDataset`` for the given split."""
    ensure_openad_importable()
    from dataset.AffordanceNet import AffordNetDataset

    root = str(Path(data_root) if data_root else DATA_DIR)
    return AffordNetDataset(root, split, partial=partial)


def load_sample(dataset, index: int) -> Sample:
    """Extract raw points, labels, and metadata for one dataset record."""
    record = dataset.all_data[index]
    info = record["data_info"]
    points = np.asarray(info["coordinate"], dtype=np.float32)
    labels = np.asarray(info["label"]).reshape(-1).astype(np.int64)
    return Sample(
        shape_id=str(record["shape_id"]),
        category=str(record["semantic class"]),
        points=points,
        labels=labels,
    )


def find_index_by_shape_id(dataset, shape_id: str) -> int:
    """Return the dataset index for a given ``shape_id`` (raises if not found)."""
    for i, record in enumerate(dataset.all_data):
        if str(record["shape_id"]) == str(shape_id):
            return i
    raise KeyError(f"shape_id {shape_id!r} not found in dataset")
