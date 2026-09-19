"""Central project paths.

All locations are derived from the repository root so nothing depends on a
personal absolute path. Data and checkpoint locations can be overridden with
environment variables (useful on Kaggle/Colab):

    OPENAD_DATA        directory holding the downloaded dataset .pkl files
    OPENAD_CHECKPOINT  path to a pretrained/finetuned checkpoint
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (src/paths.py -> src -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Upstream OpenAD checkout (gitignored; created by scripts/setup_openad).
REF_OPENAD = PROJECT_ROOT / "_ref_openad"

# Large assets (never committed).
DATA_DIR = Path(os.environ.get("OPENAD_DATA", PROJECT_ROOT / "data"))
DEFAULT_CHECKPOINT = os.environ.get("OPENAD_CHECKPOINT")  # may be None

# Version-controlled configuration.
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = CONFIGS_DIR / "prompts"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"

# Generated outputs (gitignored contents, kept dirs).
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
