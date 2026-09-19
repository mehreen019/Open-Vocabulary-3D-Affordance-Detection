"""Thin bridge to the OpenAD reference checkout in ``_ref_openad/``.

We never vendor OpenAD; we import it. This module puts the checkout on
``sys.path`` and exposes small, well-typed helpers for loading a config, building
a model, and loading a checkpoint, so the rest of ``src`` never touches OpenAD's
import quirks directly.

GPU note: importing OpenAD's model package instantiates a global CLIP text encoder
on CUDA at import time, so every helper here requires a visible GPU and the CLIP
package. Imports are therefore done lazily, inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from .paths import DATA_DIR, REF_OPENAD


def ensure_openad_importable() -> None:
    """Put the OpenAD checkout on ``sys.path`` (idempotent)."""
    if not (REF_OPENAD / "models").is_dir():
        raise RuntimeError(
            f"OpenAD not found at {REF_OPENAD}. Run scripts/setup_openad first."
        )
    path = str(REF_OPENAD)
    if path not in sys.path:
        sys.path.insert(0, path)


@contextlib.contextmanager
def _working_dir(path: Path):
    """Temporarily change the working directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_config(config_path: str | Path, data_root: str | Path | None = None) -> Any:
    """Load an OpenAD config and repoint its data paths at our locations.

    We import the config module directly from its path rather than using
    ``gorilla.config.Config.fromfile``, which copies the file to an already-open
    ``NamedTemporaryFile`` and fails on Windows. OpenAD config files run
    ``from utils import ...`` and ``os.makedirs(work_dir)`` with paths relative to
    the working directory, so we execute them from inside the checkout (any
    generated ``log/`` stays there, and it is gitignored).
    """
    import importlib.util
    import inspect
    import types

    ensure_openad_importable()
    from gorilla.config import Config

    config_path = Path(config_path).resolve()
    with _working_dir(REF_OPENAD):
        spec = importlib.util.spec_from_file_location("openad_config", str(config_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    # Keep only config values; drop imported modules, functions, and classes.
    cfg_dict = {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("__")
        and not isinstance(value, types.ModuleType)
        and not (inspect.isclass(value) or inspect.isroutine(value))
    }

    data_root = Path(data_root) if data_root else DATA_DIR
    if "data" in cfg_dict:
        cfg_dict["data"]["data_root"] = str(data_root)
    if cfg_dict.get("training_cfg", {}).get("weights_dir"):
        cfg_dict["training_cfg"]["weights_dir"] = str(data_root / "full_shape_weights.npy")

    return Config(cfg_dict)


def load_checkpoint_into(model, checkpoint_path: str | Path) -> None:
    """Load weights into ``model``, handling OpenAD's ``.t7`` and ``.pth`` formats."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint_path.suffix == ".pth" or (
        isinstance(state, dict) and "model_state_dict" in state
    ):
        state = state["model_state_dict"]
    model.load_state_dict(state)


def build_model(
    cfg: Any, checkpoint_path: str | Path | None = None, device: str = "cuda"
):
    """Build an OpenAD model, optionally load a checkpoint, and set eval mode."""
    ensure_openad_importable()
    from utils import build_model as _openad_build_model  # noqa: WPS433 (lazy)

    model = _openad_build_model(cfg).to(device)
    if checkpoint_path is not None:
        load_checkpoint_into(model, checkpoint_path)
    model.eval()
    return model
