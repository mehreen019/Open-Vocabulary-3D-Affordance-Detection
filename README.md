# Open-Vocabulary 3D Affordance Detection

An interactive system that takes a 3D object point cloud and a natural-language
query (e.g. `grasp`, `Where should I hold this?`) and highlights the object points
that support that action. It builds on the pretrained
[OpenAD](https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds)
model (PointNet++ point encoder + frozen CLIP text encoder + cosine-similarity
alignment).

**What this project does**
- Reproduces and evaluates the pretrained OpenAD model on 3D AffordanceNet.
- Studies how prompt phrasing affects predictions and applies a lightweight
  **prompt-augmented fine-tune** to improve robustness to natural phrasing.
- Compares the original and fine-tuned models quantitatively and qualitatively.
- Provides an interactive point-cloud interface to query affordances and inspect
  the predicted heatmap and confidence.

> Uses the official pretrained OpenAD weights. This is an implementation,
> evaluation, and interaction study — not a new architecture or new
> state-of-the-art weights.

## Attribution

OpenAD is **not** vendored here. It is cloned into `_ref_openad/` (gitignored) at a
pinned commit by the setup script and imported through `src/openad_bridge.py`. Our
code (in `src/`) adds the reproducible inference/evaluation wrappers, the prompt
experiments, the prompt-augmented fine-tuning, visualization, and the interactive
app.

## Setup

```bash
# 1. Clone the OpenAD reference code (pinned commit)
bash scripts/setup_openad.sh            # or: pwsh scripts/setup_openad.ps1

# 2. Create an environment (Python 3.9/3.10) and install PyTorch first, matching
#    your GPU (e.g. CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

### Data and checkpoint (not committed)

Download from OpenAD's official Google Drive (links in `_ref_openad/README.md`):

- **Dataset** — place the `full_shape_*_data.pkl` files (and `full_shape_weights.npy`)
  in `data/`, or set `OPENAD_DATA=/path/to/data`.
- **Pretrained checkpoint** — pass it with `--checkpoint` or set
  `OPENAD_CHECKPOINT=/path/to/checkpoint`.

## Usage

Run from the repo root; every module is `python -m src.<...>` and supports `--help`.

```bash
# Evaluate the pretrained checkpoint
python -m src.evaluation.run_eval --checkpoint <ckpt> --condition canonical
python -m src.evaluation.run_eval --checkpoint <ckpt> --condition openad_synonyms

# Prompt-sensitivity experiment
python -m src.evaluation.run_prompt_sensitivity --checkpoint <ckpt> --num-samples 40

# Prompt-augmented fine-tuning
python -m src.training.finetune --checkpoint <ckpt> --group head_fp1 --epochs 8 --num-workers 4 --tag finetuned

# Before/after comparison
python -m src.evaluation.compare_models --pretrained <ckpt> --finetuned results/checkpoints/finetuned_best.pth --num-workers 4

# Single-object inference and visualization
python -m src.inference.run_inference --checkpoint <ckpt> --split val --index 0 --query "grasp"
python -m src.visualization.render --prediction results/predictions/<name>.npz --gt-affordance grasp

# Interactive demo
python -m src.app.demo --checkpoint <ckpt>
```

Outputs are written under `results/` (metrics, figures, predictions, checkpoints).
Run the tests with `python -m pytest`.

## Repository layout

```
src/          data, inference, training, evaluation, visualization, app
configs/      prompt sets and experiment definitions
scripts/      setup helpers (clone OpenAD, etc.)
tests/        unit tests
results/      generated metrics, figures, predictions, checkpoints (gitignored)
_ref_openad/  OpenAD upstream (gitignored; cloned by scripts/setup_openad)
```

## Citations and licenses

- OpenAD (IROS 2023), Nguyen et al. — MIT License. arXiv:2303.02401.
- 3D AffordanceNet (CVPR 2021). PointNet++ (Qi et al., 2017). CLIP (Radford et al., 2021).

This repository's own code is released under the MIT License (see `LICENSE`).
