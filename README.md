# Interactive Open-Vocabulary 3D Affordance Detection

A system that takes a 3D object point cloud and a natural-language query
(e.g. `grasp`, `Where should I hold this?`) and highlights the object points that
support that action. One system, two course deliverables:

- **Pattern Recognition (PR):** reproduce and evaluate the pretrained
  [OpenAD](https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds)
  model, apply a lightweight **prompt-augmented fine-tune** to improve robustness
  to natural phrasing, and compare the two quantitatively and qualitatively.
- **Human-Computer Interaction (HCI):** an interactive, human-centered interface
  around the model for inspecting, questioning, and refining predictions.

> We use the official pretrained OpenAD weights. We do **not** claim a new
> architecture or a new state-of-the-art result. See `docs/plans/project-plan.md`.

## What is ours vs. OpenAD's

| OpenAD (referenced, MIT) | This repository (our work) |
|---|---|
| PointNet++ + frozen CLIP model, training/eval entry points, pretrained weights | Reproducible inference and evaluation wrappers, prompt experiments, prompt-augmented fine-tuning, visualization, the interactive app, and both reports |

OpenAD is **not vendored**. It is cloned into `_ref_openad/` (gitignored) at a
pinned commit by the setup script, and imported through `src/openad_bridge.py`.

## Setup

```bash
# 1. Clone the OpenAD reference code (pinned commit)
bash scripts/setup_openad.sh            # or: pwsh scripts/setup_openad.ps1

# 2. Create an environment (Python 3.10 recommended) and install PyTorch first,
#    matching your GPU. For an RTX 4060 (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

### Data and checkpoint (not committed)

These are downloaded manually from OpenAD's official Google Drive (also linked in
`_ref_openad/README.md`). They are large and stay out of Git.

**1. Dataset** -- https://drive.google.com/drive/folders/1f-_V_iA6POMYlBe2byuplJfdKmV72BHu

Place the full-shape files **directly** inside `data/` (or set
`OPENAD_DATA=/path/to/data`):

```
data/
  full_shape_train_data.pkl
  full_shape_val_data.pkl
  full_shape_weights.npy      # class weights, used by the fine-tuning loss
```

If the download is a zip or a nested folder, unzip so the `.pkl` files sit
directly in `data/` (not `data/full_shape/...`).

**2. Pretrained checkpoint** -- https://drive.google.com/drive/folders/17895vwgGHfIlDj3q0a7BOg6cotH5RTjm

Download the PointNet++ full-shape checkpoint (`.t7` or `.pth`) into
`results/checkpoints/`, or pass it with `--checkpoint` / set
`OPENAD_CHECKPOINT=/path/to/checkpoint`.

> Tip: use the folder's **Download** button (it zips everything) rather than
> grabbing files one by one. For large files, click through Google's
> "can't scan for viruses" warning.

### Verify the setup

```bash
python -m src.evaluation.run_eval --checkpoint results/checkpoints/<name>.t7 --condition canonical
```

An mIoU around 12-15 means the reproduction works. (Do not compare against the
40+ closed-set numbers -- that is a different regime.)

## Usage

Run from the repo root. Every module is `python -m src.<...>` and supports
`--help`. Below, `<ckpt>` is a checkpoint path.

```bash
# 1. Inference: one object + one query -> reusable prediction (.npz + .json)
python -m src.inference.run_inference --checkpoint <ckpt> --split val --index 0 --query "grasp"

# 2. Visualize a saved prediction -> colored .ply + figure (optional ground truth)
python -m src.visualization.render --prediction results/predictions/<name>.npz --gt-affordance grasp

# 3. Evaluate a checkpoint under one phrasing condition (mIoU, per-class IoU)
python -m src.evaluation.run_eval --checkpoint <ckpt> --condition canonical
python -m src.evaluation.run_eval --checkpoint <ckpt> --condition openad_synonyms   # reproduce OpenAD

# 4. Prompt-sensitivity experiment (accuracy by phrasing + heatmap consistency)
python -m src.evaluation.run_prompt_sensitivity --checkpoint <ckpt> --num-samples 40

# 5. Prompt-augmented fine-tuning from the pretrained checkpoint (our modification)
python -m src.training.finetune --checkpoint <pretrained> --group head --epochs 5 --tag finetuned

# 6. Fair before/after comparison across phrasing conditions (main PR result)
python -m src.evaluation.compare_models --pretrained <pretrained> \
    --finetuned results/checkpoints/finetuned_best.pth

# 7. Interactive demo (PR demonstration + HCI prototype)
python -m src.app.demo --checkpoint <pretrained> \
    --finetuned results/checkpoints/finetuned_best.pth
```

Outputs land under `results/` (`metrics/`, `figures/`, `predictions/`,
`checkpoints/`). Frozen prompt sets live in `configs/prompts/`. Run the tests with
`python -m pytest`.

## Repository layout

```
src/          our code (data, inference, training, evaluation, visualization, app)
configs/      frozen prompt sets and experiment definitions
scripts/      setup helpers (clone OpenAD, etc.)
tests/        unit tests
results/      generated metrics, figures, predictions, checkpoints (gitignored)
docs/         plan, PR report, HCI report, slides
demo/         demo assets / recorded outputs
_ref_openad/  OpenAD upstream (gitignored; cloned by scripts/setup_openad)
```

## Citations and licenses

- OpenAD (IROS 2023), Nguyen et al. -- MIT License. arXiv:2303.02401.
- 3D AffordanceNet (CVPR 2021). PointNet++. CLIP (Radford et al., 2021).

This repository's own code is released under the MIT License (see `LICENSE`).
