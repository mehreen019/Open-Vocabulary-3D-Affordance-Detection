"""Prompt-augmented fine-tuning of pretrained OpenAD (plan Step 8).

The modification: instead of supervising each affordance with a single fixed word,
each training step samples a paraphrase for the studied affordances (the spatial
target is unchanged, since equivalent phrases describe the same region). This adapts
the language supervision only, aiming for more consistent predictions across natural
phrasing — while the CLIP text encoder stays frozen and we start from the released
checkpoint.

Design choices (all defensible for the report):
* Start from pretrained weights; never train from scratch.
* Freeze the frozen-by-design CLIP encoder; optionally freeze the PointNet++
  backbone and train only the alignment head (see ``TRAINABLE_GROUPS``).
* Low learning rate and a small epoch budget, to adapt without disrupting
  pretrained features.
* Select the checkpoint by validation mIoU on canonical labels, so held-out
  phrasings stay a true test (no selection leakage).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ..openad_bridge import build_model, load_config
from ..paths import CHECKPOINTS_DIR, DEFAULT_CHECKPOINT, METRICS_DIR, REF_OPENAD, ensure_dir
from ..prompts import load_prompt_sets, load_vocabulary, sample_training_vocab

DEFAULT_CONFIG = REF_OPENAD / "config" / "openad_pn2" / "full_shape_cfg.py"

#: Which parameters to train, keyed by top-level module name. ``None`` = all.
#: (The CLIP encoder is a module-level global, so it never appears here and stays frozen.)
TRAINABLE_GROUPS: dict[str, tuple[str, ...] | None] = {
    "head": ("conv1", "bn1", "logit_scale"),  # alignment head only (~67k)
    "head_fp1": ("conv1", "bn1", "logit_scale", "fp1"),  # + last feature-propagation (~101k)
    "head_fp1_fp2": ("conv1", "bn1", "logit_scale", "fp1", "fp2"),  # ~283k
    "decoder": ("conv1", "bn1", "logit_scale", "fp1", "fp2", "fp3"),  # whole decoder, encoder frozen (~743k)
    "all": None,  # decoder + geometric encoder (sa1/sa2/sa3) (~1.78M)
}


def set_trainable(model, group: str) -> int:
    """Freeze/unfreeze parameters for a group; return the trainable parameter count."""
    if group not in TRAINABLE_GROUPS:
        raise ValueError(f"unknown group {group!r}; expected {list(TRAINABLE_GROUPS)}")
    prefixes = TRAINABLE_GROUPS[group]
    trainable = 0
    for name, param in model.named_parameters():
        keep = prefixes is None or name.split(".")[0] in prefixes
        param.requires_grad = keep
        if keep:
            trainable += param.numel()
    return trainable


def finetune(
    cfg,
    checkpoint_path: str | Path,
    *,
    group: str = "head",
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    num_workers: int = 0,
    augment: bool = True,
    seed: int = 1,
    device: str = "cuda",
    output_dir: Path = CHECKPOINTS_DIR,
    tag: str = "finetuned",
) -> dict:
    """Fine-tune from a checkpoint and return the training record."""
    import torch
    from torch.utils.data import DataLoader

    from ..evaluation.evaluate import evaluate_model

    random.seed(seed)
    torch.manual_seed(seed)

    vocabulary = load_vocabulary()
    prompt_sets = load_prompt_sets(vocabulary=vocabulary)
    canonical = list(vocabulary.canonical)
    rng = random.Random(seed)

    model = build_model(cfg, checkpoint_path=checkpoint_path, device=device)
    trainable = set_trainable(model, group)

    from utils import build_dataset, build_loss  # lazy: needs OpenAD + GPU

    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )

    datasets = build_dataset(cfg)
    train_loader = DataLoader(
        datasets["train_set"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    val_set = datasets["val_set"]

    out_dir = ensure_dir(Path(output_dir))
    record = {
        "tag": tag,
        "checkpoint_init": str(checkpoint_path),
        "group": group,
        "trainable_parameters": int(trainable),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "augment": augment,
        "seed": seed,
        "history": [],
    }
    best_miou, best_path = -1.0, out_dir / f"{tag}_best.pth"

    for epoch in range(1, epochs + 1):
        # Train everything only for group="all"; otherwise keep BN stats frozen
        # (eval mode) since the backbone is frozen. requires_grad flags are set once
        # by set_trainable and are unaffected by train()/eval().
        model.train() if group == "all" else model.eval()

        running_loss, steps = 0.0, 0
        start = time.time()
        for data, _, target, _, _ in train_loader:
            data = data.float().to(device).permute(0, 2, 1)  # (B, 3, N)
            target = target.reshape(target.shape[0], -1).long().to(device)  # (B, N)
            queries = sample_training_vocab(prompt_sets, rng) if augment else canonical

            pred = model(data, queries)  # (B, K, N) log-softmax
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        val = evaluate_model(
            model, val_set, canonical, condition="val_canonical", class_names=canonical,
            batch_size=batch_size, num_workers=num_workers, device=device, per_sample=False,
        )
        val_miou = val.summary["miou"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "val_miou": val_miou,
            "seconds": round(time.time() - start, 1),
        }
        record["history"].append(epoch_record)
        print(
            f"epoch {epoch}/{epochs}  loss={epoch_record['train_loss']:.4f}  "
            f"val_mIoU={val_miou * 100:.2f}  ({epoch_record['seconds']}s)",
            flush=True,  # show progress live instead of buffering to exit
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_miou": val_miou,
                 "group": group, "augment": augment},
                best_path,
            )

    record["best_val_miou"] = best_miou
    record["best_checkpoint"] = str(best_path)
    ensure_dir(Path(METRICS_DIR))
    (Path(METRICS_DIR) / f"{tag}_training_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"best val mIoU={best_miou * 100:.2f}  ->  {best_path}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="pretrained checkpoint to start from")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--group", default="head", choices=list(TRAINABLE_GROUPS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true", help="control run: canonical labels only")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--tag", default="finetuned")
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("no starting checkpoint (pass --checkpoint or set OPENAD_CHECKPOINT)")
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, data_root=args.data_root)
    finetune(
        cfg,
        args.checkpoint,
        group=args.group,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        seed=args.seed,
        device=args.device,
        output_dir=Path(args.output_dir),
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
