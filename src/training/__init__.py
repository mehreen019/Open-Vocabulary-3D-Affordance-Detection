"""Prompt-augmented fine-tuning of the pretrained OpenAD model."""

from .finetune import TRAINABLE_GROUPS, set_trainable

__all__ = ["TRAINABLE_GROUPS", "set_trainable"]
