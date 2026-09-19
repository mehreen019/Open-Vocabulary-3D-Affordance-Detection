"""Reproducible inference on top of a pretrained/finetuned OpenAD model."""

from .predictor import AffordancePredictor, normalize_points

__all__ = ["AffordancePredictor", "normalize_points"]
