"""Tests for the fine-tuning freeze/unfreeze logic (no torch/GPU needed)."""

import pytest

from src.training.finetune import TRAINABLE_GROUPS, set_trainable


class _FakeParam:
    def __init__(self, n: int):
        self._n = n
        self.requires_grad = True

    def numel(self) -> int:
        return self._n


class _FakeModel:
    """Mimics OpenAD_PN2's top-level parameter names."""

    def __init__(self):
        self._params = {
            "sa1.mlp.0.weight": _FakeParam(100),
            "fp1.mlp.0.weight": _FakeParam(40),
            "conv1.weight": _FakeParam(50),
            "bn1.weight": _FakeParam(30),
            "logit_scale": _FakeParam(1),
        }

    def named_parameters(self):
        return self._params.items()


def test_head_group_trains_only_alignment_head():
    model = _FakeModel()
    trainable = set_trainable(model, "head")
    assert trainable == 50 + 30 + 1  # conv1 + bn1 + logit_scale
    assert model._params["sa1.mlp.0.weight"].requires_grad is False
    assert model._params["fp1.mlp.0.weight"].requires_grad is False
    assert model._params["conv1.weight"].requires_grad is True


def test_head_fp1_group_also_trains_last_feature_layer():
    model = _FakeModel()
    trainable = set_trainable(model, "head_fp1")
    assert trainable == 50 + 30 + 1 + 40
    assert model._params["fp1.mlp.0.weight"].requires_grad is True
    assert model._params["sa1.mlp.0.weight"].requires_grad is False


def test_all_group_trains_everything():
    model = _FakeModel()
    trainable = set_trainable(model, "all")
    assert trainable == 100 + 40 + 50 + 30 + 1
    assert all(p.requires_grad for p in model._params.values())


def test_unknown_group_raises():
    with pytest.raises(ValueError):
        set_trainable(_FakeModel(), "does-not-exist")


def test_groups_are_documented():
    assert set(TRAINABLE_GROUPS) == {"head", "head_fp1", "all"}
