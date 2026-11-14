"""S2 Lovasz-Softmax behavioral tests."""
from __future__ import annotations

import pytest
import torch

from camo_build.losses import LovaszSoftmaxLoss


def test_lovasz_prefers_correct_logits_and_has_finite_gradient() -> None:
    target = torch.tensor([[0, 1, 2, 1]])
    correct = torch.tensor(
        [[[8.0, -2.0, -2.0], [-2.0, 8.0, -2.0], [-2.0, -2.0, 8.0], [-2.0, 8.0, -2.0]]],
        requires_grad=True,
    )
    wrong = (-correct.detach()).requires_grad_(True)
    loss_fn = LovaszSoftmaxLoss()
    correct_loss = loss_fn(correct, target)
    wrong_loss = loss_fn(wrong, target)
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert correct.grad is not None
    assert torch.isfinite(correct.grad).all()


def test_lovasz_skips_absent_classes_and_ignore_points() -> None:
    logits = torch.tensor([[[4.0, -1.0, -2.0], [-1.0, 4.0, -2.0], [0.0, 0.0, 0.0]]])
    target = torch.tensor([[0, 1, -100]])
    loss = LovaszSoftmaxLoss(ignore_index=-100)(logits, target)
    assert torch.isfinite(loss)


def test_lovasz_rejects_all_ignore_batch() -> None:
    with pytest.raises(ValueError, match="no valid"):
        LovaszSoftmaxLoss()(torch.randn(1, 4, 3), torch.full((1, 4), -100))

