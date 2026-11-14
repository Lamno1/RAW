"""Losses with explicit absent/ignored-point behavior."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StaticBalancedCrossEntropy(nn.Module):
    """Static train-set-weighted multiclass cross entropy.

    An all-ignore batch is a data error rather than a zero-loss update.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        ignore_index: int = -100,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1 or not torch.isfinite(weights).all():
            raise ValueError("class_weights must be a finite rank-1 tensor")
        if torch.any(weights <= 0):
            raise ValueError("all class weights must be positive")
        self.register_buffer("class_weights", weights)
        self.ignore_index = int(ignore_index)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != target.shape:
            raise ValueError("logits and target dimensions are incompatible")
        valid = target != self.ignore_index
        if not torch.any(valid):
            raise ValueError("batch contains no valid segmentation labels")
        valid_logits = logits[valid].float()
        valid_target = target[valid]
        if valid_target.min() < 0 or valid_target.max() >= logits.shape[-1]:
            raise ValueError("target contains an out-of-range class index")
        point_loss = F.cross_entropy(valid_logits, valid_target, reduction="none")
        point_weight = self.class_weights[valid_target]
        denominator = point_weight.sum().clamp_min(self.epsilon)
        return (point_loss * point_weight).sum() / denominator


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss."""
    if sorted_foreground.ndim != 1 or sorted_foreground.numel() == 0:
        raise ValueError("sorted_foreground must be a non-empty vector")
    count = sorted_foreground.numel()
    total_positive = sorted_foreground.sum()
    intersection = total_positive - sorted_foreground.cumsum(0)
    union = total_positive + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-6)
    if count > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass Lovasz-Softmax with explicit absent/ignore behavior.

    Logits have shape ``[..., C]`` and labels have the matching leading shape.
    Sorting and reductions are always evaluated in FP32. Classes absent from
    the valid target are skipped, matching the frozen S2 protocol.
    """

    def __init__(self, *, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != target.shape:
            raise ValueError("logits and target dimensions are incompatible")
        if logits.shape[-1] < 2:
            raise ValueError("Lovasz-Softmax requires at least two classes")
        valid = target != self.ignore_index
        if not torch.any(valid):
            raise ValueError("batch contains no valid segmentation labels")
        labels = target[valid].long()
        probabilities = F.softmax(logits[valid].float(), dim=-1)
        if int(labels.min()) < 0 or int(labels.max()) >= probabilities.shape[-1]:
            raise ValueError("target contains an out-of-range class index")

        losses: list[torch.Tensor] = []
        for class_index in range(probabilities.shape[-1]):
            foreground = (labels == class_index).float()
            if not torch.any(foreground):
                continue
            errors = (foreground - probabilities[:, class_index]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            sorted_foreground = foreground[permutation]
            losses.append(
                torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground))
            )
        if not losses:
            raise ValueError("batch contains no present class for Lovasz loss")
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Lovasz-Softmax produced NaN or Inf")
        return loss


class BalancedCEWithLovasz(nn.Module):
    """Frozen-weight sum used only after the isolated S2 calibration."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        lovasz_weight: float,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if lovasz_weight < 0.0:
            raise ValueError("lovasz_weight must be non-negative")
        self.balanced_ce = StaticBalancedCrossEntropy(
            class_weights, ignore_index=ignore_index
        )
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.balanced_ce(logits, target)
        if self.lovasz_weight == 0.0:
            return ce
        return ce + self.lovasz_weight * self.lovasz(logits, target)
