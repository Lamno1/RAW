"""Frozen S0 metrics for BuildingNet point segmentation.

The official point track reports mean Part IoU, Shape IoU and per-point
classification accuracy. Extra head/mid/tail and boundary diagnostics are
reported separately and never substituted for the official metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


def full_cloud_chunks(
    num_points: int, chunk_size: int, seed: int
) -> list[tuple[np.ndarray, int]]:
    """Return padded chunks that cover every point exactly once per vote.

    The integer paired with each chunk is the number of non-padding entries.
    Padding repeats the beginning of the same deterministic permutation.
    """
    if num_points <= 0 or chunk_size <= 0:
        raise ValueError("num_points and chunk_size must be positive")
    permutation = np.random.default_rng(seed).permutation(num_points).astype(np.int64)
    chunks: list[tuple[np.ndarray, int]] = []
    for start in range(0, num_points, chunk_size):
        valid = min(chunk_size, num_points - start)
        chunk = permutation[start : start + valid]
        if valid < chunk_size:
            repeats = int(np.ceil((chunk_size - valid) / num_points))
            padding = np.tile(permutation, repeats + 1)[: chunk_size - valid]
            chunk = np.concatenate((chunk, padding))
        chunks.append((chunk, valid))
    return chunks


def boundary_mask(
    labels: torch.Tensor, neighbors: torch.Tensor, *, ignore_index: int = -100
) -> torch.Tensor:
    """Mark valid points adjacent to a different valid label."""
    if labels.ndim != 2 or neighbors.ndim != 3 or labels.shape != neighbors.shape[:2]:
        raise ValueError("labels [B,N] and neighbors [B,N,k] are required")
    batch = torch.arange(labels.shape[0], device=labels.device)[:, None, None]
    neighbor_labels = labels[batch, neighbors.long()]
    valid = labels != ignore_index
    neighbor_valid = neighbor_labels != ignore_index
    different = neighbor_labels != labels.unsqueeze(-1)
    return valid & torch.any(neighbor_valid & different, dim=-1)


def boundary_part_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    neighbors: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = -100,
) -> dict[str, object]:
    """Macro IoU between class-specific graph-boundary point sets.

    This is a CAMO diagnostic, not an official BuildingNet leaderboard metric.
    The graph is fixed from input geometry. A point belongs to a class boundary
    when at least one valid graph neighbor has a different label.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    predicted_boundary = boundary_mask(
        prediction, neighbors, ignore_index=ignore_index
    )
    target_boundary = boundary_mask(target, neighbors, ignore_index=ignore_index)
    valid = target != ignore_index
    per_class: list[float] = []
    for class_index in range(num_classes):
        predicted_set = valid & predicted_boundary & (prediction == class_index)
        target_set = valid & target_boundary & (target == class_index)
        union = torch.count_nonzero(predicted_set | target_set)
        if int(union) == 0:
            per_class.append(float("nan"))
        else:
            intersection = torch.count_nonzero(predicted_set & target_set)
            per_class.append(float(intersection / union))
    finite = [value for value in per_class if np.isfinite(value)]
    return {
        "boundary_part_iou": float(np.mean(finite)) if finite else float("nan"),
        "per_class_boundary_iou": per_class,
    }


def _shape_iou(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> float:
    values: list[float] = []
    for label in range(classes):
        predicted = prediction == label
        expected = target == label
        union = torch.count_nonzero(predicted | expected)
        if int(union) > 0:
            intersection = torch.count_nonzero(predicted & expected)
            values.append(float(intersection / union))
    return float(np.mean(values)) if values else float("nan")


@dataclass
class SegmentationMeter:
    """Accumulate official BuildingNet point metrics without batch bias."""

    num_classes: int
    ignore_index: int = -100
    confusion: torch.Tensor = field(init=False)
    shape_ious: list[float] = field(default_factory=list, init=False)
    shape_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        model_names: Sequence[str] | None = None,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("prediction and target must both have shape [B,N]")
        if model_names is not None and len(model_names) != target.shape[0]:
            raise ValueError("model_names length must equal batch size")
        prediction = prediction.detach().to(device="cpu", dtype=torch.long)
        target = target.detach().to(device="cpu", dtype=torch.long)
        valid = target != self.ignore_index
        if torch.any(valid):
            valid_prediction = prediction[valid]
            valid_target = target[valid]
            if (
                int(valid_target.min()) < 0
                or int(valid_target.max()) >= self.num_classes
                or int(valid_prediction.min()) < 0
                or int(valid_prediction.max()) >= self.num_classes
            ):
                raise ValueError("prediction or target class is out of range")
            encoded = valid_target * self.num_classes + valid_prediction
            self.confusion += torch.bincount(
                encoded, minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)
        for row in range(target.shape[0]):
            row_valid = valid[row]
            if not torch.any(row_valid):
                continue
            self.shape_ious.append(
                _shape_iou(prediction[row, row_valid], target[row, row_valid], self.num_classes)
            )
            self.shape_names.append(
                model_names[row] if model_names is not None else str(len(self.shape_names))
            )

    def compute(
        self, class_groups: dict[str, Sequence[int]] | None = None
    ) -> dict[str, object]:
        confusion = self.confusion.double()
        intersection = confusion.diag()
        target_count = confusion.sum(1)
        prediction_count = confusion.sum(0)
        union = target_count + prediction_count - intersection
        present = union > 0
        per_class = torch.full((self.num_classes,), float("nan"), dtype=torch.double)
        per_class[present] = intersection[present] / union[present]
        total = target_count.sum()
        accuracy = float(intersection.sum() / total) if float(total) > 0 else float("nan")
        result: dict[str, object] = {
            "classification_accuracy": accuracy,
            "part_iou": float(per_class[present].mean()) if torch.any(present) else float("nan"),
            "shape_iou": float(np.nanmean(self.shape_ious)) if self.shape_ious else float("nan"),
            "per_class_iou": per_class.tolist(),
            "confusion": self.confusion.tolist(),
            "num_shapes": len(self.shape_ious),
        }
        if class_groups:
            grouped: dict[str, float] = {}
            for name, indices in class_groups.items():
                selected = per_class[torch.as_tensor(indices, dtype=torch.long)]
                selected = selected[torch.isfinite(selected)]
                grouped[name] = float(selected.mean()) if selected.numel() else float("nan")
            result["group_iou"] = grouped
        return result


class LogitVoteAccumulator:
    """Accumulate repeated full-cloud logits with explicit coverage counts."""

    def __init__(self, num_points: int, num_classes: int) -> None:
        self.logit_sum = torch.zeros(num_points, num_classes, dtype=torch.float64)
        self.count = torch.zeros(num_points, dtype=torch.int64)

    def update(self, point_index: torch.Tensor, logits: torch.Tensor) -> None:
        point_index = point_index.detach().cpu().long()
        logits = logits.detach().cpu().double()
        if point_index.ndim != 1 or logits.shape != (point_index.numel(), self.logit_sum.shape[1]):
            raise ValueError("point_index [P] and logits [P,C] are required")
        self.logit_sum.index_add_(0, point_index, logits)
        self.count.index_add_(0, point_index, torch.ones_like(point_index))

    def prediction(self) -> torch.Tensor:
        if torch.any(self.count == 0):
            missing = int(torch.count_nonzero(self.count == 0))
            raise RuntimeError(f"full-cloud voting left {missing} points uncovered")
        return (self.logit_sum / self.count[:, None]).argmax(-1)
