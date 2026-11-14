"""Memory-bounded segmented reductions for sparse edge lists.

The implementation uses native ``scatter_add_``/``scatter_reduce_`` and never
materializes a dense source-by-target matrix. Reductions that enter OT are
promoted to FP32 even when feature tensors use AMP.
"""
from __future__ import annotations

import torch


def _validate(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim < 1:
        raise ValueError("values must have an edge dimension")
    if segment_ids.ndim != 1 or segment_ids.numel() != values.shape[0]:
        raise ValueError("segment_ids must have shape [E]")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    segment_ids = segment_ids.to(device=values.device, dtype=torch.long)
    if segment_ids.numel() and (
        int(segment_ids.min()) < 0 or int(segment_ids.max()) >= num_segments
    ):
        raise ValueError("segment id is outside [0, num_segments)")
    return values, segment_ids


def _expanded_index(segment_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    shape = (segment_ids.shape[0],) + (1,) * (values.ndim - 1)
    return segment_ids.reshape(shape).expand_as(values)


def segment_sum(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum edge values into ``num_segments`` without dense incidence tensors."""
    values, segment_ids = _validate(values, segment_ids, num_segments)
    output = values.new_zeros((num_segments,) + values.shape[1:])
    output.scatter_add_(0, _expanded_index(segment_ids, values), values)
    return output


def segment_max(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segmented maximum; empty segments are ``-inf``."""
    values, segment_ids = _validate(values, segment_ids, num_segments)
    if not values.is_floating_point():
        raise TypeError("segment_max requires a floating-point tensor")
    output = values.new_full((num_segments,) + values.shape[1:], -torch.inf)
    output.scatter_reduce_(
        0,
        _expanded_index(segment_ids, values),
        values,
        reduce="amax",
        include_self=True,
    )
    return output


def segment_logsumexp(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Stable FP32 segmented LogSumExp; empty segments are ``-inf``."""
    values, segment_ids = _validate(values, segment_ids, num_segments)
    work = values.float()
    maxima = segment_max(work, segment_ids, num_segments)
    gathered_max = maxima[segment_ids]
    finite_edge = torch.isfinite(gathered_max)
    shifted = torch.where(
        finite_edge,
        torch.exp(work - gathered_max),
        torch.zeros_like(work),
    )
    totals = segment_sum(shifted, segment_ids, num_segments)
    result = maxima + torch.log(totals.clamp_min(torch.finfo(work.dtype).tiny))
    return torch.where(torch.isfinite(maxima), result, maxima)

