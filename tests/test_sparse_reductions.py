"""S3 segmented reduction tests."""
from __future__ import annotations

import torch

from camo_build.sparse_ops import segment_logsumexp, segment_max, segment_sum


def test_segment_reductions_match_reference_with_empty_segment() -> None:
    values = torch.tensor([[1.0, -1.0], [2.0, 3.0], [-4.0, 2.0], [0.5, 0.25]])
    segment = torch.tensor([0, 2, 0, 2])
    sums = segment_sum(values, segment, 4)
    maxima = segment_max(values, segment, 4)
    lse = segment_logsumexp(values, segment, 4)
    torch.testing.assert_close(sums[0], values[[0, 2]].sum(0))
    torch.testing.assert_close(sums[2], values[[1, 3]].sum(0))
    assert torch.equal(sums[1], torch.zeros(2))
    assert torch.isneginf(maxima[1]).all()
    torch.testing.assert_close(lse[0], torch.logsumexp(values[[0, 2]], dim=0))
    assert torch.isneginf(lse[3]).all()


def test_segment_logsumexp_is_stable_for_extreme_values() -> None:
    values = torch.tensor([10000.0, 9999.0, -10000.0], dtype=torch.float32)
    result = segment_logsumexp(values, torch.tensor([0, 0, 1]), 2)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.logsumexp(values[:2], dim=0))

