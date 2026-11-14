"""Sparse neighborhood construction and indexed gathering.

Production graphs use SciPy's cKDTree on CPU. A dense PyTorch fallback is
intentionally absent: the project forbids dense O(N^2) neighborhoods above
4,000 points, and maintaining two numerically different graph paths would
weaken equation-to-code traceability.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def scipy_knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    """Return k non-self neighbors for a single point cloud.

    Args:
        xyz: Finite float array with shape [N, 3].
        k: Number of neighbors; must be smaller than N.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf")
    n = xyz.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k must satisfy 0 < k < N; received k={k}, N={n}")

    # Query extra neighbors so that the exact row index can be removed even
    # when duplicate coordinates make self not appear in the first position.
    query_k = min(n, k + 2)
    _, candidates = cKDTree(xyz).query(xyz, k=query_k, workers=-1)
    if candidates.ndim == 1:
        candidates = candidates[:, None]

    rows = np.arange(n, dtype=np.int64)[:, None]
    nonself = candidates != rows
    result = np.empty((n, k), dtype=np.int64)
    for row in range(n):
        selected = candidates[row, nonself[row]]
        if selected.size < k:
            raise RuntimeError(
                f"Unable to construct {k} non-self neighbors for row {row}"
            )
        result[row] = selected[:k]
    return result


def batched_scipy_knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Build CPU cKDTree graphs for a [B, N, 3] tensor.

    This function is non-differentiable by design. Graph topology is treated
    as fixed for each forward pass.
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
    arrays = xyz.detach().to(device="cpu", dtype=torch.float32).numpy()
    graphs = [scipy_knn_indices(points, k) for points in arrays]
    return torch.from_numpy(np.stack(graphs)).to(device=xyz.device)


def gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, N, C] values using [B, N, k] indices."""
    if values.ndim != 3 or indices.ndim != 3:
        raise ValueError("values and indices must be rank-3 tensors")
    if values.shape[:2] != indices.shape[:2]:
        raise ValueError("values and indices disagree on batch/point dimensions")
    if indices.dtype != torch.long:
        indices = indices.long()
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]

