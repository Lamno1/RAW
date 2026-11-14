"""Pure-PyTorch replacement for the OpenPoints ``pointnet2_batch_cuda`` ops.

The machine has no CUDA compiler, so the official extension cannot be built.
Each wrapper below reproduces the semantics of the corresponding CUDA kernel in
``openpoints/cpp/pointnet2_batch/src`` and writes into the caller-provided
output tensor, exactly like the original wrappers.  The model architecture is
unchanged; only the operator implementation differs.

Semantics that matter for equivalence:
* furthest point sampling starts from index 0 and keeps a running minimum
  squared distance initialised to 1e10;
* ball query returns the first ``nsample`` indices, in index order, with
  squared distance strictly below ``radius**2``; remaining slots repeat the
  first hit; a query with no hit keeps the zero-initialised index 0;
* three_nn returns squared distances sorted ascending;
* gradient wrappers accumulate into the provided gradient tensor.
"""
from __future__ import annotations

import torch

_CHUNK_ELEMENTS = 64 * 1024 * 1024  # cap on the size of a temporary distance block


def _chunk_rows(batch: int, cols: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(1, batch * cols))


def furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output):
    xyz = xyz.float()
    batch = torch.arange(B, device=xyz.device)
    temp.fill_(1e10)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        output[:, i] = farthest.to(output.dtype)
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        torch.minimum(temp, dist, out=temp)
        farthest = temp.argmax(dim=1)


def gather_points_wrapper(B, C, N, npoint, features, idx, output):
    index = idx.long().unsqueeze(1).expand(B, C, npoint)
    output.copy_(features.gather(2, index))


def gather_points_grad_wrapper(B, C, N, npoint, grad_out, idx, grad_features):
    index = idx.long().unsqueeze(1).expand(B, C, npoint)
    grad_features.scatter_add_(2, index, grad_out)


def group_points_wrapper(B, C, N, npoint, nsample, features, idx, output):
    index = idx.long().reshape(B, 1, npoint * nsample).expand(B, C, npoint * nsample)
    output.copy_(features.gather(2, index).reshape(B, C, npoint, nsample))


def group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out, idx, grad_features):
    index = idx.long().reshape(B, 1, npoint * nsample).expand(B, C, npoint * nsample)
    grad_features.scatter_add_(2, index, grad_out.reshape(B, C, npoint * nsample))


def ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx):
    radius2 = float(radius) * float(radius)
    xyz = xyz.float()
    new_xyz = new_xyz.float()
    arange = torch.arange(N, device=xyz.device, dtype=torch.long)
    step = _chunk_rows(B, N)
    for start in range(0, npoint, step):
        stop = min(npoint, start + step)
        d2 = torch.cdist(new_xyz[:, start:stop], xyz).square()
        key = torch.where(d2 < radius2, arange.expand(d2.shape), torch.full(d2.shape, N, dtype=torch.long, device=d2.device))
        first, _ = torch.topk(key, nsample, dim=-1, largest=False, sorted=True)
        hit = first < N
        head = first[..., :1]
        filled = torch.where(hit, first, head.expand_as(first))
        filled = torch.where(head < N, filled, torch.zeros_like(filled))
        idx[:, start:stop] = filled.to(idx.dtype)


def three_nn_wrapper(B, N, m, unknown, known, dist2, idx):
    unknown = unknown.float()
    known = known.float()
    step = _chunk_rows(B, m)
    for start in range(0, N, step):
        stop = min(N, start + step)
        d2 = torch.cdist(unknown[:, start:stop], known).square()
        values, index = torch.topk(d2, 3, dim=-1, largest=False, sorted=True)
        dist2[:, start:stop] = values
        idx[:, start:stop] = index.to(idx.dtype)


def three_interpolate_wrapper(B, c, m, n, features, idx, weight, output):
    index = idx.long().reshape(B, 1, n * 3).expand(B, c, n * 3)
    picked = features.gather(2, index).reshape(B, c, n, 3)
    output.copy_((picked * weight.unsqueeze(1)).sum(-1))


def three_interpolate_grad_wrapper(B, c, n, m, grad_out, idx, weight, grad_features):
    index = idx.long().reshape(B, 1, n * 3).expand(B, c, n * 3)
    contribution = (grad_out.unsqueeze(-1) * weight.unsqueeze(1)).reshape(B, c, n * 3)
    grad_features.scatter_add_(2, index, contribution)


def install() -> None:
    """Make ``from openpoints.cpp import pointnet2_cuda`` resolve to this shim.

    Must run before any ``openpoints.models`` import.
    """
    import sys

    shim = sys.modules[__name__]
    import openpoints.cpp as cpp
    import openpoints.cpp.pointnet2_batch as pointnet2_batch

    cpp.pointnet2_cuda = shim
    pointnet2_batch.pointnet2_cuda = shim
    sys.modules.setdefault("pointnet2_batch_cuda", shim)
