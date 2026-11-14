"""Gravity-preserving edge invariants for CAMO-Build."""
from __future__ import annotations

import torch

from .neighbors import gather_neighbors


def gravity_invariants(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    neighbors: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""Build [r_xy, delta_z, d_3d, n_i*n_j, n_i*z, n_j*z]."""
    xyz_j = gather_neighbors(xyz, neighbors)
    normal_j = gather_neighbors(normals, neighbors)
    xyz_i = xyz[:, :, None, :]
    normal_i = normals[:, :, None, :]

    delta = xyz_j - xyz_i
    radial = torch.sqrt(delta[..., :2].square().sum(dim=-1) + epsilon)
    vertical = delta[..., 2]
    distance = torch.sqrt(delta.square().sum(dim=-1) + epsilon)
    normal_cosine = (normal_i * normal_j).sum(dim=-1).clamp(-1.0, 1.0)
    normal_i_z = normal_i[..., 2].expand_as(vertical).clamp(-1.0, 1.0)
    normal_j_z = normal_j[..., 2].clamp(-1.0, 1.0)
    result = torch.stack(
        [radial, vertical, distance, normal_cosine, normal_i_z, normal_j_z],
        dim=-1,
    )
    if not torch.isfinite(result).all():
        raise FloatingPointError("geometry invariants contain NaN or Inf")
    return result

