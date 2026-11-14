"""Numerically stable coarse global attention for the S7 experiment.

The quadratic operator is legal only at the frozen 256-proxy level.  Its
measure-aware arm implements

    alpha_ij = softmax_j(q_i k_j / sqrt(d) + b(g_ij) + log(m_j + eps)).

Mass is read as a quadrature measure and is never modified by this block.
"""
from __future__ import annotations

import math

import torch
from torch import nn


GLOBAL_MODES = frozenset({"identity", "unweighted_mha", "mass_aware_mha"})


class GravityAwareProxyAttention(nn.Module):
    """Dense FP32 attention restricted to a small proxy set.

    Args:
        width: Proxy feature width.
        heads: Number of attention heads; ``width`` must be divisible by it.
        mass_aware: Add ``log(m_j + eps)`` to every key logit.
        epsilon: Numerical guard used only inside normalization/logarithms.
        maximum_proxies: Hard safety cap preventing accidental point-level MHA.
    """

    def __init__(
        self,
        width: int = 64,
        heads: int = 4,
        *,
        mass_aware: bool = True,
        epsilon: float = 1e-6,
        maximum_proxies: int = 512,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if epsilon <= 0 or maximum_proxies <= 0:
            raise ValueError("epsilon and maximum_proxies must be positive")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.mass_aware = bool(mass_aware)
        self.eps = float(epsilon)
        self.maximum_proxies = int(maximum_proxies)

        self.input_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        # Six coefficients per head for the frozen SO(2)_z descriptors.
        self.geometry_weight = nn.Parameter(torch.zeros(heads, 6))
        self.geometry_bias = nn.Parameter(torch.zeros(heads))
        self.output_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def _relative_bias(
        self, xyz: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        """Return `[B,H,M,M]` without materializing `[B,M,M,6]`."""
        xyz32 = xyz.float()
        normal32 = torch.nn.functional.normalize(normals.float(), dim=-1, eps=self.eps)
        dx = xyz32[:, :, None, 0] - xyz32[:, None, :, 0]
        dy = xyz32[:, :, None, 1] - xyz32[:, None, :, 1]
        dz = xyz32[:, None, :, 2] - xyz32[:, :, None, 2]
        radial = torch.sqrt(dx.square() + dy.square() + self.eps)
        distance = torch.sqrt(radial.square() + dz.square() + self.eps)

        # Remove global scale while preserving SO(2)_z invariance and dz sign.
        scale = distance.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        descriptors = (
            radial / scale,
            dz / scale,
            distance / scale,
            torch.einsum("bid,bjd->bij", normal32, normal32).abs(),
            normal32[:, :, 2].abs()[:, :, None].expand_as(distance),
            normal32[:, :, 2].abs()[:, None, :].expand_as(distance),
        )
        bias = self.geometry_bias[None, :, None, None].float().expand(
            xyz.shape[0], -1, xyz.shape[1], xyz.shape[1]
        )
        for index, descriptor in enumerate(descriptors):
            bias = bias + self.geometry_weight[:, index][None, :, None, None].float() * descriptor[:, None]
        return bias

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        features: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or normals.shape != xyz.shape:
            raise ValueError("xyz and normals must have shape [B,M,3]")
        if features.ndim != 3 or features.shape[:2] != xyz.shape[:2]:
            raise ValueError("features must have shape [B,M,D]")
        if features.shape[-1] != self.width or mass.shape != xyz.shape[:2]:
            raise ValueError("feature width or mass shape mismatch")
        if xyz.shape[1] > self.maximum_proxies:
            raise ValueError(
                f"global attention is capped at {self.maximum_proxies} proxies"
            )
        tensors = (xyz, normals, features, mass)
        if any(not torch.isfinite(value).all() for value in tensors):
            raise FloatingPointError("S7 attention input contains NaN or Inf")
        if torch.any(mass <= 0):
            raise ValueError("S7 attention requires strictly positive masses")

        batch, proxies, _ = features.shape
        normalized = self.input_norm(features)
        qkv = self.qkv(normalized).reshape(
            batch, proxies, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3).float()
        key = key.permute(0, 2, 1, 3).float()
        value = value.permute(0, 2, 1, 3).float()
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self._relative_bias(xyz, normals)
        if self.mass_aware:
            log_mass = torch.log(mass.float() + self.eps)
            log_mass = log_mass - log_mass.mean(dim=-1, keepdim=True)
            logits = logits + log_mass[:, None, None, :]
        probability = torch.softmax(logits.clamp(-80.0, 80.0), dim=-1)
        attended = torch.matmul(probability, value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, proxies, self.width)
        attended = self.output(attended.to(features.dtype))
        hidden = features + attended
        result = hidden + self.ffn(self.output_norm(hidden))
        if not torch.isfinite(result).all():
            raise FloatingPointError("S7 attention output contains NaN or Inf")
        return result


def build_global_block(mode: str, width: int, heads: int) -> nn.Module:
    if mode not in GLOBAL_MODES:
        raise ValueError(f"mode must be one of {sorted(GLOBAL_MODES)}")
    if mode == "identity":
        return nn.Identity()
    return GravityAwareProxyAttention(
        width=width, heads=heads, mass_aware=mode == "mass_aware_mha"
    )

