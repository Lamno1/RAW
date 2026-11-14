"""Three-resolution S8 CAMO-Pyramid candidate.

For each sparse plan P_l with exact source marginal m_l, encoding is
    H_{l+1} = D_{m_{l+1}}^{-1} P_l^T H_l,
and decoding is its mass-weighted adjoint
    Hhat_l = D_{m_l}^{-1} P_l H_{l+1}.
The model uses the already validated S5 row-softmax coarsener at both sparse
transitions and the frozen S7 mass-aware attention only at 256 proxies. This
module is an architecture implementation, not training authorization.
"""
from __future__ import annotations

import torch
from torch import nn

from .coarsening import CoarseningResult, SparseMeasureCoarsener
from .operator import LocalMeasureAwareOperator
from .s7_attention import GravityAwareProxyAttention


class _LocalStage(nn.Module):
    def __init__(self, width: int, blocks: int = 2) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("each pyramid level needs at least one local block")
        self.blocks = nn.ModuleList(LocalMeasureAwareOperator(width) for _ in range(blocks))

    def forward(
        self, xyz: torch.Tensor, normals: torch.Tensor, hidden: torch.Tensor,
        mass: torch.Tensor, neighbors: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(xyz, normals, hidden, mass.detach(), neighbors)
        return hidden


class CAMOPyramidS8(nn.Module):
    """End-to-end 4096→1024→256 pyramid with sparse measure transfers.

    Input geometry/features are ``[B,4096,9]``. Anchor/support tensors are
    supplied by the frozen, label-free topology pipeline; all neighbor arrays
    are integer ``[B,N,k]`` tensors. The model itself creates no dense point-to-
    proxy cost matrix. The output is pointwise ``[B,4096,num_classes]`` logits.
    """

    def __init__(
        self,
        num_classes: int = 31,
        input_channels: int = 9,
        widths: tuple[int, int, int] = (64, 96, 128),
        local_blocks_per_level: int = 2,
        attention_heads: int = 4,
        expected_points: tuple[int, int, int] = (4096, 1024, 256),
    ) -> None:
        super().__init__()
        if len(widths) != 3 or any(width <= 0 for width in widths):
            raise ValueError("widths must contain three positive values")
        if any(width % attention_heads for width in widths[2:]):
            raise ValueError("proxy width must be divisible by attention_heads")
        if len(expected_points) != 3 or not (
            expected_points[0] > expected_points[1] > expected_points[2] > 0
        ):
            raise ValueError("expected_points must be strictly decreasing positive counts")
        self.expected_points = tuple(int(n) for n in expected_points)
        w0, w1, w2 = widths
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, w0), nn.LayerNorm(w0), nn.SiLU()
        )
        self.level0 = _LocalStage(w0, local_blocks_per_level)
        self.coarsen01 = SparseMeasureCoarsener(
            mode="row_softmax", epsilon=0.2, tau=0.0, max_iterations=100,
            transport_backend="reference",
        )
        self.project01 = nn.Sequential(nn.Linear(w0, w1), nn.LayerNorm(w1), nn.SiLU())
        self.level1 = _LocalStage(w1, local_blocks_per_level)
        self.coarsen12 = SparseMeasureCoarsener(
            mode="row_softmax", epsilon=0.2, tau=0.0, max_iterations=100,
            transport_backend="reference",
        )
        self.project12 = nn.Sequential(nn.Linear(w1, w2), nn.LayerNorm(w2), nn.SiLU())
        self.level2 = _LocalStage(w2, local_blocks_per_level)
        self.global_block = GravityAwareProxyAttention(
            width=w2, heads=attention_heads, mass_aware=True,
            maximum_proxies=max(512, expected_points[2]),
        )

        self.decode1_projection = nn.Sequential(
            nn.Linear(w2, w1), nn.LayerNorm(w1), nn.SiLU()
        )
        self.decode1_fusion = nn.Sequential(
            nn.Linear(2 * w1, w1), nn.LayerNorm(w1), nn.SiLU()
        )
        self.decode0_projection = nn.Sequential(
            nn.Linear(w1, w0), nn.LayerNorm(w0), nn.SiLU()
        )
        self.decode0_fusion = nn.Sequential(
            nn.Linear(2 * w0, w0), nn.LayerNorm(w0), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(w0), nn.Linear(w0, w0), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(w0, num_classes),
        )

    @staticmethod
    def _check_mass(plan: object, level: str) -> None:
        residual = plan.relative_mass_residual
        if not torch.isfinite(torch.tensor(residual)) or residual > 1e-6:
            raise FloatingPointError(f"{level} relative mass residual {residual:.3e} exceeds 1e-6")

    def forward(
        self,
        features: torch.Tensor,
        mass: torch.Tensor,
        neighbors0: torch.Tensor,
        anchor1: torch.Tensor,
        support01: torch.Tensor,
        neighbors1: torch.Tensor,
        anchor2: torch.Tensor,
        support12: torch.Tensor,
        neighbors2: torch.Tensor,
        *,
        return_pyramid: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | CoarseningResult]]:
        if features.ndim != 3 or features.shape[-1] < 6:
            raise ValueError("features must have shape [B,N,C>=6]")
        batch, points, _ = features.shape
        if points != self.expected_points[0] or mass.shape != (batch, points):
            raise ValueError("input point/mass shape does not match S8 configuration")
        if not torch.isfinite(features).all() or not torch.isfinite(mass).all():
            raise FloatingPointError("S8 input contains NaN or Inf")
        if torch.any(mass <= 0):
            raise ValueError("S8 requires strictly positive input masses")
        xyz0, normals0 = features[..., :3], features[..., 3:6]
        h0 = self.level0(xyz0, normals0, self.embedding(features), mass, neighbors0)
        coarse01 = self.coarsen01(
            xyz0, normals0, h0, mass.detach(), anchor1, support01
        )
        if coarse01.proxy_xyz.shape[1] != self.expected_points[1]:
            raise ValueError("first coarsening did not produce the configured 1024 level")
        self._check_mass(coarse01.plan, "4096->1024")

        xyz1, normals1 = coarse01.proxy_xyz, coarse01.proxy_normals
        mass1 = coarse01.proxy_mass
        h1 = self.level1(
            xyz1, normals1, self.project01(coarse01.proxy_features), mass1, neighbors1
        )
        coarse12 = self.coarsen12(
            xyz1, normals1, h1, mass1.detach(), anchor2, support12
        )
        if coarse12.proxy_xyz.shape[1] != self.expected_points[2]:
            raise ValueError("second coarsening did not produce the configured 256 level")
        self._check_mass(coarse12.plan, "1024->256")

        xyz2, normals2 = coarse12.proxy_xyz, coarse12.proxy_normals
        mass2 = coarse12.proxy_mass
        h2 = self.level2(
            xyz2, normals2, self.project12(coarse12.proxy_features), mass2, neighbors2
        )
        h2 = self.global_block(xyz2, normals2, h2, mass2)

        # Exact weighted-adjoint transport pullbacks, not nearest-neighbor lookup.
        h1_dec = coarse12.plan.pullback(h2.flatten(0, 1)).reshape(
            batch, self.expected_points[1], h2.shape[-1]
        )
        h1_dec = self.decode1_fusion(torch.cat(
            (h1, self.decode1_projection(h1_dec)), dim=-1
        ))
        h0_dec = coarse01.plan.pullback(h1_dec.flatten(0, 1)).reshape(
            batch, points, h1_dec.shape[-1]
        )
        h0_dec = self.decode0_fusion(torch.cat(
            (h0, self.decode0_projection(h0_dec)), dim=-1
        ))
        logits = self.head(h0_dec)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("S8 output logits contain NaN or Inf")
        if return_pyramid:
            return logits, {
                "level0_features": h0,
                "level1_features": h1,
                "level2_features": h2,
                "mass0": mass,
                "mass1": mass1,
                "mass2": mass2,
                "coarsening01": coarse01,
                "coarsening12": coarse12,
            }
        return logits
