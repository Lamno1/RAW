"""Controlled S7 architecture: frozen row-softmax backbone plus global block."""
from __future__ import annotations

import torch
from torch import nn

from .coarsening import CoarseningResult, SparseMeasureCoarsener
from .operator import LocalMeasureAwareOperator
from .s7_attention import GLOBAL_MODES, GravityAwareProxyAttention, build_global_block


class CAMOPyramidS7(nn.Module):
    """Fine LMAO -> 256 proxies -> proxy LMAO -> optional global MHA -> decode."""

    def __init__(
        self,
        num_classes: int = 31,
        input_channels: int = 9,
        width: int = 64,
        global_mode: str = "mass_aware_mha",
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if global_mode not in GLOBAL_MODES:
            raise ValueError(f"global_mode must be one of {sorted(GLOBAL_MODES)}")
        self.global_mode = global_mode
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.fine_block = LocalMeasureAwareOperator(width)
        self.coarsener = SparseMeasureCoarsener(
            mode="row_softmax", epsilon=0.2, tau=0.0,
            max_iterations=100, transport_backend="reference",
        )
        self.proxy_block = LocalMeasureAwareOperator(width)
        self.fusion = nn.Sequential(
            nn.Linear(2 * width, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(width, num_classes),
        )
        # Initialize candidate-only weights from the same post-backbone RNG
        # state while restoring that RNG afterward.  Common parameters and the
        # subsequent dropout stream then match exactly across paired arms.
        with torch.random.fork_rng(devices=[]):
            self.global_block = build_global_block(global_mode, width, attention_heads)

    def forward(
        self,
        features: torch.Tensor,
        mass: torch.Tensor,
        fine_neighbors: torch.Tensor,
        anchor_index: torch.Tensor,
        support_target: torch.Tensor,
        proxy_neighbors: torch.Tensor,
        *,
        return_coarsening: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CoarseningResult]:
        xyz, normals = features[..., :3], features[..., 3:6]
        hidden = self.embedding(features)
        hidden = self.fine_block(xyz, normals, hidden, mass.detach(), fine_neighbors)
        coarse = self.coarsener(
            xyz, normals, hidden, mass.detach(), anchor_index, support_target
        )
        proxy_hidden = self.proxy_block(
            coarse.proxy_xyz, coarse.proxy_normals, coarse.proxy_features,
            coarse.proxy_mass, proxy_neighbors,
        )
        if isinstance(self.global_block, GravityAwareProxyAttention):
            proxy_hidden = self.global_block(
                coarse.proxy_xyz, coarse.proxy_normals, proxy_hidden,
                coarse.proxy_mass,
            )
        else:
            proxy_hidden = self.global_block(proxy_hidden)
        batch = torch.arange(features.shape[0], device=features.device)[:, None]
        decoded = proxy_hidden[batch, coarse.decode_index]
        fused = hidden + self.fusion(torch.cat((hidden, decoded), dim=-1))
        logits = self.head(fused)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("S7 logits contain NaN or Inf")
        if return_coarsening:
            return logits, coarse
        return logits
