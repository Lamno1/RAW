"""Single-level CAMO-Pyramid architecture for the controlled S5 ablation."""
from __future__ import annotations

import torch
from torch import nn

from .coarsening import SparseMeasureCoarsener
from .coarsening import CoarseningResult
from .operator import LocalMeasureAwareOperator


class CAMOPyramidS5(nn.Module):
    """Fine LMAO -> detached coarsening -> proxy LMAO -> nearest decoder.

    The three experimental arms differ only in ``coarsening_mode``.  They have
    identical trainable parameters, local-block count, proxy count, decoder,
    objective, and head.
    """

    def __init__(
        self,
        num_classes: int = 31,
        input_channels: int = 9,
        width: int = 64,
        coarsening_mode: str = "srot",
        transport_epsilon: float = 0.2,
        transport_tau: float = 1.0,
        transport_iterations: int = 100,
        warm_start: bool = False,
        transport_backend: str = "reference",
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.fine_block = LocalMeasureAwareOperator(width)
        self.coarsener = SparseMeasureCoarsener(
            mode=coarsening_mode,
            epsilon=transport_epsilon,
            tau=transport_tau,
            max_iterations=transport_iterations,
            warm_start=warm_start,
            transport_backend=transport_backend,
        )
        self.proxy_block = LocalMeasureAwareOperator(width)
        self.fusion = nn.Sequential(
            nn.Linear(2 * width, width), nn.LayerNorm(width), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(width, num_classes),
        )

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
        xyz = features[..., :3]
        normals = features[..., 3:6]
        hidden = self.embedding(features)
        hidden = self.fine_block(
            xyz, normals, hidden, mass.detach(), fine_neighbors
        )
        coarse = self.coarsener(
            xyz, normals, hidden, mass.detach(), anchor_index, support_target
        )
        proxy_hidden = self.proxy_block(
            coarse.proxy_xyz,
            coarse.proxy_normals,
            coarse.proxy_features,
            coarse.proxy_mass,
            proxy_neighbors,
        )
        batch = torch.arange(features.shape[0], device=features.device)[:, None]
        decoded = proxy_hidden[batch, coarse.decode_index]
        fused = hidden + self.fusion(torch.cat((hidden, decoded), dim=-1))
        logits = self.head(fused)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("S5 model logits contain NaN or Inf")
        if return_coarsening:
            return logits, coarse
        return logits
