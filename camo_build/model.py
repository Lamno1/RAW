"""Standalone CAMO-Build Prototype 1 segmentation architecture."""
from __future__ import annotations

import torch
from torch import nn

from .operator import LocalMeasureAwareOperator


class CAMOPrototype1(nn.Module):
    """Two-block local measure-aware point segmentation model."""

    def __init__(
        self,
        num_classes: int = 31,
        input_channels: int = 9,
        width: int = 64,
        mass_mode: str = "density",
        kernel_min: float = 0.25,
        kernel_max: float = 4.0,
    ) -> None:
        super().__init__()
        if mass_mode not in {"none", "uniform", "density"}:
            raise ValueError("mass_mode must be none, uniform, or density")
        self.mass_mode = mass_mode
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        block_kwargs = {
            "channels": width,
            "kernel_min": kernel_min,
            "kernel_max": kernel_max,
        }
        self.blocks = nn.ModuleList(
            [LocalMeasureAwareOperator(**block_kwargs) for _ in range(2)]
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
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        xyz = features[..., :3]
        normals = features[..., 3:6]
        if self.mass_mode == "none":
            operator_mass = None
        elif self.mass_mode == "uniform":
            operator_mass = torch.ones_like(mass)
        else:
            operator_mass = mass.detach()

        hidden = self.embedding(features)
        for block in self.blocks:
            hidden = block(xyz, normals, hidden, operator_mass, neighbors)
        logits = self.head(hidden)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model logits contain NaN or Inf")
        return logits

