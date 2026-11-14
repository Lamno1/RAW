"""Local Measure-Aware Operator (LMAO), Prototype 1."""
from __future__ import annotations

import torch
from torch import nn

from .geometry import gravity_invariants
from .neighbors import gather_neighbors


class LocalMeasureAwareOperator(nn.Module):
    r"""Frozen local normalized integral operator.

    This layer preserves constant messages through row normalization. It does
    not claim conservation of the mass-weighted feature integral.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_min: float = 0.25,
        kernel_max: float = 4.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("channels must be at least 2")
        if not 0.0 < kernel_min < kernel_max:
            raise ValueError("kernel bounds must satisfy 0 < min < max")
        self.channels = channels
        self.kernel_min = float(kernel_min)
        self.kernel_max = float(kernel_max)
        self.epsilon = float(epsilon)

        hidden = max(channels // 2, 8)
        self.kernel_network = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        self.self_projection = nn.Linear(channels, channels, bias=False)
        self.value_projection = nn.Linear(channels, channels, bias=False)
        self.activation = nn.SiLU()

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        features: torch.Tensor,
        mass: torch.Tensor | None,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the local operator to [B, N, D] features."""
        if features.ndim != 3 or features.shape[-1] != self.channels:
            raise ValueError(
                f"features must have shape [B, N, {self.channels}]"
            )
        if xyz.shape != normals.shape or xyz.shape[-1] != 3:
            raise ValueError("xyz and normals must both have shape [B, N, 3]")
        if xyz.shape[:2] != features.shape[:2]:
            raise ValueError("geometry and features disagree on dimensions")
        if mass is not None and mass.shape != features.shape[:2]:
            raise ValueError("mass must have shape [B, N]")

        invariants = gravity_invariants(xyz, normals, neighbors, self.epsilon)
        device_type = features.device.type
        amp_enabled = features.is_cuda
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            raw_kernel = self.kernel_network(invariants)
            kernel = self.kernel_min + (
                self.kernel_max - self.kernel_min
            ) * torch.sigmoid(raw_kernel)
            # Project before gather: O(N D^2), not O(N k D^2).
            projected_values = self.value_projection(features)
            self_term = self.self_projection(features)
            neighbor_values = gather_neighbors(projected_values, neighbors)

        # Frozen numerical rule: mass-weighted reductions and division are FP32.
        kernel32 = kernel.float()
        values32 = neighbor_values.float()
        if mass is None:
            weighted_kernel = kernel32
        else:
            mass_neighbors = gather_neighbors(mass[..., None], neighbors)[..., 0]
            weighted_kernel = kernel32 * mass_neighbors.float()[..., None]
        partition = weighted_kernel.sum(dim=2).clamp_min(self.epsilon)
        message = (weighted_kernel * values32).sum(dim=2) / partition
        message = message.to(dtype=self_term.dtype)
        output = features + self.activation(self_term + message)
        if not torch.isfinite(output).all():
            raise FloatingPointError("LMAO output contains NaN or Inf")
        return output

