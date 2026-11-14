"""PointNeXt-S baseline adapter with the S7 call signature.

The adapter accepts the same ``features`` tensor as the S7 models
(``[B, N, 9]``: normalized XYZ, unit normals, RGB) and returns logits with
shape ``[B, N, 31]``.  The per-point mass and any precomputed neighbor tensors
are ignored on purpose: the baseline receives no density mass and builds its
own neighborhoods inside the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE.parent / "third_party"
DEFAULT_CONFIG = HERE / "cfgs" / "pointnext-s_buildingnet.yaml"

for path in (str(THIRD_PARTY), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pointnet2_torch_shim  # noqa: E402

pointnet2_torch_shim.install()

from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig  # noqa: E402


class PointNeXtBaseline(nn.Module):
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        super().__init__()
        cfg = EasyConfig()
        cfg.load(str(config_path), recursive=True)
        self.network = build_model_from_cfg(cfg.model)

    def forward(
        self,
        features: torch.Tensor,
        mass: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        *unused: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 9:
            raise ValueError(f"features must have shape [B,N,9], got {tuple(features.shape)}")
        position = features[..., :3].contiguous()
        attributes = features[..., 3:9].transpose(1, 2).contiguous()
        logits = self.network({"pos": position, "x": attributes})
        return logits.transpose(1, 2).contiguous()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
