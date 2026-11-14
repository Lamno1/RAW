"""Topology adapter for the frozen CAMO-Pyramid S5 ablation."""
from __future__ import annotations

import numpy as np
import torch

from camo_build.assignment import select_spatial_anchors
from camo_build.coarsening import build_fixed_proxy_support
from camo_build.neighbors import scipy_knn_indices
from camo_dataset import CAMOBuildingNetDataset


def build_s5_topology(
    features: torch.Tensor,
    mass: torch.Tensor,
    *,
    proxies: int,
    support_k: int,
    proxy_k: int,
) -> dict[str, torch.Tensor]:
    """Build label-free CPU topology for one sampled cloud.

    Anchor/support construction is detached and uses ``O(N*M_candidate)``
    weighted FPS plus cKDTree queries.  It never constructs ``[N,M]`` or
    ``[N,N]`` dense tensors.
    """

    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError("features must have shape [N,C>=6]")
    if mass.shape != (features.shape[0],):
        raise ValueError("mass must have shape [N]")
    if not 0 < proxies < features.shape[0]:
        raise ValueError("proxies must satisfy 0 < M < N")
    xyz = features[:, :3].detach().to(device="cpu", dtype=torch.float32).numpy()
    measure = mass.detach().to(device="cpu", dtype=torch.float32).numpy()
    anchor_index = select_spatial_anchors(xyz, measure, proxies)
    support_target = build_fixed_proxy_support(
        xyz, anchor_index, k=support_k
    )
    proxy_neighbors = scipy_knn_indices(xyz[anchor_index], proxy_k)
    return {
        "anchor_index": torch.from_numpy(anchor_index).long(),
        "support_target": torch.from_numpy(support_target).long(),
        "proxy_neighbors": torch.from_numpy(proxy_neighbors).long(),
    }


class CAMOS5Dataset(CAMOBuildingNetDataset):
    """BuildingNet crop plus label-free S5 anchor/support topology."""

    def __init__(
        self,
        *args: object,
        proxies: int = 1024,
        support_k: int = 8,
        proxy_k: int = 16,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < proxies < self.num_points:
            raise ValueError("proxies must satisfy 0 < M < num_points")
        if not 0 < support_k <= proxies:
            raise ValueError("support_k must satisfy 0 < k <= proxies")
        if not 0 < proxy_k < proxies:
            raise ValueError("proxy_k must satisfy 0 < k < proxies")
        self.proxies = int(proxies)
        self.support_k = int(support_k)
        self.proxy_k = int(proxy_k)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        topology = build_s5_topology(
            sample["features"],  # type: ignore[arg-type]
            sample["mass"],  # type: ignore[arg-type]
            proxies=self.proxies,
            support_k=self.support_k,
            proxy_k=self.proxy_k,
        )
        sample.update(topology)
        return sample
