"""Label-free two-transition sparse topology for the S8 pilot."""
from __future__ import annotations

import numpy as np
import torch

from camo_build.assignment import select_spatial_anchors
from camo_build.coarsening import SparseMeasureCoarsener, build_fixed_proxy_support
from camo_build.neighbors import scipy_knn_indices
from camo_dataset import CAMOBuildingNetDataset


def build_s8_topology(
    features: torch.Tensor,
    mass: torch.Tensor,
    *,
    proxies1: int = 1024,
    proxies2: int = 256,
    support_k: int = 8,
    k0: int = 16,
    k1: int = 16,
    k2: int = 12,
) -> dict[str, torch.Tensor]:
    """Build geometry-only graphs using exact row-softmax proxy locations.

    Topology and plans are detached CPU preprocessing. Hidden features do not
    enter the costs, so zero feature probes produce the same proxy geometry and
    masses as the training forward pass.
    """
    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError("features must have shape [N,C>=6]")
    if mass.shape != (features.shape[0],):
        raise ValueError("mass must have shape [N]")
    xyz = features[:, :3].detach().cpu().float().numpy()
    normals = features[:, 3:6].detach().cpu().float().numpy()
    measure = mass.detach().cpu().float().numpy()
    if not (0 < proxies2 < proxies1 < len(xyz)):
        raise ValueError("expected 0 < proxies2 < proxies1 < input points")
    anchor1_np = select_spatial_anchors(xyz, measure, proxies1)
    support01_np = build_fixed_proxy_support(xyz, anchor1_np, k=support_k)
    coarsener = SparseMeasureCoarsener(mode="row_softmax", epsilon=0.2, tau=0.0)
    with torch.no_grad():
        stage1 = coarsener(
            torch.from_numpy(xyz)[None], torch.from_numpy(normals)[None],
            torch.zeros(1, len(xyz), 64), torch.from_numpy(measure)[None],
            torch.from_numpy(anchor1_np)[None], torch.from_numpy(support01_np)[None],
        )
    xyz1 = stage1.proxy_xyz[0].numpy()
    normals1 = stage1.proxy_normals[0].numpy()
    mass1 = stage1.proxy_mass[0].numpy()
    anchor2_np = select_spatial_anchors(xyz1, mass1, proxies2)
    support12_np = build_fixed_proxy_support(xyz1, anchor2_np, k=support_k)
    with torch.no_grad():
        stage2 = coarsener(
            stage1.proxy_xyz, stage1.proxy_normals,
            torch.zeros(1, proxies1, 96), stage1.proxy_mass,
            torch.from_numpy(anchor2_np)[None], torch.from_numpy(support12_np)[None],
        )
    xyz2 = stage2.proxy_xyz[0].numpy()
    return {
        "neighbors0": torch.from_numpy(scipy_knn_indices(xyz, k0)).long(),
        "anchor1": torch.from_numpy(anchor1_np).long(),
        "support01": torch.from_numpy(support01_np).long(),
        "neighbors1": torch.from_numpy(scipy_knn_indices(xyz1, k1)).long(),
        "anchor2": torch.from_numpy(anchor2_np).long(),
        "support12": torch.from_numpy(support12_np).long(),
        "neighbors2": torch.from_numpy(scipy_knn_indices(xyz2, k2)).long(),
    }


class CAMOS8Dataset(CAMOBuildingNetDataset):
    """BuildingNet sampled clouds with full three-level sparse topology."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["build_neighbors"] = False
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        topology = build_s8_topology(sample["features"], sample["mass"])
        sample.update(topology)
        return sample
