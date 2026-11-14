"""Contracts for the prospective three-level S8 architecture."""
from __future__ import annotations

import numpy as np
import torch

from camo_build.coarsening import build_fixed_proxy_support
from camo_build.neighbors import scipy_knn_indices
from camo_build.pyramid_s8 import CAMOPyramidS8
from camo_s8_dataset import build_s8_topology


def _knn(points: torch.Tensor, k: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([
        scipy_knn_indices(item.numpy(), k) for item in points
    ])).long()


def test_s8_three_level_forward_backward_mass_and_adjoint_contracts() -> None:
    generator = torch.Generator().manual_seed(801)
    batch, n0, n1, n2 = 2, 32, 16, 8
    xyz = torch.randn(batch, n0, 3, generator=generator)
    normals = torch.nn.functional.normalize(
        torch.randn(batch, n0, 3, generator=generator), dim=-1
    )
    rgb = torch.rand(batch, n0, 3, generator=generator)
    features = torch.cat((xyz, normals, rgb), dim=-1)
    mass = torch.rand(batch, n0, generator=generator) + 0.2
    mass /= mass.mean(dim=1, keepdim=True)
    anchor1 = torch.tensor(np.stack([
        np.linspace(0, n0 - 1, n1, dtype=np.int64) for _ in range(batch)
    ]))
    support01 = torch.from_numpy(np.stack([
        build_fixed_proxy_support(xyz[b].numpy(), anchor1[b].numpy(), k=4)
        for b in range(batch)
    ])).long()
    xyz1 = xyz[torch.arange(batch)[:, None], anchor1]
    anchor2 = torch.tensor(np.stack([
        np.linspace(0, n1 - 1, n2, dtype=np.int64) for _ in range(batch)
    ]))
    support12 = torch.from_numpy(np.stack([
        build_fixed_proxy_support(xyz1[b].numpy(), anchor2[b].numpy(), k=4)
        for b in range(batch)
    ])).long()
    model = CAMOPyramidS8(
        widths=(16, 24, 32), local_blocks_per_level=1,
        attention_heads=4, expected_points=(n0, n1, n2),
    )
    output, state = model(
        features, mass, _knn(xyz, 4), anchor1, support01,
        _knn(xyz1, 4), anchor2, support12, _knn(xyz1[torch.arange(batch)[:, None], anchor2], 4),
        return_pyramid=True,
    )
    assert output.shape == (batch, n0, 31)
    torch.testing.assert_close(state["mass1"].sum(1), mass.sum(1), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state["mass2"].sum(1), mass.sum(1), atol=2e-5, rtol=2e-5)
    p01, p12 = state["coarsening01"].plan, state["coarsening12"].plan
    torch.testing.assert_close(p01.pullback(torch.ones(batch * n1, 3)), torch.ones(batch * n0, 3))
    torch.testing.assert_close(p12.pullback(torch.ones(batch * n2, 3)), torch.ones(batch * n1, 3))
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_s8_topology_is_label_free_and_covers_both_proxy_levels() -> None:
    generator = torch.Generator().manual_seed(902)
    xyz = torch.randn(64, 3, generator=generator)
    normals = torch.nn.functional.normalize(torch.randn(64, 3, generator=generator), dim=-1)
    rgb = torch.rand(64, 3, generator=generator)
    features = torch.cat((xyz, normals, rgb), dim=-1)
    mass = torch.rand(64, generator=generator) + 0.2
    topology = build_s8_topology(
        features, mass, proxies1=32, proxies2=8,
        support_k=4, k0=6, k1=6, k2=4,
    )
    assert topology["neighbors0"].shape == (64, 6)
    assert topology["anchor1"].shape == (32,)
    assert topology["support01"].shape == (64, 4)
    assert topology["neighbors1"].shape == (32, 6)
    assert topology["anchor2"].shape == (8,)
    assert topology["support12"].shape == (32, 4)
    assert topology["neighbors2"].shape == (8, 4)
