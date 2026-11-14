"""S7 global proxy attention contracts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from camo_build.coarsening import build_fixed_proxy_support
from camo_build.neighbors import scipy_knn_indices
from camo_build.s7 import CAMOPyramidS7
from camo_build.s7_attention import GravityAwareProxyAttention


def test_mass_aware_attention_is_invariant_to_common_mass_scale() -> None:
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(2, 16, 3, generator=generator)
    normals = torch.nn.functional.normalize(
        torch.randn(2, 16, 3, generator=generator), dim=-1
    )
    features = torch.randn(2, 16, 32, generator=generator)
    mass = torch.rand(2, 16, generator=generator) + 0.1
    block = GravityAwareProxyAttention(32, 4, mass_aware=True).eval()
    first = block(xyz, normals, features, mass)
    second = block(xyz, normals, features, 9.0 * mass)
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)


def test_s7_arms_share_initialization_for_common_parameters() -> None:
    torch.manual_seed(4409)
    control = CAMOPyramidS7(width=16, global_mode="identity", attention_heads=4)
    torch.manual_seed(4409)
    candidate = CAMOPyramidS7(width=16, global_mode="mass_aware_mha", attention_heads=4)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    candidate_only = ("global_block.",)
    for name, value in control_state.items():
        if not name.startswith(candidate_only):
            torch.testing.assert_close(value, candidate_state[name], atol=0, rtol=0)


def test_mass_aware_attention_rejects_invalid_mass_and_large_proxy_set() -> None:
    block = GravityAwareProxyAttention(16, 4, maximum_proxies=8)
    xyz = torch.randn(1, 8, 3)
    normals = torch.nn.functional.normalize(torch.randn(1, 8, 3), dim=-1)
    features = torch.randn(1, 8, 16)
    mass = torch.ones(1, 8)
    mass[0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        block(xyz, normals, features, mass)
    with pytest.raises(ValueError, match="capped"):
        block(
            torch.randn(1, 9, 3), torch.randn(1, 9, 3),
            torch.randn(1, 9, 16), torch.ones(1, 9),
        )


@pytest.mark.parametrize("mode", ["identity", "unweighted_mha", "mass_aware_mha"])
def test_s7_all_arms_forward_backward_finite(mode: str) -> None:
    generator = torch.Generator().manual_seed(2026)
    batch, points, proxies = 2, 32, 8
    xyz = torch.randn(batch, points, 3, generator=generator)
    normals = torch.nn.functional.normalize(
        torch.randn(batch, points, 3, generator=generator), dim=-1
    )
    rgb = torch.rand(batch, points, 3, generator=generator)
    features = torch.cat((xyz, normals, rgb), dim=-1)
    mass = torch.rand(batch, points, generator=generator) + 0.2
    mass /= mass.mean(dim=1, keepdim=True)
    anchors = torch.tensor(
        np.stack([np.linspace(0, points - 1, proxies, dtype=np.int64)] * batch)
    )
    support, fine_graph, proxy_graph = [], [], []
    for item in range(batch):
        support.append(torch.from_numpy(build_fixed_proxy_support(
            xyz[item].numpy(), anchors[item].numpy(), k=4
        )))
        fine_graph.append(torch.from_numpy(scipy_knn_indices(xyz[item].numpy(), 4)))
        proxy_graph.append(torch.from_numpy(scipy_knn_indices(
            xyz[item, anchors[item]].numpy(), 4
        )))
    model = CAMOPyramidS7(width=16, global_mode=mode, attention_heads=4)
    logits, coarse = model(
        features, mass, torch.stack(fine_graph), anchors,
        torch.stack(support), torch.stack(proxy_graph), return_coarsening=True,
    )
    assert logits.shape == (batch, points, 31)
    torch.testing.assert_close(
        coarse.proxy_mass.sum(1), mass.sum(1), atol=2e-5, rtol=2e-5
    )
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
