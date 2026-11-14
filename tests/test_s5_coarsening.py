"""S5 detached coarsening and single-level pyramid contract tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from camo_build.coarsening import (
    SparseMeasureCoarsener,
    build_fixed_proxy_support,
)
from camo_build.pyramid import CAMOPyramidS5
from camo_build.neighbors import scipy_knn_indices
from camo_build.transport import semi_relaxed_sparse_transport
from camo_build.warmstart_transport import semi_relaxed_sparse_transport_warm


def _fixture(batch: int = 2, points: int = 32, proxies: int = 8) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2026)
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
    support = []
    proxy_graph = []
    fine_graph = []
    for item in range(batch):
        target = build_fixed_proxy_support(
            xyz[item].numpy(), anchors[item].numpy(), k=4
        )
        support.append(torch.from_numpy(target))
        proxy_graph.append(
            torch.from_numpy(scipy_knn_indices(xyz[item, anchors[item]].numpy(), 3))
        )
        fine_graph.append(torch.from_numpy(scipy_knn_indices(xyz[item].numpy(), 4)))
    return (
        features,
        mass,
        torch.stack(fine_graph).long(),
        anchors.long(),
        torch.stack(support).long(),
        torch.stack(proxy_graph).long(),
    )


@pytest.mark.parametrize("mode", ["hard", "row_softmax", "srot"])
def test_s5_plan_preserves_each_source_mass_and_constants(mode: str) -> None:
    features, mass, _, anchors, support, _ = _fixture()
    hidden = torch.randn(2, 32, 12, generator=torch.Generator().manual_seed(91))
    result = SparseMeasureCoarsener(
        mode=mode, epsilon=0.2, tau=1.0, max_iterations=100
    )(
        features[..., :3], features[..., 3:6], hidden, mass,
        anchors, support,
    )
    plan = result.plan
    torch.testing.assert_close(
        plan.row_mass(), mass.flatten().float(), atol=2e-6, rtol=2e-6
    )
    assert plan.relative_mass_residual < 1e-6
    assert torch.all(plan.target_mass > 0)
    constant = plan.pushforward(torch.ones(2 * 32, 3))
    torch.testing.assert_close(constant, torch.ones_like(constant), atol=2e-6, rtol=2e-6)
    assert result.proxy_xyz.shape == (2, 8, 3)
    assert result.decode_index.shape == (2, 32)
    assert result.plan.value.requires_grad is False


def test_same_support_softmax_is_exact_tau_zero_srot_control() -> None:
    features, mass, _, anchors, support, _ = _fixture(batch=1)
    hidden = torch.randn(1, 32, 8, generator=torch.Generator().manual_seed(5))
    soft = SparseMeasureCoarsener(mode="row_softmax", epsilon=0.3)(
        features[..., :3], features[..., 3:6], hidden, mass,
        anchors, support,
    )
    tau_zero = SparseMeasureCoarsener(mode="srot", epsilon=0.3, tau=0.0)(
        features[..., :3], features[..., 3:6], hidden, mass,
        anchors, support,
    )
    torch.testing.assert_close(soft.plan.source_index, tau_zero.plan.source_index)
    torch.testing.assert_close(soft.plan.target_index, tau_zero.plan.target_index)
    torch.testing.assert_close(soft.plan.value, tau_zero.plan.value, atol=1e-6, rtol=1e-5)


def test_hard_control_has_one_edge_per_source_and_forces_anchor_coverage() -> None:
    features, mass, _, anchors, support, _ = _fixture(batch=1)
    hidden = torch.randn(1, 32, 8)
    result = SparseMeasureCoarsener(mode="hard")(
        features[..., :3], features[..., 3:6], hidden, mass,
        anchors, support,
    )
    assert result.plan.edge_count == 32
    assert torch.equal(
        result.plan.target_index[anchors.flatten()], torch.arange(8)
    )


def test_fixed_support_is_bounded_and_covers_every_proxy() -> None:
    xyz = np.random.default_rng(7).normal(size=(64, 3)).astype(np.float32)
    anchors = np.arange(0, 64, 8, dtype=np.int64)
    target = build_fixed_proxy_support(xyz, anchors, k=4)
    assert target.shape == (64, 4)
    assert target.dtype == np.int64
    assert set(np.unique(target)) == set(range(len(anchors)))
    for proxy, source in enumerate(anchors):
        assert proxy in target[source]


@pytest.mark.parametrize("mode", ["hard", "row_softmax", "srot"])
def test_s5_model_is_finite_and_backpropagates(mode: str) -> None:
    features, mass, fine_graph, anchors, support, proxy_graph = _fixture()
    model = CAMOPyramidS5(width=16, coarsening_mode=mode)
    logits = model(features, mass, fine_graph, anchors, support, proxy_graph)
    assert logits.shape == (2, 32, 31)
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_coarsener_rejects_nonpositive_mass_instead_of_clamping() -> None:
    features, mass, _, anchors, support, _ = _fixture(batch=1)
    mass[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        SparseMeasureCoarsener(mode="srot")(
            features[..., :3], features[..., 3:6], torch.randn(1, 32, 8),
            mass, anchors, support,
        )


def test_warm_start_reaches_same_plan_and_reduces_repeated_work() -> None:
    features, mass, _, anchors, support, _ = _fixture(batch=1)
    xyz = features[0, :, :3]
    proxy_xyz = xyz[anchors[0]]
    source = torch.arange(32)[:, None].expand(32, 4).reshape(-1)
    target = support[0].reshape(-1)
    cost = (xyz[source] - proxy_xyz[target]).square().sum(-1)
    reference, reference_diag = semi_relaxed_sparse_transport(
        mass[0], source, target, cost, num_targets=8,
        epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-6,
    )
    cold, cold_diag, dual = semi_relaxed_sparse_transport_warm(
        mass[0], source, target, cost, num_targets=8,
        epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-6,
    )
    warm, warm_diag, _ = semi_relaxed_sparse_transport_warm(
        mass[0], source, target, cost, num_targets=8,
        initial_log_v=dual, epsilon=0.2, tau=1.0,
        max_iterations=100, tolerance=1e-6,
    )
    assert reference_diag.converged and cold_diag.converged and warm_diag.converged
    torch.testing.assert_close(cold.value, reference.value, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(warm.value, reference.value, atol=2e-5, rtol=2e-4)
    assert warm_diag.iterations <= cold_diag.iterations
    assert warm_diag.max_relative_row_residual < 1e-5
