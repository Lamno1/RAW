"""S4 full-cloud assignment and sparse semi-relaxed OT tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from camo_build.assignment import build_full_cloud_assignment
from camo_build.transport import (
    build_sparse_proxy_support,
    geometry_transport_cost,
    pushforward_axial_normals,
    semi_relaxed_sparse_transport,
)


def _problem() -> tuple[torch.Tensor, ...]:
    source_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0],
         [1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    proxy_xyz = source_xyz[[0, 2, 4]]
    mass = torch.tensor([0.5, 1.0, 2.0, 0.75, 1.25, 0.5])
    edge_source, edge_target = build_sparse_proxy_support(
        source_xyz.numpy(), proxy_xyz.numpy(), k=2, reciprocal=1
    )
    cost = (source_xyz[edge_source] - proxy_xyz[edge_target]).square().sum(-1)
    return source_xyz, proxy_xyz, mass, edge_source, edge_target, cost


def test_full_cloud_assignment_preserves_mass_and_adjoint() -> None:
    source_xyz, proxy_xyz, mass, *_ = _problem()
    plan = build_full_cloud_assignment(
        source_xyz.numpy(), mass.numpy(), anchor_xyz=proxy_xyz.numpy(), k=2
    )
    assert plan.edge_count <= source_xyz.shape[0] * 2 + proxy_xyz.shape[0]
    assert plan.relative_mass_residual < 1e-6
    assert torch.all(plan.target_mass > 0)
    source_feature = torch.randn(source_xyz.shape[0], 4, generator=torch.Generator().manual_seed(3))
    target_feature = torch.randn(proxy_xyz.shape[0], 4, generator=torch.Generator().manual_seed(4))
    assert plan.adjoint_error(source_feature, target_feature) < 1e-5
    pulled_constant = plan.pullback(torch.ones(proxy_xyz.shape[0], 2))
    torch.testing.assert_close(pulled_constant, torch.ones_like(pulled_constant), atol=1e-6, rtol=1e-6)


def test_sparse_srot_hard_row_mass_and_positive_targets() -> None:
    _, proxy_xyz, mass, edge_source, edge_target, cost = _problem()
    plan, diagnostics = semi_relaxed_sparse_transport(
        mass,
        edge_source,
        edge_target,
        cost,
        num_targets=proxy_xyz.shape[0],
        epsilon=0.2,
        tau=1.0,
        max_iterations=100,
        tolerance=1e-6,
    )
    assert diagnostics.relative_mass_residual < 1e-6
    assert diagnostics.max_relative_row_residual < 1e-5
    assert torch.all(plan.target_mass > 0)
    assert torch.isfinite(plan.value).all()
    assert plan.value.requires_grad is False


def test_tau_zero_matches_same_support_row_softmax() -> None:
    _, proxy_xyz, mass, edge_source, edge_target, cost = _problem()
    epsilon = 0.3
    plan, _ = semi_relaxed_sparse_transport(
        mass, edge_source, edge_target, cost,
        num_targets=proxy_xyz.shape[0], epsilon=epsilon, tau=0.0,
        max_iterations=2, tolerance=1e-7,
    )
    expected = torch.empty_like(plan.value)
    for source in range(mass.numel()):
        mask = edge_source == source
        expected[mask] = mass[source] * torch.softmax(-cost[mask] / epsilon, dim=0)
    torch.testing.assert_close(plan.value, expected, atol=1e-6, rtol=1e-5)


def test_srot_handles_duplicate_points_and_extreme_finite_costs() -> None:
    xyz = np.zeros((5, 3), dtype=np.float32)
    proxy = np.zeros((2, 3), dtype=np.float32)
    source, target = build_sparse_proxy_support(xyz, proxy, k=2, reciprocal=1)
    cost = torch.linspace(0.0, 1e4, source.numel())
    plan, diagnostics = semi_relaxed_sparse_transport(
        torch.ones(5), source, target, cost, num_targets=2,
        epsilon=10.0, tau=1.0, max_iterations=100,
    )
    assert torch.isfinite(plan.value).all()
    assert diagnostics.relative_mass_residual < 1e-6


def test_transport_rejects_missing_source_edges() -> None:
    with pytest.raises(ValueError, match="source"):
        semi_relaxed_sparse_transport(
            torch.ones(3), torch.tensor([0, 2]), torch.tensor([0, 1]),
            torch.ones(2), num_targets=2,
        )


def test_differentiable_transport_is_refused_before_s5() -> None:
    _, proxy_xyz, mass, edge_source, edge_target, cost = _problem()
    with pytest.raises(NotImplementedError, match="forbidden"):
        semi_relaxed_sparse_transport(
            mass, edge_source, edge_target, cost,
            num_targets=proxy_xyz.shape[0], detach_plan=False,
        )


def test_geometry_cost_is_invariant_to_normal_sign_by_default() -> None:
    source_xyz, proxy_xyz, _, edge_source, edge_target, _ = _problem()
    generator = torch.Generator().manual_seed(11)
    source_normal = torch.nn.functional.normalize(
        torch.randn(6, 3, generator=generator), dim=-1
    )
    proxy_normal = source_normal[[0, 2, 4]]
    original = geometry_transport_cost(
        source_xyz, proxy_xyz, edge_source, edge_target,
        source_normals=source_normal, proxy_normals=proxy_normal,
    )
    flipped = geometry_transport_cost(
        source_xyz, proxy_xyz, edge_source, edge_target,
        source_normals=-source_normal, proxy_normals=proxy_normal,
    )
    torch.testing.assert_close(original, flipped)


def test_axial_normal_pushforward_is_invariant_to_source_sign() -> None:
    source_xyz, proxy_xyz, mass, *_ = _problem()
    plan = build_full_cloud_assignment(
        source_xyz.numpy(), mass.numpy(), anchor_xyz=proxy_xyz.numpy(), k=2
    )
    generator = torch.Generator().manual_seed(19)
    normal = torch.nn.functional.normalize(
        torch.randn(source_xyz.shape[0], 3, generator=generator), dim=-1
    )
    axis_a = pushforward_axial_normals(plan, normal)
    axis_b = pushforward_axial_normals(plan, -normal)
    # Eigenvector signs are arbitrary; compare the underlying axes.
    torch.testing.assert_close(
        (axis_a * axis_b).sum(-1).abs(), torch.ones(axis_a.shape[0]),
        atol=1e-5, rtol=1e-5,
    )
