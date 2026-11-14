"""Behavioral tests for the frozen CAMO-Build Prototype 1 specification."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from camo_build.geometry import gravity_invariants
from camo_build.losses import StaticBalancedCrossEntropy
from camo_build.measure import estimate_surface_mass
from camo_build.model import CAMOPrototype1
from camo_build.neighbors import scipy_knn_indices


def sample_cloud(batch: int = 2, points: int = 48) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(batch, points, 3, generator=generator)
    normals = torch.nn.functional.normalize(
        torch.randn(batch, points, 3, generator=generator), dim=-1
    )
    rgb = torch.rand(batch, points, 3, generator=generator)
    features = torch.cat((xyz, normals, rgb), dim=-1)
    graph = torch.stack(
        [
            torch.from_numpy(scipy_knn_indices(cloud.numpy(), 8))
            for cloud in xyz
        ]
    )
    mass = torch.rand(batch, points, generator=generator) + 0.1
    mass = mass / mass.mean(dim=1, keepdim=True)
    return xyz, normals, features, graph.long(), mass


def test_knn_excludes_self() -> None:
    xyz = np.random.default_rng(1).normal(size=(32, 3)).astype(np.float32)
    graph = scipy_knn_indices(xyz, 5)
    assert graph.shape == (32, 5)
    assert not np.any(graph == np.arange(32)[:, None])


def test_mass_is_finite_positive_and_mean_one() -> None:
    xyz = np.random.default_rng(2).normal(size=(128, 3)).astype(np.float32)
    mass = estimate_surface_mass(xyz, k=8)
    assert np.isfinite(mass).all()
    assert np.all(mass > 0.0)
    assert float(mass.mean()) == pytest.approx(1.0, abs=1e-5)


def test_gravity_invariants_survive_z_rotation_and_translation() -> None:
    xyz, normals, _, graph, _ = sample_cloud(batch=1)
    angle = 0.73
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=xyz.dtype,
    )
    translation = torch.tensor([3.0, -2.0, 4.0])
    original = gravity_invariants(xyz, normals, graph)
    transformed = gravity_invariants(
        xyz @ rotation.T + translation,
        normals @ rotation.T,
        graph,
    )
    torch.testing.assert_close(original, transformed, rtol=1e-4, atol=1e-5)


def test_model_is_finite_and_backpropagates() -> None:
    _, _, features, graph, mass = sample_cloud()
    model = CAMOPrototype1(width=32, mass_mode="density")
    logits = model(features, mass, graph)
    assert logits.shape == (2, 48, 31)
    logits.square().mean().backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_none_and_uniform_controls_are_equivalent() -> None:
    _, _, features, graph, mass = sample_cloud()
    none_model = CAMOPrototype1(width=16, mass_mode="none")
    uniform_model = CAMOPrototype1(width=16, mass_mode="uniform")
    uniform_model.load_state_dict(none_model.state_dict())
    none_model.eval()
    uniform_model.eval()
    with torch.no_grad():
        none_logits = none_model(features, mass, graph)
        uniform_logits = uniform_model(features, mass, graph)
    torch.testing.assert_close(none_logits, uniform_logits, rtol=1e-5, atol=1e-6)


def test_static_balanced_ce_matches_manual_equation() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [3.0, -1.0]]])
    target = torch.tensor([[0, 1, -100]])
    weights = torch.tensor([1.0, 2.0])
    loss_fn = StaticBalancedCrossEntropy(weights)
    loss = loss_fn(logits, target)
    point_loss = torch.nn.functional.cross_entropy(
        logits[0, :2], target[0, :2], reduction="none"
    )
    expected = (point_loss * weights[target[0, :2]]).sum() / 3.0
    torch.testing.assert_close(loss, expected)


def test_static_balanced_ce_rejects_all_ignore_batch() -> None:
    loss_fn = StaticBalancedCrossEntropy(torch.ones(3))
    with pytest.raises(ValueError, match="no valid"):
        loss_fn(torch.randn(2, 4, 3), torch.full((2, 4), -100))

