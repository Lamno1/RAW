"""Sparse semi-relaxed optimal transport for CAMO-Pyramid.

For fixed sparse support ``E`` the optimized problem is

    min_P <P,C> + eps KL(P || R) + tau KL(P^T 1 || nu)
    subject to P 1 = m, P >= 0.

All solver state is FP32 and represented by edge vectors. No dense ``[N,M]``
tensor is constructed. The returned plan is detached by default for Prototype
2a/S4, as required by the research blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .sparse_ops import segment_logsumexp, segment_sum


@dataclass(frozen=True)
class TransportDiagnostics:
    iterations: int
    converged: bool
    absolute_mass_residual: float
    relative_mass_residual: float
    max_relative_row_residual: float
    target_kl: float
    entropy: float
    objective: float
    max_log_scaling_change: float


@dataclass(frozen=True)
class SparseTransportPlan:
    """A mass plan ``P`` stored as coalesced COO edge vectors."""

    source_index: torch.Tensor
    target_index: torch.Tensor
    value: torch.Tensor
    source_mass: torch.Tensor
    target_mass: torch.Tensor

    def __post_init__(self) -> None:
        if self.source_index.ndim != 1 or self.target_index.ndim != 1:
            raise ValueError("transport indices must be vectors")
        if not (
            self.source_index.numel()
            == self.target_index.numel()
            == self.value.numel()
        ):
            raise ValueError("transport edge arrays have inconsistent lengths")
        if self.source_index.dtype != torch.long or self.target_index.dtype != torch.long:
            raise TypeError("transport indices must be int64")
        if self.source_mass.ndim != 1 or self.target_mass.ndim != 1:
            raise ValueError("source_mass and target_mass must be vectors")
        if not torch.isfinite(self.value).all() or torch.any(self.value < 0):
            raise FloatingPointError("transport plan contains invalid values")
        if torch.any(self.source_mass <= 0) or torch.any(self.target_mass <= 0):
            raise FloatingPointError("transport plan requires positive active masses")

    @property
    def edge_count(self) -> int:
        return self.value.numel()

    @property
    def relative_mass_residual(self) -> float:
        residual = torch.abs(self.target_mass.sum() - self.source_mass.sum())
        return float(residual / self.source_mass.sum().clamp_min(1e-12))

    def row_mass(self) -> torch.Tensor:
        return segment_sum(self.value, self.source_index, self.source_mass.numel())

    def pushforward(self, source_feature: torch.Tensor) -> torch.Tensor:
        """Apply ``D_target^-1 P^T`` to source features."""
        if source_feature.shape[0] != self.source_mass.numel():
            raise ValueError("source feature count does not match plan")
        edge_value = self.value.to(source_feature.device)
        source_index = self.source_index.to(source_feature.device)
        target_index = self.target_index.to(source_feature.device)
        feature = source_feature.float()
        weight_shape = (edge_value.numel(),) + (1,) * (feature.ndim - 1)
        weighted = feature[source_index] * edge_value.reshape(weight_shape)
        numerator = segment_sum(weighted, target_index, self.target_mass.numel())
        target_mass = self.target_mass.to(feature.device).float()
        denominator_shape = (target_mass.numel(),) + (1,) * (feature.ndim - 1)
        if torch.any(target_mass <= 1e-12):
            raise FloatingPointError("inactive target reached pushforward")
        return numerator / target_mass.reshape(denominator_shape)

    def pullback(self, target_feature: torch.Tensor) -> torch.Tensor:
        """Apply the weighted adjoint ``D_source^-1 P``."""
        if target_feature.shape[0] != self.target_mass.numel():
            raise ValueError("target feature count does not match plan")
        edge_value = self.value.to(target_feature.device)
        source_index = self.source_index.to(target_feature.device)
        target_index = self.target_index.to(target_feature.device)
        feature = target_feature.float()
        weight_shape = (edge_value.numel(),) + (1,) * (feature.ndim - 1)
        weighted = feature[target_index] * edge_value.reshape(weight_shape)
        numerator = segment_sum(weighted, source_index, self.source_mass.numel())
        source_mass = self.source_mass.to(feature.device).float()
        denominator_shape = (source_mass.numel(),) + (1,) * (feature.ndim - 1)
        if torch.any(source_mass <= 1e-12):
            raise FloatingPointError("inactive source reached pullback")
        return numerator / source_mass.reshape(denominator_shape)

    def adjoint_error(
        self, source_feature: torch.Tensor, target_feature: torch.Tensor
    ) -> float:
        """Return relative error in the mass-weighted adjoint identity."""
        encoded = self.pushforward(source_feature)
        decoded = self.pullback(target_feature)
        target_mass = self.target_mass.to(encoded.device).float()
        source_mass = self.source_mass.to(decoded.device).float()
        expand_target = (target_mass.numel(),) + (1,) * (encoded.ndim - 1)
        expand_source = (source_mass.numel(),) + (1,) * (decoded.ndim - 1)
        left = (encoded * target_feature.float() * target_mass.reshape(expand_target)).sum()
        right = (source_feature.float() * decoded * source_mass.reshape(expand_source)).sum()
        scale = torch.maximum(left.abs(), right.abs()).clamp_min(1.0)
        return float((left - right).abs() / scale)


def pushforward_axial_normals(
    plan: SparseTransportPlan,
    source_normals: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Push unoriented normals through ``P`` via second moments ``nn^T``.

    The principal eigenvector is an axis, not an oriented vector. Its arbitrary
    sign is safe only with the frozen sign-invariant normal descriptors/cost.
    """
    if source_normals.shape != (plan.source_mass.numel(), 3):
        raise ValueError("source_normals must have shape [N,3]")
    normal = torch.nn.functional.normalize(
        source_normals.float(), dim=-1, eps=epsilon
    )
    second_moment = normal[:, :, None] * normal[:, None, :]
    target_moment = plan.pushforward(second_moment)
    if not torch.isfinite(target_moment).all():
        raise FloatingPointError("normal second-moment pushforward is non-finite")
    target_moment = 0.5 * (
        target_moment + target_moment.transpose(-1, -2)
    )
    _, eigenvectors = torch.linalg.eigh(target_moment)
    axis = eigenvectors[..., -1]
    axis = torch.nn.functional.normalize(axis, dim=-1, eps=epsilon)
    if not torch.isfinite(axis).all():
        raise FloatingPointError("principal normal axis is non-finite")
    return axis


def geometry_transport_cost(
    source_xyz: torch.Tensor,
    proxy_xyz: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    *,
    source_normals: torch.Tensor | None = None,
    proxy_normals: torch.Tensor | None = None,
    distance_weight: float = 1.0,
    normal_weight: float = 0.25,
    sign_invariant_normals: bool = True,
    epsilon: float = 1e-6,
    clip: float = 20.0,
) -> torch.Tensor:
    """Robust geometry-only SROT cost on an existing sparse support.

    The default normal term is ``1-|n_i^T n_j|`` because the S4 audit did not
    establish globally consistent normal signs. Signed normals must be enabled
    explicitly after a complete audit.
    """
    if distance_weight < 0 or normal_weight < 0 or epsilon <= 0 or clip <= 0:
        raise ValueError("invalid geometry cost hyperparameter")
    source = edge_source.to(source_xyz.device).long()
    target = edge_target.to(source_xyz.device).long()
    squared_distance = (
        source_xyz.float()[source] - proxy_xyz.float().to(source_xyz.device)[target]
    ).square().sum(-1)

    def robust_normalize(value: torch.Tensor) -> torch.Tensor:
        median = value.median()
        mad = (value - median).abs().median().clamp_min(epsilon)
        return ((value - median) / mad).clamp(min=-clip, max=clip)

    cost = distance_weight * robust_normalize(squared_distance)
    if normal_weight > 0.0:
        if source_normals is None or proxy_normals is None:
            raise ValueError("normal tensors are required when normal_weight > 0")
        source_normal = torch.nn.functional.normalize(
            source_normals.float(), dim=-1, eps=epsilon
        )[source]
        proxy_normal = torch.nn.functional.normalize(
            proxy_normals.float().to(source_xyz.device), dim=-1, eps=epsilon
        )[target]
        dot = (source_normal * proxy_normal).sum(-1).clamp(-1.0, 1.0)
        if sign_invariant_normals:
            dot = dot.abs()
        normal_cost = 1.0 - dot
        cost = cost + normal_weight * robust_normalize(normal_cost)
    # A global additive constant does not change the optimum under the hard
    # source marginal and makes the reported geometry cost non-negative.
    return (cost - cost.min()).clamp(max=clip).float()


def build_sparse_proxy_support(
    source_xyz: np.ndarray,
    proxy_xyz: np.ndarray,
    *,
    k: int = 8,
    reciprocal: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build forward kNN plus bounded reciprocal coverage edges on CPU."""
    source_xyz = np.asarray(source_xyz, dtype=np.float32)
    proxy_xyz = np.asarray(proxy_xyz, dtype=np.float32)
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz must have shape [N,3]")
    if proxy_xyz.ndim != 2 or proxy_xyz.shape[1] != 3:
        raise ValueError("proxy_xyz must have shape [M,3]")
    if not np.isfinite(source_xyz).all() or not np.isfinite(proxy_xyz).all():
        raise ValueError("coordinates contain NaN or Inf")
    n, m = len(source_xyz), len(proxy_xyz)
    if not 0 < k <= m or not 0 <= reciprocal <= n:
        raise ValueError("invalid k or reciprocal coverage")

    _, target = cKDTree(proxy_xyz).query(source_xyz, k=k, workers=-1)
    if k == 1:
        target = target[:, None]
    source = np.repeat(np.arange(n, dtype=np.int64), k)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if reciprocal:
        _, reciprocal_source = cKDTree(source_xyz).query(
            proxy_xyz, k=reciprocal, workers=-1
        )
        if reciprocal == 1:
            reciprocal_source = reciprocal_source[:, None]
        source = np.concatenate(
            (source, np.asarray(reciprocal_source, dtype=np.int64).reshape(-1))
        )
        target = np.concatenate(
            (target, np.repeat(np.arange(m, dtype=np.int64), reciprocal))
        )

    encoded = source * m + target
    encoded = np.unique(encoded)
    source = encoded // m
    target = encoded % m
    if np.unique(source).size != n:
        raise RuntimeError("sparse support left a source without an edge")
    if np.unique(target).size != m:
        raise RuntimeError("sparse support left a proxy without an edge")
    return torch.from_numpy(source).long(), torch.from_numpy(target).long()


def _generalized_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    return (p * (torch.log(p.clamp_min(tiny)) - torch.log(q.clamp_min(tiny))) - p + q).sum()


def semi_relaxed_sparse_transport(
    source_mass: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    cost: torch.Tensor,
    *,
    num_targets: int,
    target_reference: torch.Tensor | None = None,
    epsilon: float = 0.1,
    tau: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    log_kernel_clip: float = 80.0,
    detach_plan: bool = True,
) -> tuple[SparseTransportPlan, TransportDiagnostics]:
    """Solve fixed-support semi-relaxed OT with a hard source marginal."""
    if not detach_plan:
        raise NotImplementedError(
            "differentiable SROT is forbidden before the detached S5 gate"
        )
    if epsilon <= 0.0 or tau < 0.0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid SROT hyperparameter")
    if num_targets <= 0 or log_kernel_clip <= 0.0:
        raise ValueError("num_targets and log_kernel_clip must be positive")
    device = source_mass.device
    mass = source_mass.detach().to(dtype=torch.float32)
    source = edge_source.to(device=device, dtype=torch.long)
    target = edge_target.to(device=device, dtype=torch.long)
    edge_cost = cost.detach().to(device=device, dtype=torch.float32)
    if mass.ndim != 1 or source.ndim != 1 or target.ndim != 1:
        raise ValueError("masses and edge arrays must be vectors")
    if not (source.numel() == target.numel() == edge_cost.numel()):
        raise ValueError("edge arrays have inconsistent lengths")
    if not torch.isfinite(mass).all() or torch.any(mass <= 0):
        raise ValueError("source masses must be finite and positive")
    if not torch.isfinite(edge_cost).all():
        raise ValueError("cost contains NaN or Inf")
    n = mass.numel()
    if source.numel() == 0 or int(source.min()) < 0 or int(source.max()) >= n:
        raise ValueError("source edge index is invalid")
    if int(target.min()) < 0 or int(target.max()) >= num_targets:
        raise ValueError("target edge index is invalid")
    source_degree = torch.bincount(source, minlength=n)
    target_degree = torch.bincount(target, minlength=num_targets)
    if torch.any(source_degree == 0):
        raise ValueError("source support contains an empty row")
    if torch.any(target_degree == 0):
        raise ValueError("target support contains an empty column")

    total_mass = mass.sum()
    if target_reference is None:
        nu = torch.full(
            (num_targets,), float(total_mass / num_targets),
            device=device, dtype=torch.float32,
        )
    else:
        nu = target_reference.detach().to(device=device, dtype=torch.float32)
        if nu.shape != (num_targets,) or not torch.isfinite(nu).all() or torch.any(nu <= 0):
            raise ValueError("target_reference must be a positive [M] vector")
        nu = nu * (total_mass / nu.sum().clamp_min(1e-12))

    edge_nu = nu[target]
    row_nu = segment_sum(edge_nu, source, n)
    reference = mass[source] * edge_nu / row_nu[source].clamp_min(1e-12)
    tiny = torch.finfo(torch.float32).tiny
    scaled_cost = (edge_cost / float(epsilon)).clamp(
        min=-log_kernel_clip, max=log_kernel_clip
    )
    log_kernel = torch.log(reference.clamp_min(tiny)) - scaled_cost
    log_mass = torch.log(mass)
    log_nu = torch.log(nu)
    log_v = torch.zeros_like(nu)
    exponent = float(tau / (epsilon + tau)) if tau > 0.0 else 0.0
    converged = False
    maximum_change = float("inf")

    for iteration in range(1, max_iterations + 1):
        log_kv = segment_logsumexp(log_kernel + log_v[target], source, n)
        if not torch.isfinite(log_kv).all():
            raise FloatingPointError("SROT encountered an empty/non-finite source segment")
        log_u = log_mass - log_kv
        log_ktu = segment_logsumexp(log_kernel + log_u[source], target, num_targets)
        if not torch.isfinite(log_ktu).all():
            raise FloatingPointError("SROT encountered an empty/non-finite target segment")
        new_log_v = exponent * (log_nu - log_ktu)
        maximum_change = float(torch.max(torch.abs(new_log_v - log_v)))
        log_v = new_log_v
        if maximum_change <= tolerance:
            converged = True
            break

    # Recompute u with the final v, then restore the hard row marginal exactly
    # in log space. This guard is part of the mathematical contract.
    log_kv = segment_logsumexp(log_kernel + log_v[target], source, n)
    log_u = log_mass - log_kv
    log_plan = log_kernel + log_u[source] + log_v[target]
    log_row = segment_logsumexp(log_plan, source, n)
    log_plan = log_plan + log_mass[source] - log_row[source]
    plan_value = torch.exp(log_plan)
    row_mass = segment_sum(plan_value, source, n)
    # One FP32 multiplicative correction reduces the residual from exp rounding.
    plan_value = plan_value * (mass / row_mass.clamp_min(tiny))[source]
    row_mass = segment_sum(plan_value, source, n)
    target_mass = segment_sum(plan_value, target, num_targets)
    if not torch.isfinite(plan_value).all() or torch.any(target_mass <= 0):
        raise FloatingPointError("SROT produced a non-finite or inactive transport plan")

    absolute_residual = torch.abs(target_mass.sum() - total_mass)
    relative_residual = absolute_residual / total_mass.clamp_min(1e-12)
    row_relative = torch.max(torch.abs(row_mass - mass) / mass.clamp_min(1e-12))
    target_kl = _generalized_kl(target_mass, nu)
    plan_kl = _generalized_kl(plan_value, reference)
    normalized_plan = plan_value / total_mass.clamp_min(1e-12)
    entropy = -(
        normalized_plan * torch.log(normalized_plan.clamp_min(tiny))
    ).sum()
    objective = (plan_value * edge_cost).sum() + epsilon * plan_kl + tau * target_kl

    plan_value = plan_value.detach()
    target_mass = target_mass.detach()
    mass = mass.detach()
    plan = SparseTransportPlan(source, target, plan_value, mass, target_mass)
    diagnostics = TransportDiagnostics(
        iterations=iteration,
        converged=converged,
        absolute_mass_residual=float(absolute_residual),
        relative_mass_residual=float(relative_residual),
        max_relative_row_residual=float(row_relative),
        target_kl=float(target_kl),
        entropy=float(entropy),
        objective=float(objective),
        max_log_scaling_change=maximum_change,
    )
    return plan, diagnostics
