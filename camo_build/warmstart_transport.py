"""Experimental warm-start SROT for a bounded latency study.

Warm-starting the dual variable changes only the initial point of the same
fixed-point iteration.  The final plan is still accepted only after the
frozen tolerance and exact row-marginal correction pass.  This module is not
the production solver until the varying-batch benchmark passes.
"""
from __future__ import annotations

import torch

from .sparse_ops import segment_logsumexp, segment_sum
from .transport import SparseTransportPlan, TransportDiagnostics


def semi_relaxed_sparse_transport_warm(
    source_mass: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    cost: torch.Tensor,
    *,
    num_targets: int,
    initial_log_v: torch.Tensor | None = None,
    epsilon: float = 0.1,
    tau: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
) -> tuple[SparseTransportPlan, TransportDiagnostics, torch.Tensor]:
    """Reference SROT iteration with an optional detached dual warm-start."""

    if epsilon <= 0 or tau < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid SROT hyperparameter")
    device = source_mass.device
    mass = source_mass.detach().to(device=device, dtype=torch.float32)
    source = edge_source.to(device=device, dtype=torch.long)
    target = edge_target.to(device=device, dtype=torch.long)
    edge_cost = cost.detach().to(device=device, dtype=torch.float32)
    if mass.ndim != 1 or source.ndim != 1 or target.ndim != 1:
        raise ValueError("masses and edge arrays must be vectors")
    if source.numel() == 0 or not (
        source.numel() == target.numel() == edge_cost.numel()
    ):
        raise ValueError("edge arrays are empty or inconsistent")
    if not torch.isfinite(mass).all() or torch.any(mass <= 0):
        raise ValueError("source masses must be finite and positive")
    if not torch.isfinite(edge_cost).all():
        raise ValueError("cost contains NaN or Inf")
    n = mass.numel()
    if int(source.min()) < 0 or int(source.max()) >= n:
        raise ValueError("source edge index is invalid")
    if int(target.min()) < 0 or int(target.max()) >= num_targets:
        raise ValueError("target edge index is invalid")
    source_degree = torch.bincount(source, minlength=n)
    target_degree = torch.bincount(target, minlength=num_targets)
    if torch.any(source_degree == 0) or torch.any(target_degree == 0):
        raise ValueError("transport support has an empty row or column")

    total_mass = mass.sum()
    nu = torch.full(
        (num_targets,), total_mass / num_targets,
        device=device, dtype=torch.float32,
    )
    edge_nu = nu[target]
    row_nu = segment_sum(edge_nu, source, n)
    reference = mass[source] * edge_nu / row_nu[source].clamp_min(1e-12)
    tiny = torch.finfo(torch.float32).tiny
    log_kernel = torch.log(reference.clamp_min(tiny)) - (
        edge_cost / float(epsilon)
    ).clamp(min=-80.0, max=80.0)
    log_mass = torch.log(mass)
    log_nu = torch.log(nu)
    if initial_log_v is None:
        log_v = torch.zeros_like(nu)
    else:
        log_v = initial_log_v.detach().to(device=device, dtype=torch.float32)
        if log_v.shape != nu.shape or not torch.isfinite(log_v).all():
            raise ValueError("initial_log_v must be finite with shape [num_targets]")
    exponent = float(tau / (epsilon + tau)) if tau > 0 else 0.0
    converged = False
    maximum_change = float("inf")
    for iteration in range(1, max_iterations + 1):
        log_kv = segment_logsumexp(log_kernel + log_v[target], source, n)
        if not torch.isfinite(log_kv).all():
            raise FloatingPointError("warm SROT encountered a non-finite source segment")
        log_u = log_mass - log_kv
        log_ktu = segment_logsumexp(
            log_kernel + log_u[source], target, num_targets
        )
        if not torch.isfinite(log_ktu).all():
            raise FloatingPointError("warm SROT encountered a non-finite target segment")
        new_log_v = exponent * (log_nu - log_ktu)
        maximum_change = float(torch.max(torch.abs(new_log_v - log_v)))
        log_v = new_log_v
        if maximum_change <= tolerance:
            converged = True
            break

    log_kv = segment_logsumexp(log_kernel + log_v[target], source, n)
    log_u = log_mass - log_kv
    log_plan = log_kernel + log_u[source] + log_v[target]
    log_row = segment_logsumexp(log_plan, source, n)
    log_plan = log_plan + log_mass[source] - log_row[source]
    value = torch.exp(log_plan)
    row_mass = segment_sum(value, source, n)
    value = value * (mass / row_mass.clamp_min(tiny))[source]
    row_mass = segment_sum(value, source, n)
    target_mass = segment_sum(value, target, num_targets)
    if not torch.isfinite(value).all() or torch.any(target_mass <= 0):
        raise FloatingPointError("warm SROT produced an invalid plan")

    absolute_residual = torch.abs(target_mass.sum() - total_mass)
    relative_residual = absolute_residual / total_mass.clamp_min(1e-12)
    row_relative = torch.max(torch.abs(row_mass - mass) / mass.clamp_min(1e-12))
    target_kl = (
        target_mass * (torch.log(target_mass.clamp_min(tiny)) - torch.log(nu))
        - target_mass + nu
    ).sum()
    plan_kl = (
        value * (torch.log(value.clamp_min(tiny)) - torch.log(reference))
        - value + reference
    ).sum()
    normalized = value / total_mass.clamp_min(1e-12)
    entropy = -(normalized * torch.log(normalized.clamp_min(tiny))).sum()
    objective = (value * edge_cost).sum() + epsilon * plan_kl + tau * target_kl
    plan = SparseTransportPlan(
        source, target, value.detach(), mass.detach(), target_mass.detach()
    )
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
    return plan, diagnostics, log_v.detach()
