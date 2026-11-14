"""Triton segmented reductions for the detached S5 SROT solver.

This module is deliberately experimental: the frozen reference implementation
in :mod:`camo_build.transport` remains the production solver until this backend
passes the paired numerical and latency gate.  It only operates on a fixed
width source-to-proxy support (the layout emitted by ``SparseMeasureCoarsener``)
and never materializes a dense source/proxy matrix.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

try:  # Triton is an optional acceleration backend.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on machines without Triton
    triton = None
    tl = None


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _source_lse_kernel(
        log_kernel_ptr,
        edge_target_ptr,
        log_v_ptr,
        log_mass_ptr,
        output_ptr,
        num_sources,
        row_width,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = row * row_width + tl.arange(0, BLOCK)
        mask = offsets < (row + 1) * row_width
        values = tl.load(log_kernel_ptr + offsets, mask=mask, other=-float("inf"))
        target = tl.load(edge_target_ptr + offsets, mask=mask, other=0)
        values += tl.load(log_v_ptr + target, mask=mask, other=0.0)
        maximum = tl.max(tl.where(mask, values, -float("inf")), axis=0)
        shifted = tl.where(mask, tl.exp(values - maximum), 0.0)
        denominator = tl.sum(shifted, axis=0)
        result = tl.load(log_mass_ptr + row) - (
            maximum + tl.log(tl.maximum(denominator, 1.17549435e-38))
        )
        tl.store(output_ptr + row, result)

    @triton.jit
    def _target_lse_kernel(
        sorted_log_kernel_ptr,
        sorted_source_ptr,
        log_u_ptr,
        offsets_ptr,
        output_ptr,
        num_targets,
        BLOCK: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        target = tl.program_id(0)
        start = tl.load(offsets_ptr + target)
        end = tl.load(offsets_ptr + target + 1)
        maximum = -float("inf")
        for base in tl.range(0, MAX_DEGREE, BLOCK):
            edge = start + base + tl.arange(0, BLOCK)
            mask = edge < end
            values = tl.load(sorted_log_kernel_ptr + edge, mask=mask, other=-float("inf"))
            source = tl.load(sorted_source_ptr + edge, mask=mask, other=0)
            values += tl.load(log_u_ptr + source, mask=mask, other=0.0)
            maximum = tl.maximum(maximum, tl.max(tl.where(mask, values, -float("inf")), axis=0))
        denominator = 0.0
        for base in tl.range(0, MAX_DEGREE, BLOCK):
            edge = start + base + tl.arange(0, BLOCK)
            mask = edge < end
            values = tl.load(sorted_log_kernel_ptr + edge, mask=mask, other=-float("inf"))
            source = tl.load(sorted_source_ptr + edge, mask=mask, other=0)
            values += tl.load(log_u_ptr + source, mask=mask, other=0.0)
            denominator += tl.sum(tl.where(mask, tl.exp(values - maximum), 0.0), axis=0)
        tl.store(
            output_ptr + target,
            maximum + tl.log(tl.maximum(denominator, 1.17549435e-38)),
        )

    @triton.jit
    def _pushforward_kernel(
        feature_ptr,
        sorted_source_ptr,
        sorted_value_ptr,
        offsets_ptr,
        target_mass_ptr,
        output_ptr,
        num_targets,
        feature_dim,
        source_stride,
        feature_stride,
        output_stride,
        BLOCK_E: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_DEGREE: tl.constexpr,
    ):
        target = tl.program_id(0)
        feature_block = tl.program_id(1)
        start = tl.load(offsets_ptr + target)
        end = tl.load(offsets_ptr + target + 1)
        columns = feature_block * BLOCK_D + tl.arange(0, BLOCK_D)
        column_mask = columns < feature_dim
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for base in tl.range(0, MAX_DEGREE, BLOCK_E):
            edge = start + base + tl.arange(0, BLOCK_E)
            edge_mask = edge < end
            source = tl.load(sorted_source_ptr + edge, mask=edge_mask, other=0)
            weight = tl.load(sorted_value_ptr + edge, mask=edge_mask, other=0.0)
            pointers = (
                feature_ptr
                + source[:, None] * source_stride
                + columns[None, :] * feature_stride
            )
            values = tl.load(
                pointers,
                mask=edge_mask[:, None] & column_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(values * weight[:, None], axis=0)
        denominator = tl.load(target_mass_ptr + target)
        result = accumulator / tl.maximum(denominator, 1.0e-12)
        output_ptrs = output_ptr + target * output_stride + columns
        tl.store(output_ptrs, result, mask=column_mask)


def _check_layout(
    source_mass: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_cost: torch.Tensor,
    num_targets: int,
) -> Tuple[int, int]:
    if not available():
        raise RuntimeError("Triton/CUDA backend is unavailable")
    tensors = (source_mass, edge_source, edge_target, edge_cost)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("Triton SROT requires CUDA tensors")
    if source_mass.dtype != torch.float32 or edge_cost.dtype != torch.float32:
        raise TypeError("Triton SROT reductions require FP32 masses and costs")
    if edge_source.dtype != torch.long or edge_target.dtype != torch.long:
        raise TypeError("transport indices must be int64")
    n = int(source_mass.numel())
    e = int(edge_source.numel())
    if e == 0 or e != int(edge_target.numel()) or e != int(edge_cost.numel()):
        raise ValueError("edge arrays have inconsistent lengths")
    if e % n:
        raise ValueError("Triton backend requires fixed-width source rows")
    row_width = e // n
    expected = torch.arange(n, device=edge_source.device, dtype=torch.long)
    expected = expected[:, None].expand(n, row_width).reshape(-1)
    if not torch.equal(edge_source, expected):
        raise ValueError("edge_source must be contiguous fixed-width rows")
    if int(edge_target.min()) < 0 or int(edge_target.max()) >= num_targets:
        raise ValueError("edge_target is outside [0,num_targets)")
    if not torch.isfinite(source_mass).all() or torch.any(source_mass <= 0):
        raise ValueError("source masses must be finite and positive")
    if not torch.isfinite(edge_cost).all():
        raise ValueError("cost contains NaN or Inf")
    return n, row_width


def segmented_logsumexp_source(
    log_kernel: torch.Tensor,
    edge_target: torch.Tensor,
    log_v: torch.Tensor,
    log_mass: torch.Tensor,
    *,
    row_width: int,
) -> torch.Tensor:
    """Compute source-row LogSumExp with one Triton program per row."""
    if log_kernel.ndim != 1 or edge_target.ndim != 1:
        raise ValueError("edge tensors must be vectors")
    output = torch.empty_like(log_mass, dtype=torch.float32)
    block = _next_power_of_two(row_width)
    _source_lse_kernel[(log_mass.numel(),)](
        log_kernel, edge_target, log_v, log_mass, output,
        log_mass.numel(), row_width, BLOCK=block,
        num_warps=1,
    )
    return output


def segmented_logsumexp_target(
    log_kernel: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    log_u: torch.Tensor,
    *,
    num_targets: int,
) -> torch.Tensor:
    """Compute target-column LogSumExp using a temporary CSR edge permutation."""
    order = torch.argsort(edge_target)
    sorted_target = edge_target[order]
    sorted_kernel = log_kernel[order]
    sorted_source = edge_source[order]
    counts = torch.bincount(sorted_target, minlength=num_targets)
    offsets = torch.empty(num_targets + 1, device=edge_target.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return segmented_logsumexp_target_prepared(
        sorted_kernel, sorted_source, offsets, log_u, num_targets=num_targets
    )


def segmented_logsumexp_target_prepared(
    sorted_log_kernel: torch.Tensor,
    sorted_source: torch.Tensor,
    offsets: torch.Tensor,
    log_u: torch.Tensor,
    *,
    num_targets: int,
) -> torch.Tensor:
    """Target reduction when the CSR permutation/offsets are already cached."""
    if offsets.shape != (num_targets + 1,):
        raise ValueError("offsets must have shape [num_targets+1]")
    max_degree = int((offsets[1:] - offsets[:-1]).max())
    if max_degree <= 0:
        raise ValueError("target support contains an empty column")
    output = torch.empty(num_targets, device=sorted_log_kernel.device, dtype=torch.float32)
    _target_lse_kernel[(num_targets,)](
        sorted_log_kernel, sorted_source, log_u, offsets, output,
        num_targets, BLOCK=64, MAX_DEGREE=max_degree,
        num_warps=1,
    )
    return output


def prepare_target_csr(
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    num_targets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.argsort(edge_target)
    sorted_source = edge_source[order]
    sorted_target = edge_target[order]
    counts = torch.bincount(sorted_target, minlength=num_targets)
    offsets = torch.empty(num_targets + 1, device=edge_target.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    if int(counts.min()) <= 0:
        raise ValueError("target support contains an empty column")
    return order, sorted_source, offsets


def pushforward_triton(
    source_feature: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_value: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    num_targets: int,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Memory-bounded gather/reduce pushforward for non-gradient geometry."""
    if source_feature.ndim != 2:
        raise ValueError("source_feature must have shape [N,D]")
    if not available() or source_feature.device.type != "cuda":
        raise RuntimeError("Triton pushforward requires CUDA")
    if prepared is None:
        order, sorted_source, offsets = prepare_target_csr(
            edge_source, edge_target, num_targets
        )
    else:
        order, sorted_source, offsets = prepared
    sorted_value = edge_value.float()[order]
    output = torch.empty(
        (num_targets, source_feature.shape[1]), device=source_feature.device,
        dtype=torch.float32,
    )
    max_degree = int((offsets[1:] - offsets[:-1]).max())
    grid = (num_targets, triton.cdiv(source_feature.shape[1], 64))
    _pushforward_kernel[grid](
        source_feature, sorted_source, sorted_value, offsets, target_mass.float(),
        output, num_targets, source_feature.shape[1], source_feature.stride(0),
        source_feature.stride(1), output.stride(0), BLOCK_E=64, BLOCK_D=64,
        MAX_DEGREE=max_degree, num_warps=1,
    )
    return output


class _AutogradPushforward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, source_feature, edge_source, edge_target, edge_value, target_mass,
        num_targets, order, sorted_source, offsets,
    ):
        ctx.save_for_backward(edge_source, edge_target, edge_value.float(), target_mass.float())
        ctx.num_targets = int(num_targets)
        ctx.source_count = int(source_feature.shape[0])
        return pushforward_triton(
            source_feature, edge_source, edge_target, edge_value, target_mass,
            num_targets=int(num_targets),
            prepared=(order, sorted_source, offsets),
        )

    @staticmethod
    def backward(ctx, grad_output):
        edge_source, edge_target, edge_value, target_mass = ctx.saved_tensors
        denominator = target_mass[edge_target].clamp_min(1e-12)
        weighted = grad_output.float()[edge_target] * (
            edge_value / denominator
        ).reshape(-1, 1)
        from .sparse_ops import segment_sum
        grad_source = segment_sum(weighted, edge_source, ctx.source_count)
        return grad_source, None, None, None, None, None, None, None, None, None


def pushforward_triton_autograd(
    source_feature: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_value: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    num_targets: int,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prepared is None:
        prepared = prepare_target_csr(edge_source, edge_target, num_targets)
    order, sorted_source, offsets = prepared
    return _AutogradPushforward.apply(
        source_feature, edge_source, edge_target, edge_value, target_mass,
        num_targets, order, sorted_source, offsets,
    )


def semi_relaxed_sparse_transport_triton(
    source_mass: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    cost: torch.Tensor,
    *,
    num_targets: int,
    epsilon: float = 0.1,
    tau: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    log_kernel_clip: float = 80.0,
    target_backend: str = "triton",
) -> tuple[torch.Tensor, torch.Tensor, int, bool, float]:
    """Solve the frozen SROT dual iterations with Triton segmented reductions.

    The returned tuple is ``(plan_value, target_mass, iterations, converged,
    final_log_scaling_change)``.
    Final diagnostics and the detached plan wrapper are intentionally left to
    the reference Python solver until this backend passes the S5 gate.
    """
    if epsilon <= 0 or tau < 0 or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("invalid SROT hyperparameter")
    if target_backend not in {"triton", "native"}:
        raise ValueError("target_backend must be 'triton' or 'native'")
    n, row_width = _check_layout(
        source_mass, edge_source, edge_target, cost, num_targets
    )
    mass = source_mass.detach().float()
    edge_cost = cost.detach().float()
    tiny = torch.finfo(torch.float32).tiny
    total_mass = mass.sum()
    nu = torch.full(
        (num_targets,), total_mass / float(num_targets),
        device=mass.device, dtype=torch.float32,
    )
    edge_nu = nu[edge_target]
    # This matches the reference solver's semi-relaxed KL base measure.
    row_nu = torch.zeros(n, device=mass.device, dtype=torch.float32)
    row_nu.scatter_add_(0, edge_source, edge_nu)
    reference = mass[edge_source] * edge_nu / row_nu[edge_source].clamp_min(1e-12)
    scaled_cost = (edge_cost / float(epsilon)).clamp(
        min=-log_kernel_clip, max=log_kernel_clip
    )
    log_kernel = torch.log(reference.clamp_min(tiny)) - scaled_cost
    log_mass = torch.log(mass)
    log_nu = torch.log(nu)
    log_v = torch.zeros_like(nu)
    exponent = float(tau / (epsilon + tau)) if tau > 0 else 0.0
    # Target support is static during all dual iterations; build CSR once to
    # avoid an argsort/bincount allocation on every iteration.
    order = torch.argsort(edge_target)
    sorted_source = edge_source[order]
    sorted_target = edge_target[order]
    counts = torch.bincount(sorted_target, minlength=num_targets)
    offsets = torch.empty(num_targets + 1, device=edge_target.device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    max_degree = int(counts.max())
    if max_degree <= 0:
        raise ValueError("target support contains an empty column")
    target_block = _next_power_of_two(max_degree)
    sorted_log_kernel = log_kernel[order]
    converged = False
    change = float("inf")
    for iteration in range(1, max_iterations + 1):
        log_u = segmented_logsumexp_source(
            log_kernel, edge_target, log_v, log_mass, row_width=row_width
        )
        if target_backend == "triton":
            log_ktu = segmented_logsumexp_target_prepared(
                sorted_log_kernel, sorted_source, offsets, log_u,
                num_targets=num_targets,
            )
        else:
            from .sparse_ops import segment_logsumexp
            log_ktu = segment_logsumexp(
                log_kernel + log_u[edge_source], edge_target, num_targets
            )
        new_log_v = exponent * (log_nu - log_ktu)
        change = float(torch.max(torch.abs(new_log_v - log_v)).item())
        log_v = new_log_v
        if change <= tolerance:
            converged = True
            break
    # Final row-normalized plan is computed in FP32 and corrected exactly as
    # in the reference solver.  No dense tensor is created here.
    log_u = segmented_logsumexp_source(
        log_kernel, edge_target, log_v, log_mass, row_width=row_width
    )
    log_plan = log_kernel + log_u[edge_source] + log_v[edge_target]
    log_row = -segmented_logsumexp_source(
        log_plan, edge_target, torch.zeros_like(log_v), torch.zeros_like(log_mass),
        row_width=row_width,
    )
    log_plan = log_plan + log_mass[edge_source] - log_row[edge_source]
    plan = torch.exp(log_plan)
    row_mass = torch.zeros_like(mass)
    row_mass.scatter_add_(0, edge_source, plan)
    plan = plan * (mass / row_mass.clamp_min(tiny))[edge_source]
    target_mass = torch.zeros(num_targets, device=mass.device, dtype=torch.float32)
    target_mass.scatter_add_(0, edge_target, plan)
    if not torch.isfinite(plan).all() or torch.any(target_mass <= 0):
        raise FloatingPointError("Triton SROT produced an invalid plan")
    return plan.detach(), target_mass.detach(), iteration, converged, change
