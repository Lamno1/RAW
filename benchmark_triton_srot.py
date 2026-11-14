"""Correctness and latency smoke test for the experimental Triton SROT path."""
from __future__ import annotations

import json
import time

import torch

from camo_build.transport import semi_relaxed_sparse_transport
from camo_build.triton_transport import (
    available,
    segmented_logsumexp_source,
    segmented_logsumexp_target,
    semi_relaxed_sparse_transport_triton,
)
from camo_build.sparse_ops import segment_logsumexp, segment_sum


def main() -> None:
    if not available():
        raise RuntimeError("Triton/CUDA is unavailable")
    torch.manual_seed(2026)
    device = torch.device("cuda")
    n, m, k = 256, 64, 8
    source = torch.arange(n, device=device, dtype=torch.long)[:, None].expand(n, k).reshape(-1)
    target = torch.randint(0, m, (n * k,), device=device, dtype=torch.long)
    target[:m] = torch.arange(m, device=device)
    mass = torch.rand(n, device=device, dtype=torch.float32) + 0.05
    cost = torch.rand(n * k, device=device, dtype=torch.float32)

    # Reduction-level audit before comparing the complete solver.
    total_mass = mass.sum()
    nu = torch.full((m,), total_mass / float(m), device=device)
    edge_nu = nu[target]
    row_nu = segment_sum(edge_nu, source, n)
    reference = mass[source] * edge_nu / row_nu[source].clamp_min(1e-12)
    log_kernel = torch.log(reference.clamp_min(torch.finfo(torch.float32).tiny)) - cost / 0.2
    log_v0 = torch.zeros_like(nu)
    log_mass = torch.log(mass)
    ref_source_lse = segment_logsumexp(log_kernel, source, n)
    tri_source_logkv = -segmented_logsumexp_source(
        log_kernel, target, log_v0, torch.zeros_like(log_mass), row_width=k
    )
    source_error = float((ref_source_lse - tri_source_logkv).abs().max().item())
    ref_target_lse = segment_logsumexp(
        log_kernel + (log_mass - ref_source_lse)[source], target, m
    )
    tri_target_lse = segmented_logsumexp_target(
        log_kernel, source, target, log_mass - tri_source_logkv, num_targets=m
    )
    target_error = float((ref_target_lse - tri_target_lse).abs().max().item())
    if source_error > 2e-5 or target_error > 2e-5:
        raise AssertionError(f"reduction mismatch source={source_error} target={target_error}")

    ref_plan, ref_diag = semi_relaxed_sparse_transport(
        mass, source, target, cost, num_targets=m,
        epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-5,
    )
    # Warm up compilation and compare the exact detached sparse output.
    tri_plan, tri_target, tri_iters, tri_converged, tri_change = semi_relaxed_sparse_transport_triton(
        mass, source, target, cost, num_targets=m,
        epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-5,
    )
    max_abs = float((tri_plan - ref_plan.value).abs().max().item())
    mass_residual = float((tri_target.sum() - mass.sum()).abs().item())
    print({"source_reduction_error": source_error, "target_reduction_error": target_error,
           "ref_iters": ref_diag.iterations, "tri_iters": tri_iters,
           "ref_converged": ref_diag.converged, "tri_converged": tri_converged,
           "max_abs": max_abs})
    if max_abs > 3e-4 or mass_residual > 1e-5 or tri_converged != ref_diag.converged:
        raise AssertionError(
            f"Triton mismatch: max_abs={max_abs}, residual={mass_residual}, "
            f"converged={tri_converged}/{ref_diag.converged}"
        )
    for _ in range(5):
        semi_relaxed_sparse_transport_triton(
            mass, source, target, cost, num_targets=m,
            epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-5,
        )
    torch.cuda.synchronize()
    begin = time.perf_counter()
    repeats = 20
    for _ in range(repeats):
        semi_relaxed_sparse_transport_triton(
            mass, source, target, cost, num_targets=m,
            epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-5,
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - begin) / repeats
    report = {
        "torch": torch.__version__,
        "triton_available": available(),
        "n": n,
        "m": m,
        "k": k,
        "reference_iterations": ref_diag.iterations,
        "triton_iterations": tri_iters,
        "reference_converged": ref_diag.converged,
        "triton_converged": tri_converged,
        "max_abs_plan_error": max_abs,
        "absolute_mass_residual": mass_residual,
        "triton_mean_seconds": elapsed,
        "passed": True,
    }
    print(json.dumps(report, indent=2))
    with open("triton_srot_smoke_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
