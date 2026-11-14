"""S4 assignment/SROT mathematical and hardware gate."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.assignment import build_full_cloud_assignment, select_spatial_anchors
from camo_build.transport import (
    build_sparse_proxy_support,
    geometry_transport_cost,
    semi_relaxed_sparse_transport,
)


def _repeat_disjoint_graph(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    batch: int,
    num_source: int,
    num_target: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated_source = torch.cat([source + item * num_source for item in range(batch)])
    repeated_target = torch.cat([target + item * num_target for item in range(batch)])
    repeated_cost = cost.repeat(batch)
    return (
        repeated_source.to(device), repeated_target.to(device), repeated_cost.to(device)
    )


def _srot_gate(
    num_source: int,
    num_target: int,
    batch: int,
    proxy_k: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    generator = np.random.default_rng(2026 + num_source)
    source_xyz = generator.normal(size=(num_source, 3)).astype(np.float32)
    proxy_index = generator.choice(num_source, num_target, replace=False)
    proxy_xyz = source_xyz[proxy_index]
    source_normal = generator.normal(size=(num_source, 3)).astype(np.float32)
    source_normal /= np.maximum(
        np.linalg.norm(source_normal, axis=1, keepdims=True), 1e-6
    )
    proxy_normal = source_normal[proxy_index]
    source, target = build_sparse_proxy_support(
        source_xyz, proxy_xyz, k=proxy_k, reciprocal=1
    )
    xyz_tensor = torch.from_numpy(source_xyz)
    proxy_tensor = torch.from_numpy(proxy_xyz)
    cost = geometry_transport_cost(
        xyz_tensor,
        proxy_tensor,
        source,
        target,
        source_normals=torch.from_numpy(source_normal),
        proxy_normals=torch.from_numpy(proxy_normal),
        sign_invariant_normals=True,
    )
    source, target, cost = _repeat_disjoint_graph(
        source, target, cost, batch=batch,
        num_source=num_source, num_target=num_target, device=device,
    )
    mass = torch.ones(batch * num_source, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    plan, diagnostics = semi_relaxed_sparse_transport(
        mass, source, target, cost,
        num_targets=batch * num_target,
        epsilon=0.2, tau=1.0, max_iterations=iterations,
        tolerance=1e-5,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    constant = plan.pullback(torch.ones(batch * num_target, 1, device=device))
    constant_error = float(torch.max(torch.abs(constant - 1.0)))
    if diagnostics.relative_mass_residual > 1e-6:
        raise RuntimeError("SROT failed relative mass gate")
    if diagnostics.max_relative_row_residual > 1e-5:
        raise RuntimeError("SROT failed hard-row gate")
    if constant_error > 1e-5:
        raise RuntimeError("SROT failed constant pullback gate")
    return {
        "shape": f"{batch}x{num_source}->{batch}x{num_target}",
        "edges": plan.edge_count,
        "seconds": elapsed,
        "constant_error": constant_error,
        "diagnostics": diagnostics.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-points", type=int, default=100000)
    parser.add_argument("--working-points", type=int, default=4096)
    parser.add_argument("--middle-points", type=int, default=1024)
    parser.add_argument("--coarse-points", type=int, default=256)
    parser.add_argument("--assignment-k", type=int, default=2)
    parser.add_argument("--proxy-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-allocated-mib", type=float, default=3200.0)
    parser.add_argument("--output", type=Path, default=Path("transport_benchmark.json"))
    args = parser.parse_args()
    if not (args.full_points >= args.working_points > args.middle_points > args.coarse_points > 0):
        parser.error("expected full >= working > middle > coarse > 0")

    generator = np.random.default_rng(2026)
    full_xyz = generator.normal(size=(args.full_points, 3)).astype(np.float32)
    full_mass = generator.lognormal(size=args.full_points).astype(np.float32)
    full_mass /= max(float(full_mass.mean()), 1e-6)
    started = time.perf_counter()
    anchor_index = select_spatial_anchors(
        full_xyz, full_mass, args.working_points
    )
    anchor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    assignment = build_full_cloud_assignment(
        full_xyz, full_mass, anchor_xyz=full_xyz[anchor_index],
        k=args.assignment_k,
    )
    assignment_seconds = time.perf_counter() - started
    assignment_constant = assignment.pullback(
        torch.ones(args.working_points, 1)
    )
    assignment_error = float(torch.max(torch.abs(assignment_constant - 1.0)))
    if assignment.relative_mass_residual > 1e-6 or assignment_error > 1e-5:
        raise RuntimeError("full-cloud assignment failed conservation gate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    levels = [
        _srot_gate(
            args.working_points, args.middle_points, args.batch_size,
            args.proxy_k, args.iterations, device,
        ),
        _srot_gate(
            args.middle_points, args.coarse_points, args.batch_size,
            args.proxy_k, args.iterations, device,
        ),
    ]
    allocated = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    reserved = torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0.0
    if allocated > args.max_allocated_mib:
        raise RuntimeError(
            f"VRAM Limit Exceeded: {allocated:.1f} MiB > {args.max_allocated_mib:.1f} MiB"
        )
    stress_fp16_mib = args.batch_size * args.full_points * 16 * 64 * 2 / 1024**2
    solver_converged_all = all(
        bool(level["diagnostics"]["converged"]) for level in levels
    )
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "full_assignment": {
            "shape": f"{args.full_points}->{args.working_points}",
            "anchor_seconds": anchor_seconds,
            "assignment_seconds": assignment_seconds,
            "edges": assignment.edge_count,
            "relative_mass_residual": assignment.relative_mass_residual,
            "constant_error": assignment_error,
        },
        "srot_levels": levels,
        "max_memory_allocated_mib": allocated,
        "max_memory_reserved_mib": reserved,
        "forbidden_100k_edge_tensor_fp16_mib": stress_fp16_mib,
        "hardware_passed": allocated <= args.max_allocated_mib,
        "mathematical_passed": solver_converged_all,
        "passed": allocated <= args.max_allocated_mib and solver_converged_all,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
