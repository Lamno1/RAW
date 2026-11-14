"""Run the S4 measure pyramid on one real BuildingNet cloud."""
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
    pushforward_axial_normals,
    semi_relaxed_sparse_transport,
)
from camo_dataset import CAMOBuildingNetDataset


def _coarsen(
    xyz: torch.Tensor,
    normal_axis: torch.Tensor,
    mass: torch.Tensor,
    num_target: int,
    *,
    proxy_k: int,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    anchor_index = select_spatial_anchors(
        xyz.cpu().numpy(), mass.cpu().numpy(), num_target
    )
    proxy_xyz = xyz[anchor_index]
    proxy_normal = normal_axis[anchor_index]
    edge_source, edge_target = build_sparse_proxy_support(
        xyz.cpu().numpy(), proxy_xyz.cpu().numpy(), k=proxy_k, reciprocal=1
    )
    edge_source = edge_source.to(device)
    edge_target = edge_target.to(device)
    cost = geometry_transport_cost(
        xyz.to(device), proxy_xyz.to(device), edge_source, edge_target,
        source_normals=normal_axis.to(device),
        proxy_normals=proxy_normal.to(device),
        sign_invariant_normals=True,
    )
    started = time.perf_counter()
    plan, diagnostics = semi_relaxed_sparse_transport(
        mass.to(device), edge_source, edge_target, cost,
        num_targets=num_target, epsilon=0.2, tau=1.0,
        max_iterations=iterations, tolerance=1e-5,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    next_xyz = plan.pushforward(xyz.to(device))
    next_normal = pushforward_axial_normals(plan, normal_axis.to(device))
    generator = torch.Generator(device=device).manual_seed(31 + num_target)
    source_feature = torch.randn(xyz.shape[0], 4, generator=generator, device=device)
    target_feature = torch.randn(num_target, 4, generator=generator, device=device)
    adjoint_error = plan.adjoint_error(source_feature, target_feature)
    report = {
        "source_points": xyz.shape[0], "target_points": num_target,
        "edges": plan.edge_count, "seconds": elapsed,
        "adjoint_error": adjoint_error,
        "diagnostics": diagnostics.__dict__,
    }
    if not diagnostics.converged:
        raise RuntimeError(f"real-data SROT did not converge: {report}")
    if adjoint_error > 1e-5:
        raise RuntimeError(f"real-data adjoint gate failed: {adjoint_error}")
    return next_xyz.detach().cpu(), next_normal.detach().cpu(), plan.target_mass.detach().cpu(), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--working-points", type=int, default=4096)
    parser.add_argument("--middle-points", type=int, default=1024)
    parser.add_argument("--coarse-points", type=int, default=256)
    parser.add_argument("--assignment-k", type=int, default=2)
    parser.add_argument("--proxy-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("transport_real_validation.json"))
    args = parser.parse_args()
    dataset = CAMOBuildingNetDataset(
        args.data_dir, args.split, num_points=32, k=8,
        augment=False, build_neighbors=False,
    )
    full = dataset.load_full(args.index)
    features = full["features"]
    xyz = features[:, :3]
    normals = features[:, 3:6]
    mass = full["mass"]
    if not (
        len(xyz) >= args.working_points > args.middle_points > args.coarse_points
    ):
        parser.error("expected full >= working > middle > coarse")

    anchor_index = select_spatial_anchors(
        xyz.numpy(), mass.numpy(), args.working_points
    )
    started = time.perf_counter()
    assignment = build_full_cloud_assignment(
        xyz.numpy(), mass.numpy(), anchor_xyz=xyz[anchor_index].numpy(),
        k=args.assignment_k,
    )
    assignment_seconds = time.perf_counter() - started
    working_xyz = assignment.pushforward(xyz)
    working_normal = pushforward_axial_normals(assignment, normals)
    working_mass = assignment.target_mass
    assignment_adjoint = assignment.adjoint_error(
        torch.randn(len(xyz), 3, generator=torch.Generator().manual_seed(7)),
        torch.randn(args.working_points, 3, generator=torch.Generator().manual_seed(8)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    middle_xyz, middle_normal, middle_mass, level0 = _coarsen(
        working_xyz, working_normal, working_mass, args.middle_points,
        proxy_k=args.proxy_k, iterations=args.iterations, device=device,
    )
    _, _, _, level1 = _coarsen(
        middle_xyz, middle_normal, middle_mass, args.coarse_points,
        proxy_k=args.proxy_k, iterations=args.iterations, device=device,
    )
    report = {
        "model_name": full["model_name"], "split": args.split,
        "full_points": len(xyz),
        "assignment": {
            "edges": assignment.edge_count,
            "seconds": assignment_seconds,
            "relative_mass_residual": assignment.relative_mass_residual,
            "adjoint_error": assignment_adjoint,
        },
        "levels": [level0, level1],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "max_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "passed": True,
    }
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
