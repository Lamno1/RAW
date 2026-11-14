"""Full frozen-shape S8 forward/backward hardware and numerical gate.

This is a synthetic dry-run only: it does not load labels, train, evaluate
validation/test, or authorize an optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.coarsening import build_fixed_proxy_support
from camo_build.neighbors import scipy_knn_indices
from camo_build.pyramid_s8 import CAMOPyramidS8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batched_neighbors(xyz: np.ndarray, k: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([
        scipy_knn_indices(points, k) for points in xyz
    ])).long()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/s8_dry_run/benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S8 hardware gate requires CUDA")
    torch.manual_seed(801)
    np.random.seed(801)
    device = torch.device("cuda")
    batch, n0, n1, n2 = 8, 4096, 1024, 256
    rng = np.random.default_rng(801)
    xyz_np = rng.normal(size=(batch, n0, 3)).astype(np.float32)
    normal_np = rng.normal(size=(batch, n0, 3)).astype(np.float32)
    normal_np /= np.maximum(np.linalg.norm(normal_np, axis=-1, keepdims=True), 1e-6)
    rgb_np = rng.random((batch, n0, 3), dtype=np.float32)
    xyz = torch.from_numpy(xyz_np)
    normals = torch.from_numpy(normal_np)
    features = torch.cat((xyz, normals, torch.from_numpy(rgb_np)), dim=-1).to(device)
    mass = torch.rand(batch, n0, device=device) + 0.2
    mass = mass / mass.mean(dim=1, keepdim=True)

    anchor1_np = np.stack([
        np.linspace(0, n0 - 1, n1, dtype=np.int64) for _ in range(batch)
    ])
    support01_np = np.stack([
        build_fixed_proxy_support(xyz_np[b], anchor1_np[b], k=8)
        for b in range(batch)
    ])
    xyz1_np = xyz_np[np.arange(batch)[:, None], anchor1_np]
    anchor2_np = np.stack([
        np.linspace(0, n1 - 1, n2, dtype=np.int64) for _ in range(batch)
    ])
    support12_np = np.stack([
        build_fixed_proxy_support(xyz1_np[b], anchor2_np[b], k=8)
        for b in range(batch)
    ])
    xyz2_np = xyz1_np[np.arange(batch)[:, None], anchor2_np]
    graph0 = batched_neighbors(xyz_np, 16).to(device)
    graph1 = batched_neighbors(xyz1_np, 16).to(device)
    graph2 = batched_neighbors(xyz2_np, 12).to(device)
    anchor1 = torch.from_numpy(anchor1_np).to(device)
    anchor2 = torch.from_numpy(anchor2_np).to(device)
    support01 = torch.from_numpy(support01_np).to(device)
    support12 = torch.from_numpy(support12_np).to(device)

    model = CAMOPyramidS8().to(device).train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, state = model(
            features, mass, graph0, anchor1, support01, graph1,
            anchor2, support12, graph2, return_pyramid=True,
        )
        loss = logits.float().square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite S8 dry-run loss")
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    grad_nonfinite = [
        name for name, param in model.named_parameters()
        if param.grad is None or not torch.isfinite(param.grad).all()
    ]
    if grad_nonfinite:
        raise FloatingPointError(f"missing/non-finite gradients: {grad_nonfinite[:5]}")
    p01 = state["coarsening01"].plan
    p12 = state["coarsening12"].plan
    test_h0 = torch.randn(batch * n0, 5, device=device)
    test_h1 = torch.randn(batch * n1, 5, device=device)
    test_h2 = torch.randn(batch * n2, 5, device=device)
    adjoint01 = p01.adjoint_error(test_h0, test_h1)
    adjoint12 = p12.adjoint_error(test_h1, test_h2)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    mass_residuals = {
        "4096_to_1024": p01.relative_mass_residual,
        "1024_to_256": p12.relative_mass_residual,
    }
    passed = (
        peak_mib <= 3200.0
        and max(mass_residuals.values()) <= 1e-6
        and max(adjoint01, adjoint12) <= 1e-5
    )
    root = Path(__file__).resolve().parent
    record = {
        "schema_version": "1.0",
        "stage": "S8_DRY_RUN",
        "state": "PASSED" if passed else "HARDWARE_OR_NUMERICAL_GATE_FAILED",
        "training_authorized": False,
        "split_data_loaded": False,
        "test_split_used": False,
        "shape": {"batch": batch, "resolutions": [n0, n1, n2], "widths": [64, 96, 128]},
        "device": torch.cuda.get_device_name(device),
        "allocated_peak_mib": peak_mib,
        "reserved_peak_mib": reserved_mib,
        "forward_backward_seconds": elapsed,
        "mass_relative_residuals": mass_residuals,
        "adjoint_relative_errors": {"4096_to_1024": adjoint01, "1024_to_256": adjoint12},
        "all_parameter_gradients_finite": True,
        "gate_thresholds": {"allocated_peak_mib": 3200.0, "mass_relative_residual": 1e-6, "adjoint_relative_error": 1e-5},
        "source_sha256": {
            name: sha256(root / name)
            for name in ("camo_build/pyramid_s8.py", "benchmark_s8_dry_run.py")
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("S8 dry-run gate failed; no training may start")


if __name__ == "__main__":
    main()
