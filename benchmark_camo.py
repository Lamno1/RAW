"""Independent debug and hardware gates for CAMO-Build Prototype 1."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.losses import BalancedCEWithLovasz, StaticBalancedCrossEntropy
from camo_build.model import CAMOPrototype1
from camo_build.neighbors import scipy_knn_indices


MIB = 1024**2


def make_loss(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    weights = torch.ones(31, device=device)
    if args.lovasz_weight > 0.0:
        return BalancedCEWithLovasz(
            weights, lovasz_weight=args.lovasz_weight
        )
    return StaticBalancedCrossEntropy(weights)


def dummy_batch(
    batch_size: int,
    points: int,
    neighbors: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    generator = np.random.default_rng(42)
    xyz_np = generator.normal(size=(batch_size, points, 3)).astype(np.float32)
    graph_np = np.stack(
        [scipy_knn_indices(cloud, neighbors) for cloud in xyz_np]
    )
    normal_np = generator.normal(size=(batch_size, points, 3)).astype(np.float32)
    normal_np /= np.maximum(np.linalg.norm(normal_np, axis=-1, keepdims=True), 1e-6)
    rgb_np = generator.random(size=(batch_size, points, 3)).astype(np.float32)
    features_np = np.concatenate((xyz_np, normal_np, rgb_np), axis=-1)
    mass_np = generator.lognormal(size=(batch_size, points)).astype(np.float32)
    mass_np /= np.maximum(mass_np.mean(axis=1, keepdims=True), 1e-6)
    labels_np = generator.integers(0, 31, size=(batch_size, points), dtype=np.int64)
    return (
        torch.from_numpy(features_np).to(device),
        torch.from_numpy(mass_np).to(device),
        torch.from_numpy(graph_np).long().to(device),
        torch.from_numpy(labels_np).to(device),
    )


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def debug_gate(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    small_points = min(args.points, 256)
    batch = dummy_batch(min(args.batch_size, 2), small_points, args.k, device)
    model = CAMOPrototype1(width=args.width, mass_mode=args.mass_mode).to(device)
    loss_fn = make_loss(args, device)
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    with torch.autograd.set_detect_anomaly(True):
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=autocast_dtype
        ):
            logits = model(batch[0], batch[1], batch[2])
            loss = loss_fn(logits, batch[3])
        loss.backward()
    assert_finite_gradients(model)
    return {"debug_loss": float(loss.detach())}


def performance_gate(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        raise RuntimeError("hardware gate requires CUDA")
    batch = dummy_batch(args.batch_size, args.points, args.k, device)
    model = CAMOPrototype1(width=args.width, mass_mode=args.mass_mode).to(device)
    loss_fn = make_loss(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "float16", init_scale=args.amp_init_scale
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measured_seconds = 0.0
    measured_steps = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if step >= args.warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits = model(batch[0], batch[1], batch[2])
            loss = loss_fn(logits, batch[3])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert_finite_gradients(model)
        scaler.step(optimizer)
        scaler.update()
        if step >= args.warmup:
            torch.cuda.synchronize()
            measured_seconds += time.perf_counter() - started
            measured_steps += 1

    allocated_mib = torch.cuda.max_memory_allocated() / MIB
    reserved_mib = torch.cuda.max_memory_reserved() / MIB
    if allocated_mib > args.max_allocated_mib:
        raise RuntimeError(
            f"VRAM Limit Exceeded: {allocated_mib:.1f} MiB > "
            f"{args.max_allocated_mib:.1f} MiB"
        )
    points_per_second = (
        measured_steps * args.batch_size * args.points / measured_seconds
    )
    return {
        "loss": float(loss.detach()),
        "max_memory_allocated_mib": allocated_mib,
        "max_memory_reserved_mib": reserved_mib,
        "points_per_second": points_per_second,
        "seconds_per_step": measured_seconds / measured_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--mass-mode", choices=("none", "uniform", "density"), default="density")
    parser.add_argument("--max-allocated-mib", type=float, default=3200.0)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--amp-init-scale", type=float, default=32768.0)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lovasz_weight < 0.0:
        parser.error("lovasz-weight must be non-negative")
    if not np.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0.0:
        parser.error("amp-init-scale must be finite and positive")
    if args.steps <= args.warmup:
        parser.error("steps must be larger than warmup")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp_dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 requested but unsupported by this CUDA device")
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config": vars(args) | {"output": str(args.output) if args.output else None},
    }
    result.update(debug_gate(args, device))
    result.update(performance_gate(args, device))
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
