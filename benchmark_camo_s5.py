"""Numerical, throughput and 4 GB VRAM gates for CAMO-Pyramid S5."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.losses import BalancedCEWithLovasz
from camo_build.neighbors import scipy_knn_indices
from camo_build.pyramid import CAMOPyramidS5
from camo_s5_dataset import build_s5_topology


MIB = 1024**2


def dummy_batch(
    batch_size: int,
    points: int,
    proxies: int,
    fine_k: int,
    proxy_k: int,
    support_k: int,
    device: torch.device,
    seed: int = 20260924,
) -> tuple[torch.Tensor, ...]:
    generator = np.random.default_rng(seed)
    xyz = generator.normal(size=(batch_size, points, 3)).astype(np.float32)
    normal = generator.normal(size=(batch_size, points, 3)).astype(np.float32)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    rgb = generator.random(size=(batch_size, points, 3)).astype(np.float32)
    feature = np.concatenate((xyz, normal, rgb), axis=-1)
    mass = generator.lognormal(size=(batch_size, points)).astype(np.float32)
    mass /= np.maximum(mass.mean(axis=1, keepdims=True), 1e-6)
    fine_graph = np.stack([scipy_knn_indices(cloud, fine_k) for cloud in xyz])
    anchors, support, proxy_graph = [], [], []
    for item in range(batch_size):
        topology = build_s5_topology(
            torch.from_numpy(feature[item]), torch.from_numpy(mass[item]),
            proxies=proxies, support_k=support_k, proxy_k=proxy_k,
        )
        anchors.append(topology["anchor_index"])
        support.append(topology["support_target"])
        proxy_graph.append(topology["proxy_neighbors"])
    labels = generator.integers(0, 31, size=(batch_size, points), dtype=np.int64)
    return (
        torch.from_numpy(feature).to(device),
        torch.from_numpy(mass).to(device),
        torch.from_numpy(fine_graph).long().to(device),
        torch.stack(anchors).to(device),
        torch.stack(support).to(device),
        torch.stack(proxy_graph).to(device),
        torch.from_numpy(labels).to(device),
    )


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hard", "row_softmax", "srot"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--proxies", type=int, default=1024)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fine-k", type=int, default=16)
    parser.add_argument("--proxy-k", type=int, default=16)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--transport-epsilon", type=float, default=0.2)
    parser.add_argument("--transport-tau", type=float, default=1.0)
    parser.add_argument("--transport-iterations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-allocated-mib", type=float, default=3200.0)
    parser.add_argument("--max-latency-ratio", type=float, default=1.25)
    parser.add_argument("--hard-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.proxies < args.points:
        parser.error("expected 0 < proxies < points")
    if args.steps <= args.warmup:
        parser.error("steps must exceed warmup")
    if args.mode != "hard" and args.hard_baseline is None:
        parser.error("non-hard arms require --hard-baseline for the frozen latency gate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("S5 hardware gate requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frozen S5 BF16 path is unsupported on this GPU")
    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)

    topology_started = time.perf_counter()
    batch = dummy_batch(
        args.batch_size, args.points, args.proxies, args.fine_k,
        args.proxy_k, args.support_k, device,
    )
    topology_seconds = time.perf_counter() - topology_started
    model = CAMOPyramidS5(
        width=args.width,
        coarsening_mode=args.mode,
        transport_epsilon=args.transport_epsilon,
        transport_tau=args.transport_tau,
        transport_iterations=args.transport_iterations,
    ).to(device)
    loss_fn = BalancedCEWithLovasz(torch.ones(31, device=device), lovasz_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Small anomaly-detection pass before the timed full-size path.
    small_points = 128
    small_proxies = 32
    small = dummy_batch(1, small_points, small_proxies, 8, 8, 4, device)
    debug_model = CAMOPyramidS5(
        width=min(args.width, 32), coarsening_mode=args.mode,
        transport_epsilon=args.transport_epsilon,
        transport_tau=args.transport_tau,
        transport_iterations=args.transport_iterations,
    ).to(device)
    with torch.autograd.set_detect_anomaly(True):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            debug_logits = debug_model(*small[:6])
            debug_loss = loss_fn(debug_logits, small[6])
        debug_loss.backward()
    assert_finite_gradients(debug_model)
    del debug_model, small, debug_logits, debug_loss

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measured_seconds = 0.0
    measured_steps = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if step >= args.warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, coarsening = model(*batch[:6], return_coarsening=True)
            loss = loss_fn(logits, batch[6])
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite S5 gradient norm")
        assert_finite_gradients(model)
        optimizer.step()
        if step >= args.warmup:
            torch.cuda.synchronize()
            measured_seconds += time.perf_counter() - started
            measured_steps += 1

    allocated = torch.cuda.max_memory_allocated() / MIB
    reserved = torch.cuda.max_memory_reserved() / MIB
    if allocated > args.max_allocated_mib:
        raise RuntimeError(
            f"VRAM Limit Exceeded: {allocated:.1f} MiB > {args.max_allocated_mib:.1f} MiB"
        )
    seconds_per_step = measured_seconds / measured_steps
    if args.mode == "hard":
        latency_ratio = 1.0
    else:
        assert args.hard_baseline is not None
        hard_report = json.loads(args.hard_baseline.read_text(encoding="utf-8"))
        latency_ratio = seconds_per_step / float(hard_report["seconds_per_step"])
    latency_passed = latency_ratio <= args.max_latency_ratio
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "config": vars(args) | {
            "output": str(args.output),
            "hard_baseline": str(args.hard_baseline) if args.hard_baseline else None,
            "amp_dtype": "bfloat16",
            "lovasz_weight": 0.5,
        },
        "topology_seconds_for_one_batch": topology_seconds,
        "loss": float(loss.detach()),
        "max_memory_allocated_mib": allocated,
        "max_memory_reserved_mib": reserved,
        "seconds_per_step": seconds_per_step,
        "latency_ratio_vs_hard": latency_ratio,
        "latency_gate_passed": latency_passed,
        "points_per_second": measured_steps * args.batch_size * args.points / measured_seconds,
        "transport_diagnostics": (
            coarsening.diagnostics.__dict__ if coarsening.diagnostics is not None else None
        ),
        "stress_calculation": {
            "input_B8_N100k_D9_fp32_mib": 8 * 100000 * 9 * 4 / MIB,
            "forbidden_edge_B8_N100k_k16_D64_fp16_mib": 8 * 100000 * 16 * 64 * 2 / MIB,
            "rule": "100k-point LMAO is forbidden; full-cloud assignment must stream to 4096 points"
        },
        "numerical_and_memory_gate_passed": True,
        "passed": latency_passed,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
