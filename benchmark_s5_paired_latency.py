"""Paired/interleaved S5 latency gate resistant to GPU clock drift."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark_camo_s5 import dummy_batch
from camo_build.losses import BalancedCEWithLovasz
from camo_build.pyramid import CAMOPyramidS5


def timed_step(
    model: CAMOPyramidS5,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
    loss_fn: torch.nn.Module,
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(*batch[:6])
        loss = loss_fn(logits, batch[6])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite paired-benchmark gradient")
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("row_softmax", "srot"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--proxies", type=int, default=1024)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fine-k", type=int, default=16)
    parser.add_argument("--proxy-k", type=int, default=16)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--transport-iterations", type=int, default=100)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--measured-pairs", type=int, default=8)
    parser.add_argument("--maximum-ratio", type=float, default=1.25)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--transport-backend", choices=("reference", "triton", "triton_hybrid"), default="reference")
    parser.add_argument("--alternate-batches", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("paired S5 benchmark requires BF16 CUDA")
    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    device = torch.device("cuda")
    batch = dummy_batch(
        args.batch_size, args.points, args.proxies, args.fine_k,
        args.proxy_k, args.support_k, device,
    )
    alternate_batch = (
        dummy_batch(
            args.batch_size, args.points, args.proxies, args.fine_k,
            args.proxy_k, args.support_k, device, seed=20260925,
        )
        if args.alternate_batches else None
    )
    hard = CAMOPyramidS5(
        width=args.width, coarsening_mode="hard",
        transport_iterations=args.transport_iterations,
    ).to(device)
    candidate = CAMOPyramidS5(
        width=args.width, coarsening_mode=args.candidate,
        transport_iterations=args.transport_iterations,
        warm_start=args.warm_start,
        transport_backend=args.transport_backend,
    ).to(device)
    # Identical initialization ensures the timed paths differ only by plan.
    candidate.load_state_dict(hard.state_dict(), strict=False)
    hard_optimizer = torch.optim.AdamW(hard.parameters(), lr=1e-3, weight_decay=1e-4)
    candidate_optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = BalancedCEWithLovasz(torch.ones(31, device=device), lovasz_weight=0.5)
    hard_times: list[float] = []
    candidate_times: list[float] = []
    total_pairs = args.warmup_pairs + args.measured_pairs
    for pair in range(total_pairs):
        candidate_first = pair % 2 == 1
        order = (
            ((candidate, candidate_optimizer, candidate_times),
             (hard, hard_optimizer, hard_times))
            if candidate_first
            else
            ((hard, hard_optimizer, hard_times),
             (candidate, candidate_optimizer, candidate_times))
        )
        for model, optimizer, destination in order:
            active_batch = (
                alternate_batch if alternate_batch is not None and pair % 2 else batch
            )
            elapsed = timed_step(model, optimizer, active_batch, loss_fn)
            if pair >= args.warmup_pairs:
                destination.append(elapsed)
    if len(hard_times) != args.measured_pairs or len(candidate_times) != args.measured_pairs:
        raise RuntimeError("paired latency accounting failed")
    paired_ratios = [
        candidate_time / hard_time
        for candidate_time, hard_time in zip(candidate_times, hard_times, strict=True)
    ]
    ratio = statistics.median(paired_ratios)
    report = {
        "method": "paired_interleaved_full_forward_backward_optimizer_wall_time",
        "candidate": args.candidate,
        "config": vars(args) | {"output": str(args.output)},
        "hard_seconds": hard_times,
        "candidate_seconds": candidate_times,
        "paired_ratios": paired_ratios,
        "hard_median_seconds": statistics.median(hard_times),
        "candidate_median_seconds": statistics.median(candidate_times),
        "median_paired_latency_ratio": ratio,
        "maximum_ratio": args.maximum_ratio,
        "passed": ratio <= args.maximum_ratio,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
