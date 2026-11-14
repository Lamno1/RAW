"""Prospective absolute-budget gate for the non-retroactive SROT v2 branch.

Unlike the v1 solver-only comparison, the hard gate here times fresh real-data
loading, topology construction, host-to-device copies, forward, backward, and
the optimizer step.  Hard pooling is still paired and reported, but its ratio
is diagnostic only under v2.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from camo_build.losses import BalancedCEWithLovasz
from camo_build.protocol import verify_protocol
from camo_build.pyramid import CAMOPyramidS5
from camo_s5_dataset import CAMOS5Dataset
from dataset import IGNORE_INDEX


ROOT = Path(__file__).resolve().parent
MUTATION = ROOT / "protocol" / "v2" / "srot_latency_mutation.json"
MIB = 1024**2


def materialize_batch(
    dataset: CAMOS5Dataset,
    indices: list[int],
    device: torch.device,
) -> dict[str, object]:
    batch = default_collate([dataset[index] for index in indices])
    for key in (
        "features", "mass", "neighbors", "anchor_index", "support_target",
        "proxy_neighbors", "label",
    ):
        batch[key] = batch[key].to(device, non_blocking=False)  # type: ignore[union-attr]
    return batch


def timed_full_step(
    model: CAMOPyramidS5,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    dataset: CAMOS5Dataset,
    indices: list[int],
    device: torch.device,
) -> tuple[float, object]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    batch = materialize_batch(dataset, indices, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, coarse = model(
            batch["features"], batch["mass"], batch["neighbors"],
            batch["anchor_index"], batch["support_target"],
            batch["proxy_neighbors"], return_coarsening=True,
        )
        loss = loss_fn(logits, batch["label"])
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite gradient in SROT v2 gate")
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, coarse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mutation = json.loads(MUTATION.read_text(encoding="utf-8"))
    verification = verify_protocol(ROOT, ROOT / "protocol" / "s0")
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if mutation["state"] != "FROZEN_BEFORE_ANY_V2_TRAINING_METRIC":
        raise RuntimeError("SROT v2 mutation is not frozen")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SROT v2 gate requires BF16 CUDA")

    frozen = mutation["frozen_configuration"]
    gates = mutation["v2_hardware_gates"]
    device = torch.device("cuda")
    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    np.random.seed(20260924)
    dataset = CAMOS5Dataset(
        ROOT, "train", frozen["points_per_sample"], 16, False, 20260924,
        proxies=frozen["proxies"], support_k=frozen["support_k"], proxy_k=16,
    )
    batch_size = int(frozen["batch_size"])
    warmup = int(frozen["warmup_repetitions"])
    measured = int(frozen["timed_repetitions"])
    if len(dataset) < batch_size * (warmup + measured):
        raise RuntimeError("dataset is too small for non-repeating v2 pairs")

    hard = CAMOPyramidS5(
        width=64, coarsening_mode="hard", transport_iterations=100,
        transport_backend="reference",
    ).to(device)
    srot = CAMOPyramidS5(
        width=64, coarsening_mode="srot", transport_epsilon=0.2,
        transport_tau=1.0, transport_iterations=100, warm_start=False,
        transport_backend="reference",
    ).to(device)
    srot.load_state_dict(hard.state_dict(), strict=False)
    hard_optimizer = torch.optim.AdamW(hard.parameters(), lr=1e-3, weight_decay=1e-4)
    srot_optimizer = torch.optim.AdamW(srot.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = torch.from_numpy(
        np.load(ROOT / "class_weights.npy", allow_pickle=False)
    ).float().to(device)
    loss_fn = BalancedCEWithLovasz(
        class_weights, lovasz_weight=0.5, ignore_index=IGNORE_INDEX
    )

    hard_times: list[float] = []
    srot_times: list[float] = []
    last_coarse = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total = warmup + measured
    for pair in range(total):
        start = pair * batch_size
        indices = list(range(start, start + batch_size))
        order = (
            ((srot, srot_optimizer, srot_times, True),
             (hard, hard_optimizer, hard_times, False))
            if pair % 2
            else
            ((hard, hard_optimizer, hard_times, False),
             (srot, srot_optimizer, srot_times, True))
        )
        for model, optimizer, destination, is_srot in order:
            elapsed, coarse = timed_full_step(
                model, optimizer, loss_fn, dataset, indices, device
            )
            if is_srot:
                last_coarse = coarse
            if pair >= warmup:
                destination.append(elapsed)

    if last_coarse is None or last_coarse.diagnostics is None:
        raise RuntimeError("SROT v2 gate did not produce transport diagnostics")
    diagnostics = last_coarse.diagnostics
    source_feature = torch.randn(
        last_coarse.plan.source_mass.numel(), 3, device=device
    )
    target_feature = torch.randn(
        last_coarse.plan.target_mass.numel(), 3, device=device
    )
    adjoint_error = last_coarse.plan.adjoint_error(source_feature, target_feature)
    srot_median = statistics.median(srot_times)
    hard_median = statistics.median(hard_times)
    points_per_second = batch_size * frozen["points_per_sample"] / srot_median
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    projected_hours = srot_median * steps_per_epoch * 50 * 3 / 3600.0
    allocated = torch.cuda.max_memory_allocated() / MIB
    paired_ratios = [s / h for s, h in zip(srot_times, hard_times, strict=True)]
    numerical_pass = (
        diagnostics.converged
        and diagnostics.relative_mass_residual
        <= mutation["unchanged_mathematical_gates"]["maximum_relative_source_mass_residual"]
        and adjoint_error <= 1e-5
    )
    hardware_pass = (
        points_per_second
        >= gates["minimum_end_to_end_model_step_throughput_points_per_second"]
        and srot_median <= gates["equivalent_maximum_median_step_latency_seconds"]
        and allocated <= gates["maximum_allocated_vram_mib"]
        and projected_hours <= gates["maximum_projected_three_seed_wall_hours"]
    )
    report = {
        "protocol_version": mutation["protocol_version"],
        "mutation_id": mutation["mutation_id"],
        "method": "paired_real_data_dataloader_topology_h2d_forward_backward_optimizer",
        "s0_verified": True,
        "fresh_geometry_each_pair": True,
        "warm_start": False,
        "hard_seconds": hard_times,
        "srot_seconds": srot_times,
        "paired_ratios": paired_ratios,
        "hard_median_seconds": hard_median,
        "srot_median_seconds": srot_median,
        "median_paired_ratio_diagnostic": statistics.median(paired_ratios),
        "points_per_second": points_per_second,
        "projected_three_seed_wall_hours": projected_hours,
        "max_memory_allocated_mib": allocated,
        "transport_diagnostics": diagnostics.__dict__,
        "adjoint_relative_error": adjoint_error,
        "numerical_gate_passed": numerical_pass,
        "hardware_gate_passed": hardware_pass,
        "passed": numerical_pass and hardware_pass,
        "segmentation_training_authorized": False,
        "authorization_note": "A passing benchmark only permits the separately specified representation study; it never authorizes segmentation by itself."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
