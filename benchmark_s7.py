"""Ten-step real-batch hardware and numerical dry-run for all S7 arms."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from camo_build.losses import BalancedCEWithLovasz
from camo_build.protocol import verify_protocol
from camo_build.s7 import CAMOPyramidS7
from camo_s5_dataset import CAMOS5Dataset
from dataset import IGNORE_INDEX
from train_camo_s7 import PREREG, ROOT, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/s7_dry_run/benchmark.json")
    args = parser.parse_args()
    verification = verify_protocol(ROOT, ROOT / "protocol/s0")
    if not verification["passed"]:
        raise RuntimeError(verification)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S7 benchmark requires BF16 CUDA")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dataset = CAMOS5Dataset(
        ROOT, "train", 4096, 16, True, 42,
        proxies=256, support_k=8, proxy_k=16,
    )
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)))
    device = torch.device("cuda")
    keys = ("features", "mass", "neighbors", "anchor_index", "support_target", "proxy_neighbors", "label")
    tensors = {key: batch[key].to(device, non_blocking=True) for key in keys}
    weights = torch.from_numpy(np.load(ROOT / "class_weights.npy", allow_pickle=False)).float().to(device)
    loss_fn = BalancedCEWithLovasz(weights, lovasz_weight=0.5, ignore_index=IGNORE_INDEX)
    results: dict[str, object] = {}
    for mode in ("identity", "unweighted_mha", "mass_aware_mha"):
        model = CAMOPyramidS7(width=64, global_mode=mode, attention_heads=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings: list[float] = []
        residuals: list[float] = []
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            anomaly = step == 0
            with torch.autograd.set_detect_anomaly(anomaly):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, coarse = model(
                        tensors["features"], tensors["mass"], tensors["neighbors"],
                        tensors["anchor_index"], tensors["support_target"],
                        tensors["proxy_neighbors"], return_coarsening=True,
                    )
                    loss = loss_fn(logits, tensors["label"])
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"{mode} produced non-finite gradient")
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= 4:
                timings.append(elapsed)
            input_mass = tensors["mass"].float().sum(1)
            output_mass = coarse.proxy_mass.float().sum(1)
            residuals.append(float(((output_mass - input_mass).abs() / input_mass).max()))
        median = statistics.median(timings)
        allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        results[mode] = {
            "median_step_seconds": median,
            "points_per_second": 8 * 4096 / median,
            "max_memory_allocated_mib": allocated,
            "max_memory_reserved_mib": reserved,
            "max_relative_mass_residual": max(residuals),
            "finite_gradient": True,
            "memory_gate_passed": allocated <= 3200.0,
            "mass_gate_passed": max(residuals) <= 1e-6,
        }
        del model, optimizer
        torch.cuda.empty_cache()
    identity = float(results["identity"]["median_step_seconds"])  # type: ignore[index]
    for mode in ("unweighted_mha", "mass_aware_mha"):
        ratio = float(results[mode]["median_step_seconds"]) / identity  # type: ignore[index]
        results[mode]["latency_ratio_vs_identity"] = ratio  # type: ignore[index]
        results[mode]["latency_gate_passed"] = ratio <= 1.25  # type: ignore[index]
    artifact = {
        "schema_version": "1.0", "stage": "S7", "phase": "DRY_RUN",
        "configuration": {"batch": 8, "points": 4096, "proxies": 256, "width": 64, "heads": 4, "steps": 10, "timed_steps": 6},
        "s7_preregistration_sha256": sha256(PREREG),
        "protocol_verification": verification, "results": results,
        "all_numerical_and_memory_gates_passed": all(
            value["memory_gate_passed"] and value["mass_gate_passed"] and value["finite_gradient"]
            for value in results.values()  # type: ignore[union-attr]
        ),
        "test_split_used": False,
    }
    write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2))
    if not artifact["all_numerical_and_memory_gates_passed"]:
        raise RuntimeError("S7 numerical or memory dry-run gate failed")


if __name__ == "__main__":
    main()

