"""Frozen full-cloud validation for S2 and S5 CAMO checkpoints.

Every validation point is covered exactly once per frozen vote.  Point logits
are accumulated at their original indices; padding predictions are excluded.
No test split, label-dependent topology, or dense O(N^2) tensor is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.metrics import LogitVoteAccumulator, SegmentationMeter, full_cloud_chunks
from camo_build.model import CAMOPrototype1
from camo_build.neighbors import scipy_knn_indices
from camo_build.protocol import verify_protocol
from camo_build.pyramid import CAMOPyramidS5
from camo_s5_dataset import build_s5_topology
from camo_dataset import CAMOBuildingNetDataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES


ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "protocol" / "s0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    def sanitize(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): sanitize(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(inner) for inner in item]
        return item

    clean = sanitize(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_chunk(
    features: torch.Tensor,
    mass: torch.Tensor,
    index: np.ndarray,
    *,
    architecture: str,
    fine_k: int,
    proxies: int,
    support_k: int,
    proxy_k: int,
) -> dict[str, torch.Tensor]:
    chunk_features = features[index].clone()
    chunk_mass = mass[index].clone().float()
    chunk_mass = chunk_mass / chunk_mass.mean().clamp_min(1e-6)
    xyz = chunk_features[:, :3].numpy()
    result = {
        "features": chunk_features,
        "mass": chunk_mass,
        "neighbors": torch.from_numpy(scipy_knn_indices(xyz, fine_k)).long(),
    }
    if architecture == "s5":
        result.update(
            build_s5_topology(
                chunk_features,
                chunk_mass,
                proxies=proxies,
                support_k=support_k,
                proxy_k=proxy_k,
            )
        )
    return result


def stack_chunks(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = chunks[0].keys()
    return {key: torch.stack([chunk[key] for chunk in chunks]) for key in keys}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: CAMOBuildingNetDataset,
    device: torch.device,
    *,
    architecture: str,
    vote_seeds: list[int],
    class_groups: dict[str, list[int]],
    chunk_size: int,
    inference_batch_size: int,
    fine_k: int,
    proxies: int,
    support_k: int,
    proxy_k: int,
    max_models: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    model.eval()
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    model_count = len(dataset) if max_models is None else min(max_models, len(dataset))
    total_points = 0
    total_chunks = 0
    started = time.perf_counter()
    for sample_index in range(model_count):
        full = dataset.load_full(sample_index)
        features = full["features"]
        mass = full["mass"]
        target = full["label"]
        if not isinstance(features, torch.Tensor) or not isinstance(mass, torch.Tensor):
            raise TypeError("full-cloud feature and mass tensors are required")
        if not isinstance(target, torch.Tensor):
            raise TypeError("full-cloud label tensor is required")
        accumulator = LogitVoteAccumulator(len(features), NUM_PART_CLASSES)
        for vote_seed in vote_seeds:
            groups = full_cloud_chunks(
                len(features), chunk_size, vote_seed + sample_index
            )
            for start in range(0, len(groups), inference_batch_size):
                selected = groups[start : start + inference_batch_size]
                cpu_chunks = [
                    build_chunk(
                        features,
                        mass,
                        index,
                        architecture=architecture,
                        fine_k=fine_k,
                        proxies=proxies,
                        support_k=support_k,
                        proxy_k=proxy_k,
                    )
                    for index, _ in selected
                ]
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in stack_chunks(cpu_chunks).items()
                }
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.bfloat16,
                ):
                    if architecture == "s5":
                        logits = model(
                            batch["features"], batch["mass"], batch["neighbors"],
                            batch["anchor_index"], batch["support_target"],
                            batch["proxy_neighbors"],
                        )
                    else:
                        logits = model(
                            batch["features"], batch["mass"], batch["neighbors"]
                        )
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("full-cloud logits contain NaN or Inf")
                for row, (index, valid_count) in enumerate(selected):
                    point_index = torch.from_numpy(index[:valid_count]).long()
                    accumulator.update(point_index, logits[row, :valid_count])
                total_chunks += len(selected)
        expected_votes = len(vote_seeds)
        if not torch.all(accumulator.count == expected_votes):
            minimum = int(accumulator.count.min())
            maximum = int(accumulator.count.max())
            raise RuntimeError(
                f"vote coverage mismatch: expected {expected_votes}, got [{minimum},{maximum}]"
            )
        prediction = accumulator.prediction()
        meter.update(
            prediction.unsqueeze(0),
            target.unsqueeze(0),
            model_names=[str(full["model_name"])],
        )
        total_points += len(features)
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == model_count:
            elapsed = time.perf_counter() - started
            print(
                f"full_cloud architecture={architecture} "
                f"models={sample_index + 1}/{model_count} "
                f"points={total_points} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    metrics = meter.compute(class_groups)
    diagnostics = {
        "models": model_count,
        "total_points": total_points,
        "total_chunks_including_votes": total_chunks,
        "votes": len(vote_seeds),
        "elapsed_seconds": elapsed,
        "points_per_second": total_points * len(vote_seeds) / max(elapsed, 1e-9),
    }
    return metrics, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("s2", "s5"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--fine-k", type=int, default=16)
    parser.add_argument("--proxies", type=int, default=1024)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--proxy-k", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite full-cloud artifact: {args.output}")
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    protocol = json.loads(
        (PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text(encoding="utf-8")
    )
    if args.chunk_size != 4096 or args.fine_k != 16:
        raise ValueError("full-cloud evaluation must use frozen N=4096 and k=16")
    if args.architecture == "s5" and (
        args.proxies != 1024 or args.support_k != 8 or args.proxy_k != 16
    ):
        raise ValueError("S5 full-cloud topology does not match preregistration")
    vote_seeds = [int(value) for value in protocol["crop_seeds"]]
    if args.dry_run:
        vote_seeds = vote_seeds[:1]
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-cloud evaluation requires BF16 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    dataset = CAMOBuildingNetDataset(
        ROOT, "val", args.chunk_size, args.fine_k, False,
        args.training_seed, build_neighbors=False,
    )
    if args.architecture == "s5":
        model: torch.nn.Module = CAMOPyramidS5(
            width=64,
            coarsening_mode="row_softmax",
            transport_epsilon=0.2,
            transport_tau=0.0,
            transport_iterations=100,
            transport_backend="reference",
        )
    else:
        model = CAMOPrototype1(width=64, mass_mode="density")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    metrics, diagnostics = evaluate(
        model,
        dataset,
        device,
        architecture=args.architecture,
        vote_seeds=vote_seeds,
        class_groups=protocol["class_groups"],
        chunk_size=args.chunk_size,
        inference_batch_size=args.inference_batch_size,
        fine_k=args.fine_k,
        proxies=args.proxies,
        support_k=args.support_k,
        proxy_k=args.proxy_k,
        max_models=1 if args.dry_run else None,
    )
    report = {
        "schema_version": "1.0",
        "protocol_version": protocol["version"],
        "split": "validation",
        "metric_role": "primary_development_metric",
        "architecture": args.architecture,
        "training_seed": args.training_seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "vote_seeds": vote_seeds,
        "chunk_size": args.chunk_size,
        "inference_batch_size": args.inference_batch_size,
        "dry_run": args.dry_run,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "amp_dtype": "bfloat16",
        },
        "test_split_used": False,
        "protocol_verification_passed": True,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(report, indent=2, default=str, allow_nan=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
