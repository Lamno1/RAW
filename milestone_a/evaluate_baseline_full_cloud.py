"""Full-cloud validation for the Milestone A baseline checkpoints.

Uses the same chunking, padding, vote seeds, logit accumulation and metric
implementation as the frozen S7 evaluation.  Unlike the S7 path it builds no
neighbor graph or topology, so the measured wall time contains only the
baseline's own cost.  Validation split only; the test split is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from camo_build.metrics import LogitVoteAccumulator, SegmentationMeter, full_cloud_chunks  # noqa: E402
from camo_build.protocol import verify_protocol  # noqa: E402
from camo_dataset import CAMOBuildingNetDataset  # noqa: E402
from dataset import IGNORE_INDEX, NUM_PART_CLASSES  # noqa: E402
from pointnext_baseline import PointNeXtBaseline, parameter_count  # noqa: E402

PROTOCOL_DIR = ROOT / "protocol" / "s0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    def sanitize(item: object) -> object:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(key): sanitize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(value) for value in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: CAMOBuildingNetDataset,
    device: torch.device,
    *,
    vote_seeds: list[int],
    class_groups: dict[str, list[int]],
    chunk_size: int,
    inference_batch_size: int,
    max_models: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    model.eval()
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    model_count = len(dataset) if max_models is None else min(max_models, len(dataset))
    total_points, total_chunks = 0, 0
    started = time.perf_counter()
    for sample_index in range(model_count):
        full = dataset.load_full(sample_index)
        features, target = full["features"], full["label"]
        accumulator = LogitVoteAccumulator(len(features), NUM_PART_CLASSES)
        for vote_seed in vote_seeds:
            groups = full_cloud_chunks(len(features), chunk_size, vote_seed + sample_index)
            for start in range(0, len(groups), inference_batch_size):
                selected = groups[start : start + inference_batch_size]
                batch = torch.stack([features[index] for index, _ in selected]).to(
                    device, non_blocking=True
                )
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
                ):
                    logits = model(batch)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("full-cloud logits contain NaN or Inf")
                for row, (index, valid_count) in enumerate(selected):
                    accumulator.update(
                        torch.from_numpy(index[:valid_count]).long(), logits[row, :valid_count]
                    )
                total_chunks += len(selected)
        if not torch.all(accumulator.count == len(vote_seeds)):
            raise RuntimeError(
                f"vote coverage mismatch: expected {len(vote_seeds)}, "
                f"got [{int(accumulator.count.min())},{int(accumulator.count.max())}]"
            )
        meter.update(
            accumulator.prediction().unsqueeze(0), target.unsqueeze(0),
            model_names=[str(full["model_name"])],
        )
        total_points += len(features)
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == model_count:
            print(f"full_cloud baseline models={sample_index + 1}/{model_count} "
                  f"points={total_points} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    elapsed = time.perf_counter() - started
    diagnostics = {
        "models": model_count, "total_points": total_points,
        "total_chunks_including_votes": total_chunks, "votes": len(vote_seeds),
        "elapsed_seconds": elapsed,
        "points_per_second": total_points * len(vote_seeds) / max(elapsed, 1e-9),
    }
    return meter.compute(class_groups), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite full-cloud artifact: {args.output}")
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    # A dry run measures nothing and supports no claim, so it may proceed past a
    # failed S0 check; every non-dry run must stop.
    if not verification["passed"] and not args.dry_run:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-cloud validation requires BF16 CUDA")
    protocol = json.loads((PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text(encoding="utf-8"))
    vote_seeds = [int(value) for value in protocol["crop_seeds"]]
    if args.dry_run:
        vote_seeds = vote_seeds[:1]
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    dataset = CAMOBuildingNetDataset(
        ROOT, "val", 4096, 16, False, args.training_seed, build_neighbors=False
    )
    model = PointNeXtBaseline()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics, diagnostics = evaluate(
        model, dataset, device, vote_seeds=vote_seeds, class_groups=protocol["class_groups"],
        chunk_size=4096, inference_batch_size=4, max_models=1 if args.dry_run else None,
    )
    report = {
        "schema_version": "1.0", "protocol_version": protocol["version"],
        "split": "validation", "metric_role": "primary_development_metric",
        "architecture": "pointnext_s_reduced_to_budget", "training_seed": args.training_seed,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "metrics_implementation_sha256": sha256(ROOT / "camo_build" / "metrics.py"),
        "vote_seeds": vote_seeds, "chunk_size": 4096, "inference_batch_size": 4,
        "parameters": parameter_count(model), "dry_run": args.dry_run,
        "metrics": metrics,
        "diagnostics": {**diagnostics, "wall_seconds": time.perf_counter() - started},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "amp_dtype": "bfloat16",
        },
        "test_split_used": False, "protocol_verification_passed": bool(verification["passed"]),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
