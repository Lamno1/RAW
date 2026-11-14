"""Full-cloud validation for S7 identity/mass-aware global proxy checkpoints."""
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

from camo_build.protocol import verify_protocol
from camo_build.s7 import CAMOPyramidS7
from camo_dataset import CAMOBuildingNetDataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES
from evaluate_camo_full_cloud import evaluate

ROOT = Path(__file__).resolve().parent
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
    clean_payload = sanitize(payload)
    temporary.write_text(
        json.dumps(clean_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--global-mode", choices=("identity", "mass_aware_mha"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite full-cloud artifact: {args.output}")
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S7 full-cloud validation requires BF16 CUDA")
    protocol = json.loads((PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text(encoding="utf-8"))
    vote_seeds = [int(value) for value in protocol["crop_seeds"]]
    if args.dry_run:
        vote_seeds = vote_seeds[:1]
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    dataset = CAMOBuildingNetDataset(
        ROOT, "val", 4096, 16, False, args.training_seed, build_neighbors=False
    )
    model = CAMOPyramidS7(
        width=64, global_mode=args.global_mode, attention_heads=4
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics, diagnostics = evaluate(
        model, dataset, device,
        architecture="s5",  # Reuse the frozen chunk/topology/voting implementation.
        vote_seeds=vote_seeds,
        class_groups=protocol["class_groups"],
        chunk_size=4096,
        inference_batch_size=4,
        fine_k=16,
        proxies=256,
        support_k=8,
        proxy_k=16,
        max_models=1 if args.dry_run else None,
    )
    report = {
        "schema_version": "1.0",
        "protocol_version": protocol["version"],
        "split": "validation",
        "metric_role": "primary_development_metric",
        "architecture": f"s7_{args.global_mode}",
        "global_mode": args.global_mode,
        "training_seed": args.training_seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "base_voting_evaluator_sha256": sha256(ROOT / "evaluate_camo_full_cloud.py"),
        "vote_seeds": vote_seeds,
        "chunk_size": 4096,
        "proxies": 256,
        "support_k": 8,
        "proxy_k": 16,
        "inference_batch_size": 4,
        "dry_run": args.dry_run,
        "metrics": metrics,
        "diagnostics": {**diagnostics, "wall_seconds": time.perf_counter() - started},
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
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
