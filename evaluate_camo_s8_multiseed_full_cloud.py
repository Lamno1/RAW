"""Locked, validation-only S8 multi-seed full-cloud evaluation."""
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
from camo_build.protocol import verify_protocol
from camo_build.pyramid_s8 import CAMOPyramidS8
from camo_dataset import CAMOBuildingNetDataset
from camo_s8_dataset import build_s8_topology
from dataset import IGNORE_INDEX, NUM_PART_CLASSES


ROOT = Path(__file__).resolve().parent
S0 = ROOT / "protocol" / "s0"
S8 = ROOT / "protocol" / "s8"
AUTH = S8 / "multiseed_authorization.json"
CONFIG = S8 / "multiseed_configuration.json"
PREREG = S8 / "multiseed_preregistration.json"
CHUNK_SIZE = 4096
INFERENCE_BATCH_SIZE = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    def clean(value: object) -> object:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_lock(seed: int, checkpoint: Path, output: Path) -> tuple[dict[str, object], dict[str, object]]:
    if not PREREG.is_file() or not AUTH.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("S8 multi-seed preregistration/authorization is missing")
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(PREREG.read_text(encoding="utf-8"))
    if auth.get("state") != "AUTHORIZED_AFTER_INDEPENDENT_AUDIT":
        raise RuntimeError("S8 multi-seed authorization awaits independent audit")
    if lock.get("state") != "FROZEN_BEFORE_MULTI_SEED_DATA_ACCESS":
        raise RuntimeError("S8 multi-seed preregistration is not frozen")
    if auth.get("preregistration_sha256") != sha256(PREREG):
        raise RuntimeError("S8 multi-seed authorization preregistration hash mismatch")
    if seed not in auth.get("seeds", []) or seed not in config["training"]["seeds"]:
        raise RuntimeError("S8 evaluation seed is not locked")
    if (auth.get("epochs") != config["training"]["epochs"]
            or lock["training"]["seeds"] != config["training"]["seeds"]
            or lock["training"]["epochs"] != config["training"]["epochs"]
            or lock["evaluation"]["vote_seeds"] != config["evaluation"]["vote_seeds"]
            or lock["evaluation"]["chunk_size"] != config["evaluation"]["chunk_size"]
            or lock["evaluation"]["chunk_size"] != CHUNK_SIZE
            or lock["evaluation"]["inference_batch_size"] != INFERENCE_BATCH_SIZE):
        raise RuntimeError("S8 multi-seed evaluation contract disagrees with the locked configuration")
    expected_hashes = lock["locked_sha256"]
    actual_hashes = {
        "evaluator": sha256(Path(__file__).resolve()),
        "training_runner": sha256(ROOT / "train_camo_s8_multiseed.py"),
        "multiseed_configuration": sha256(CONFIG),
        "s8_model": sha256(ROOT / "camo_build" / "pyramid_s8.py"),
        "s8_dataset": sha256(ROOT / "camo_s8_dataset.py"),
        "base_dataset": sha256(ROOT / "camo_dataset.py"),
        "dataset_constants": sha256(ROOT / "dataset.py"),
        "coarsening": sha256(ROOT / "camo_build" / "coarsening.py"),
        "operator": sha256(ROOT / "camo_build" / "operator.py"),
        "transport": sha256(ROOT / "camo_build" / "transport.py"),
        "sparse_ops": sha256(ROOT / "camo_build" / "sparse_ops.py"),
        "s7_attention": sha256(ROOT / "camo_build" / "s7_attention.py"),
        "losses": sha256(ROOT / "camo_build" / "losses.py"),
        "assignment": sha256(ROOT / "camo_build" / "assignment.py"),
        "neighbors": sha256(ROOT / "camo_build" / "neighbors.py"),
        "metrics_implementation": sha256(ROOT / "camo_build" / "metrics.py"),
        "s0_protocol": sha256(S0 / "camo_pyramid_protocol.json"),
        "training_split": sha256(ROOT / "splits" / "train_split.txt"),
        "validation_split": sha256(ROOT / "splits" / "val_split.txt"),
        "s7_paired_validation_summary": sha256(ROOT / "runs" / "s7_confirmatory" / "paired_summary.json"),
        "architecture_preregistration": sha256(S0 / "mutations" / "s8_architecture_preregistration.json"),
    }
    if expected_hashes != actual_hashes:
        raise RuntimeError(f"S8 multi-seed lock hash mismatch: {actual_hashes}")
    if auth.get("test_split_access") is not False or lock["evaluation"].get("test_split_access") is not False:
        raise RuntimeError("S8 multi-seed evaluation must remain validation-only")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = checkpoint.parent
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "TRAINING_COMPLETE" or status.get("best_epoch") is None:
        raise RuntimeError("S8 multi-seed training has not completed")
    record = json.loads((run_dir / "authorization_record.json").read_text(encoding="utf-8"))
    if record.get("multiseed_preregistration_sha256") != sha256(PREREG):
        raise RuntimeError("S8 checkpoint run has a different preregistration")
    verification = verify_protocol(ROOT, S0)
    if not verification["passed"]:
        raise RuntimeError(f"frozen S0 protocol verification failed: {verification}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S8 multi-seed evaluation requires BF16-capable CUDA")
    return lock, json.loads((S0 / "camo_pyramid_protocol.json").read_text(encoding="utf-8"))


def make_chunk(
    features: torch.Tensor, mass: torch.Tensor, indices: np.ndarray
) -> dict[str, torch.Tensor]:
    selected_features = features[indices].clone().float()
    selected_mass = mass[indices].clone().float()
    selected_mass = selected_mass / selected_mass.mean().clamp_min(1e-6)
    topology = build_s8_topology(selected_features, selected_mass)
    return {"features": selected_features, "mass": selected_mass, **topology}


def stack(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([chunk[key] for chunk in chunks]) for key in chunks[0]}


@torch.inference_mode()
def run(model: torch.nn.Module, dataset: CAMOBuildingNetDataset, device: torch.device,
        vote_seeds: list[int], class_groups: dict[str, list[int]]) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    model.eval()
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    total_points = 0
    total_chunks = 0
    per_building: list[dict[str, object]] = []
    started = time.perf_counter()
    for sample_index in range(len(dataset)):
        full = dataset.load_full(sample_index)
        features, mass, target = full["features"], full["mass"], full["label"]
        if not all(isinstance(x, torch.Tensor) for x in (features, mass, target)):
            raise TypeError("full-cloud dataset returned invalid tensors")
        accumulator = LogitVoteAccumulator(len(features), NUM_PART_CLASSES)
        for vote_seed in vote_seeds:
            chunks = full_cloud_chunks(len(features), CHUNK_SIZE, vote_seed + sample_index)
            for start in range(0, len(chunks), INFERENCE_BATCH_SIZE):
                selected = chunks[start:start + INFERENCE_BATCH_SIZE]
                cpu_chunks = [make_chunk(features, mass, indices) for indices, _ in selected]
                batch = {key: value.to(device, non_blocking=True) for key, value in stack(cpu_chunks).items()}
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        batch["features"], batch["mass"], batch["neighbors0"],
                        batch["anchor1"], batch["support01"], batch["neighbors1"],
                        batch["anchor2"], batch["support12"], batch["neighbors2"],
                    )
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("S8 full-cloud logits contain NaN/Inf")
                for row, (indices, valid_count) in enumerate(selected):
                    accumulator.update(torch.from_numpy(indices[:valid_count]).long(), logits[row, :valid_count])
                total_chunks += len(selected)
        if not torch.all(accumulator.count == len(vote_seeds)):
            raise RuntimeError("full-cloud voting did not cover every point exactly once per vote")
        prediction = accumulator.prediction()
        valid_points = int(torch.count_nonzero(target != IGNORE_INDEX))
        if valid_points == 0:
            raise RuntimeError("full-cloud building contains no valid labels")
        before = meter.confusion.clone()
        meter.update(prediction.unsqueeze(0), target.unsqueeze(0), model_names=[str(full["model_name"])])
        per_building.append({
            "model_name": str(full["model_name"]),
            "confusion": (meter.confusion - before).tolist(),
            "shape_iou": meter.shape_ious[-1],
            "valid_points": valid_points,
        })
        total_points += len(features)
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == len(dataset):
            print(f"S8 full-cloud val buildings={sample_index + 1}/{len(dataset)} points={total_points} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    elapsed = time.perf_counter() - started
    return meter.compute(class_groups), {
        "buildings": len(dataset), "points": total_points, "chunks_including_votes": total_chunks,
        "votes": len(vote_seeds), "elapsed_seconds": elapsed,
        "points_per_second_including_votes": total_points * len(vote_seeds) / max(elapsed, 1e-9),
    }, per_building


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    run_dir = ROOT / "runs" / "s8_multiseed" / f"seed{seed}"
    checkpoint_path = run_dir / "best.pth"
    output = run_dir / "full_cloud_validation.json"
    lock, protocol = validate_lock(seed, checkpoint_path, output)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = CAMOBuildingNetDataset(ROOT, "val", CHUNK_SIZE, 16, False, seed, build_neighbors=False)
    model = CAMOPyramidS8(widths=(64, 96, 128), local_blocks_per_level=2, attention_heads=4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("config_sha256") != sha256(CONFIG)
            or checkpoint.get("preregistration_sha256") != sha256(PREREG)
            or checkpoint.get("authorization_sha256") != sha256(AUTH)):
        raise RuntimeError("S8 selected checkpoint does not match the locked configuration")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if checkpoint.get("best_epoch") != status.get("best_epoch") or checkpoint.get("epoch") != status.get("best_epoch"):
        raise RuntimeError("S8 selected checkpoint does not match the earliest-best epoch")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    metrics, diagnostics, per_building = run(
        model, dataset, device,
        [int(x) for x in lock["evaluation"]["vote_seeds"]],
        protocol["class_groups"],
    )
    report = {
        "schema_version": "1.0", "stage": "S8_MULTI_SEED_FULL_CLOUD_VALIDATION",
        "split": "validation", "metric_role": "internal_development_gate",
        "architecture": "CAMOPyramidS8_4096_1024_256_mass_aware_mha_row_softmax",
        "training_seed": seed, "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path), "evaluator_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(PREREG), "authorization_sha256": sha256(AUTH),
        "configuration_sha256": sha256(CONFIG),
        "vote_seeds": lock["evaluation"]["vote_seeds"],
        "chunk_size": CHUNK_SIZE, "inference_batch_size": INFERENCE_BATCH_SIZE,
        "metrics": metrics, "diagnostics": diagnostics, "per_building": per_building,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "max_memory_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                     "max_memory_reserved_mib": torch.cuda.max_memory_reserved()/1024**2, "amp_dtype": "bfloat16"},
        "test_split_access": False, "headline_claim_authorized": False,
        "full_cloud_test_evaluation_authorized": False,
        "protocol_verification_passed": True,
    }
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
