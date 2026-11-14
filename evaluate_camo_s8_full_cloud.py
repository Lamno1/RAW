"""Locked, validation-only full-cloud evaluation for the S8 seed-42 pilot."""
from __future__ import annotations

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
PILOT_AUTH = S8 / "pilot_authorization.json"
PILOT_CONFIG = S8 / "pilot_configuration.json"
PREREG = S8 / "pilot_full_cloud_validation_preregistration.json"
RETRY_PREREG = S8 / "pilot_full_cloud_validation_retry_preregistration.json"
CHECKPOINT = ROOT / "runs" / "s8_pilot" / "mass_aware_mha_seed42" / "best.pth"
OUTPUT = ROOT / "runs" / "s8_pilot" / "mass_aware_mha_seed42" / "full_cloud_validation.json"
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


def validate_lock() -> tuple[dict[str, object], dict[str, object]]:
    if not PREREG.is_file() or not PILOT_AUTH.is_file() or not PILOT_CONFIG.is_file():
        raise FileNotFoundError("S8 pilot full-cloud preregistration/authorization is missing")
    auth = json.loads(PILOT_AUTH.read_text(encoding="utf-8"))
    config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(PREREG.read_text(encoding="utf-8"))
    expected_hashes = lock["locked_sha256"]
    actual_hashes = {
        "evaluator": sha256(Path(__file__).resolve()),
        "checkpoint": sha256(CHECKPOINT),
        "pilot_authorization": sha256(PILOT_AUTH),
        "pilot_configuration": sha256(PILOT_CONFIG),
        "s0_protocol": sha256(S0 / "camo_pyramid_protocol.json"),
        "validation_split": sha256(ROOT / "splits" / "val_split.txt"),
    }
    if not RETRY_PREREG.is_file():
        raise FileNotFoundError("S8 post-failure retry lock is missing")
    retry = json.loads(RETRY_PREREG.read_text(encoding="utf-8"))
    retry_expected = retry["locked_sha256"]
    if {key: value for key, value in expected_hashes.items() if key != "evaluator"} != {
        key: value for key, value in actual_hashes.items() if key != "evaluator"
    }:
        raise RuntimeError(f"S8 original preregistration dependency hash mismatch: {actual_hashes}")
    if retry_expected != actual_hashes or retry.get("original_preregistration_sha256") != sha256(PREREG):
        raise RuntimeError(f"S8 retry lock hash mismatch: {actual_hashes}")
    if lock.get("state") != "LOCKED_BEFORE_FULL_CLOUD_VALIDATION_METRIC_ACCESS":
        raise RuntimeError("S8 full-cloud evaluation preregistration is not locked")
    if auth.get("seed") != 42 or auth.get("epochs") != 5:
        raise RuntimeError("pilot authorization scope mismatch")
    if "val" not in auth.get("allowed_splits", []) or auth.get("test_split_access") is not False:
        raise RuntimeError("pilot authorization does not permit validation-only access")
    if auth.get("confirmatory_authorized") is not False:
        raise RuntimeError("pilot authorization cannot authorize confirmatory training")
    if lock.get("scope") != "validation_only" or lock.get("evaluation", {}).get("test_split_access") is not False:
        raise RuntimeError("full-cloud evaluation lock is not validation-only")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {OUTPUT}")
    verification = verify_protocol(ROOT, S0)
    if not verification["passed"]:
        raise RuntimeError(f"frozen S0 protocol verification failed: {verification}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S8 full-cloud evaluation requires BF16-capable CUDA")
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
        vote_seeds: list[int], class_groups: dict[str, list[int]]) -> tuple[dict[str, object], dict[str, object]]:
    model.eval()
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    total_points = 0
    total_chunks = 0
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
        meter.update(accumulator.prediction().unsqueeze(0), target.unsqueeze(0), model_names=[str(full["model_name"])])
        total_points += len(features)
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == len(dataset):
            print(f"S8 full-cloud val buildings={sample_index + 1}/{len(dataset)} points={total_points} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    elapsed = time.perf_counter() - started
    return meter.compute(class_groups), {
        "buildings": len(dataset), "points": total_points, "chunks_including_votes": total_chunks,
        "votes": len(vote_seeds), "elapsed_seconds": elapsed,
        "points_per_second_including_votes": total_points * len(vote_seeds) / max(elapsed, 1e-9),
    }


def main() -> None:
    lock, protocol = validate_lock()
    seed = int(lock["training_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = CAMOBuildingNetDataset(ROOT, "val", CHUNK_SIZE, 16, False, seed, build_neighbors=False)
    model = CAMOPyramidS8(widths=(64, 96, 128), local_blocks_per_level=2, attention_heads=4)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    metrics, diagnostics = run(
        model, dataset, device,
        [int(x) for x in lock["evaluation"]["vote_seeds"]],
        protocol["class_groups"],
    )
    report = {
        "schema_version": "1.0", "stage": "S8_SINGLE_SEED_PILOT_FULL_CLOUD_VALIDATION",
        "split": "validation", "metric_role": "descriptive_pilot_development_metric",
        "architecture": "CAMOPyramidS8_4096_1024_256_mass_aware_mha_row_softmax",
        "training_seed": seed, "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT), "evaluator_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(PREREG), "pilot_authorization_sha256": sha256(PILOT_AUTH),
        "pilot_configuration_sha256": sha256(PILOT_CONFIG),
        "retry_preregistration_sha256": sha256(RETRY_PREREG),
        "vote_seeds": lock["evaluation"]["vote_seeds"],
        "chunk_size": CHUNK_SIZE, "inference_batch_size": INFERENCE_BATCH_SIZE,
        "metrics": metrics, "diagnostics": diagnostics,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "max_memory_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                     "max_memory_reserved_mib": torch.cuda.max_memory_reserved()/1024**2, "amp_dtype": "bfloat16"},
        "test_split_access": False, "confirmatory_authorized": False,
        "full_cloud_test_evaluation_authorized": False,
        "protocol_verification_passed": True,
    }
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
