"""Prospective validation-only latency decomposition for frozen S7/S8 checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

from camo_build.metrics import LogitVoteAccumulator, full_cloud_chunks
from camo_build.protocol import verify_protocol
from camo_build.s7 import CAMOPyramidS7
from camo_build.pyramid_s8 import CAMOPyramidS8
from camo_dataset import CAMOBuildingNetDataset
from dataset import NUM_PART_CLASSES
from evaluate_camo_full_cloud import build_chunk as build_s7_chunk, stack_chunks as stack_s7
from evaluate_camo_s8_multiseed_full_cloud import make_chunk as build_s8_chunk, stack as stack_s8


ROOT = Path(__file__).resolve().parent
S0 = ROOT / "protocol" / "s0"
PREREG = ROOT / "protocol" / "s8" / "latency_profile_preregistration.json"
AUTH = ROOT / "protocol" / "s8" / "latency_profile_authorization.json"
RUN_DIR = ROOT / "runs" / "s8_latency_profile"
OUTPUT = RUN_DIR / "latency_profile.json"
STATUS = RUN_DIR / "status.json"
CHECKPOINTS = {
    "s7": ROOT / "runs" / "s7_confirmatory" / "mass_aware_mha_seed42" / "best.pth",
    "s8": ROOT / "runs" / "s8_multiseed" / "seed42" / "best.pth",
}
DEPENDENCIES = {
    "profiler": Path(__file__).resolve(),
    "s7_checkpoint": CHECKPOINTS["s7"],
    "s8_checkpoint": CHECKPOINTS["s8"],
    "s7_model": ROOT / "camo_build" / "s7.py",
    "s8_model": ROOT / "camo_build" / "pyramid_s8.py",
    "s7_evaluator": ROOT / "evaluate_camo_s7_full_cloud.py",
    "s7_chunk_builder": ROOT / "evaluate_camo_full_cloud.py",
    "s8_evaluator": ROOT / "evaluate_camo_s8_multiseed_full_cloud.py",
    "s7_topology": ROOT / "camo_s5_dataset.py",
    "s8_topology": ROOT / "camo_s8_dataset.py",
    "base_dataset": ROOT / "camo_dataset.py",
    "dataset_constants": ROOT / "dataset.py",
    "assignment": ROOT / "camo_build" / "assignment.py",
    "neighbors": ROOT / "camo_build" / "neighbors.py",
    "coarsening": ROOT / "camo_build" / "coarsening.py",
    "transport": ROOT / "camo_build" / "transport.py",
    "sparse_ops": ROOT / "camo_build" / "sparse_ops.py",
    "triton_transport": ROOT / "camo_build" / "triton_transport.py",
    "warmstart_transport": ROOT / "camo_build" / "warmstart_transport.py",
    "operator": ROOT / "camo_build" / "operator.py",
    "s7_attention": ROOT / "camo_build" / "s7_attention.py",
    "metrics": ROOT / "camo_build" / "metrics.py",
    "s0_protocol": S0 / "camo_pyramid_protocol.json",
    "validation_split": ROOT / "splits" / "val_split.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def preflight() -> tuple[dict, dict]:
    if OUTPUT.exists() or STATUS.exists():
        raise FileExistsError("latency profile output already exists; refusing to overwrite")
    if not PREREG.is_file() or not AUTH.is_file():
        raise FileNotFoundError("latency profile preregistration/authorization missing")
    lock = json.loads(PREREG.read_text(encoding="utf-8"))
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if lock.get("state") != "FROZEN_BEFORE_LATENCY_DATA_ACCESS":
        raise RuntimeError("latency profile preregistration not frozen")
    if auth.get("state") != "AUTHORIZED_BY_USER_DIRECTION_BEFORE_DATA_ACCESS":
        raise RuntimeError("latency profile not authorized")
    if auth.get("preregistration_sha256") != sha256(PREREG):
        raise RuntimeError("latency profile authorization hash mismatch")
    if lock.get("test_split_access") is not False or auth.get("test_split_access") is not False:
        raise RuntimeError("test split access forbidden")
    actual = {key: sha256(path) for key, path in DEPENDENCIES.items()}
    if lock.get("locked_sha256") != actual:
        raise RuntimeError("latency profile dependency hash mismatch")
    expected = lock["measurement"]
    if (expected.get("split") != "val" or expected.get("training_seed") != 42
            or expected.get("chunk_size") != 4096 or expected.get("vote_seed") != 1103
            or expected.get("batch_sizes") != [1, 4] or expected.get("repetitions") != 2
            or expected.get("building_indices") != [0, 26, 53, 80, 107, 134, 161, 186]):
        raise RuntimeError("latency measurement parameters differ from the frozen design")
    verification = verify_protocol(ROOT, S0)
    if not verification["passed"]:
        raise RuntimeError("S0 protocol verification failed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA required")
    return lock, {"authorization_sha256": sha256(AUTH), "protocol_verification": verification}


def load_models(device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {
        "s7": CAMOPyramidS7(width=64, global_mode="mass_aware_mha", attention_heads=4),
        "s8": CAMOPyramidS8(widths=(64, 96, 128), local_blocks_per_level=2,
                            attention_heads=4),
    }
    for name, model in models.items():
        checkpoint = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
    return models


def build_one(name: str, features: torch.Tensor, mass: torch.Tensor, indices) -> dict:
    if name == "s7":
        return build_s7_chunk(features, mass, indices, architecture="s5", fine_k=16,
                              proxies=256, support_k=8, proxy_k=16)
    return build_s8_chunk(features, mass, indices)


def forward_one(name: str, model: torch.nn.Module, batch: dict) -> torch.Tensor:
    if name == "s7":
        return model(batch["features"], batch["mass"], batch["neighbors"],
                     batch["anchor_index"], batch["support_target"],
                     batch["proxy_neighbors"])
    return model(batch["features"], batch["mass"], batch["neighbors0"],
                 batch["anchor1"], batch["support01"], batch["neighbors1"],
                 batch["anchor2"], batch["support12"], batch["neighbors2"])


@torch.inference_mode()
def measure_condition(name: str, batch_size: int, model: torch.nn.Module,
                      features: torch.Tensor, mass: torch.Tensor, chunks: list,
                      device: torch.device) -> dict:
    stages = {key: 0.0 for key in ("topology", "stack", "transfer", "forward",
                                    "finite_check", "vote_and_coverage")}
    accumulator = LogitVoteAccumulator(len(features), NUM_PART_CLASSES)
    started = time.perf_counter()
    for offset in range(0, len(chunks), batch_size):
        selected = chunks[offset:offset + batch_size]
        t = time.perf_counter()
        cpu_chunks = [build_one(name, features, mass, indices) for indices, _ in selected]
        stages["topology"] += time.perf_counter() - t

        t = time.perf_counter()
        stacked = stack_s7(cpu_chunks) if name == "s7" else stack_s8(cpu_chunks)
        stages["stack"] += time.perf_counter() - t

        torch.cuda.synchronize(device)
        t = time.perf_counter()
        batch = {key: value.to(device, non_blocking=True) for key, value in stacked.items()}
        torch.cuda.synchronize(device)
        stages["transfer"] += time.perf_counter() - t

        t = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_one(name, model, batch)
        torch.cuda.synchronize(device)
        stages["forward"] += time.perf_counter() - t

        t = time.perf_counter()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"{name} nonfinite logits")
        stages["finite_check"] += time.perf_counter() - t

        t = time.perf_counter()
        for row, (indices, valid_count) in enumerate(selected):
            accumulator.update(torch.from_numpy(indices[:valid_count]).long(),
                               logits[row, :valid_count])
        stages["vote_and_coverage"] += time.perf_counter() - t

    t = time.perf_counter()
    if not torch.all(accumulator.count == 1):
        raise RuntimeError(f"{name} vote coverage mismatch")
    stages["vote_and_coverage"] += time.perf_counter() - t
    total = time.perf_counter() - started
    return {
        "architecture": name,
        "batch_size": batch_size,
        "chunks": len(chunks),
        "forward_calls": math.ceil(len(chunks) / batch_size),
        "points": len(features),
        "stage_seconds": stages,
        "stage_accounting_residual_seconds": total - sum(stages.values()),
        "total_seconds": total,
        "coverage_passed": True,
    }


def main() -> None:
    lock, context = preflight()
    measurement = lock["measurement"]
    atomic_json(STATUS, {"state": "RUNNING", "stage": "S7_S8_LATENCY_PROFILE",
                         "started_utc": time.time(), "test_split_access": False})
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
        models = load_models(device)
        dataset = CAMOBuildingNetDataset(ROOT, "val", 4096, 16, False, 42,
                                         build_neighbors=False)
        if len(dataset) != 187:
            raise RuntimeError("validation building count changed")
        records = []
        loads = []
        orders = [(("s7", 1), ("s8", 1), ("s8", 4), ("s7", 4)),
                  (("s7", 4), ("s8", 4), ("s8", 1), ("s7", 1))]
        if measurement["condition_order_by_repetition"] != [
            [list(pair) for pair in order] for order in orders
        ]:
            raise RuntimeError("condition order differs from lock")
        for sample_index in measurement["building_indices"]:
            t = time.perf_counter()
            full = dataset.load_full(sample_index)
            load_seconds = time.perf_counter() - t
            features, mass = full["features"], full["mass"]
            t = time.perf_counter()
            chunks = full_cloud_chunks(len(features), 4096,
                                       measurement["vote_seed"] + sample_index)
            chunking_seconds = time.perf_counter() - t
            loads.append({"building_index": sample_index, "model_name": str(full["model_name"]),
                          "load_seconds": load_seconds, "chunking_seconds": chunking_seconds,
                          "chunks": len(chunks), "points": len(features)})
            for repetition, order in enumerate(orders, start=1):
                for name, batch_size in order:
                    record = measure_condition(name, batch_size, models[name],
                                               features, mass, chunks, device)
                    record["building_index"] = sample_index
                    record["repetition"] = repetition
                    records.append(record)
                    print(f"profile building={sample_index} rep={repetition} "
                          f"arch={name} batch={batch_size} total_s={record['total_seconds']:.2f}",
                          flush=True)
        aggregates = {}
        for name in ("s7", "s8"):
            for batch_size in (1, 4):
                group = [record for record in records if record["architecture"] == name
                         and record["batch_size"] == batch_size]
                key = f"{name}_batch{batch_size}"
                aggregates[key] = {
                    "runs": len(group),
                    "chunks": sum(record["chunks"] for record in group),
                    "forward_calls": sum(record["forward_calls"] for record in group),
                    "total_seconds": sum(record["total_seconds"] for record in group),
                    "median_building_run_seconds": statistics.median(record["total_seconds"] for record in group),
                    "stage_seconds": {stage: sum(record["stage_seconds"][stage] for record in group)
                                      for stage in group[0]["stage_seconds"]},
                }
                aggregates[key]["seconds_per_chunk"] = (
                    aggregates[key]["total_seconds"] / aggregates[key]["chunks"])
        result = {
            "schema_version": "1.0",
            "stage": "S7_S8_LATENCY_DECOMPOSITION",
            "state": "COMPLETE",
            "scope": "validation_subset_timing_only_no_metric_or_promotion_retest",
            "preregistration_sha256": sha256(PREREG),
            "authorization_sha256": context["authorization_sha256"],
            "profiler_sha256": sha256(Path(__file__).resolve()),
            "checkpoint_sha256": {name: sha256(path) for name, path in CHECKPOINTS.items()},
            "measurement": measurement,
            "loads": loads,
            "records": records,
            "aggregates": aggregates,
            "hardware": {"gpu": torch.cuda.get_device_name(0),
                         "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2},
            "test_split_access": False,
            "protocol_verification_passed": context["protocol_verification"]["passed"],
            "architecture_promotion_reopened": False,
        }
        atomic_json(OUTPUT, result)
        atomic_json(STATUS, {"state": "COMPLETE", "stage": "S7_S8_LATENCY_PROFILE",
                             "finished_utc": time.time(), "result_sha256": sha256(OUTPUT),
                             "test_split_access": False})
    except Exception as error:
        atomic_json(STATUS, {"state": "FAILED", "stage": "S7_S8_LATENCY_PROFILE",
                             "failed_utc": time.time(), "error_type": type(error).__name__,
                             "error": str(error), "test_split_access": False})
        raise


if __name__ == "__main__":
    main()
