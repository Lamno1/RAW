"""One authorized, frozen test-set evaluation campaign for the selected S7 model."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from camo_build.protocol import verify_protocol
from camo_build.s7 import CAMOPyramidS7
from camo_dataset import CAMOBuildingNetDataset
from dataset import NUM_PART_CLASSES
from evaluate_camo_full_cloud import evaluate


ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "protocol" / "s0"
AUTH_PATH = ROOT / "protocol" / "s7" / "s7_architecture_freeze_test_authorization.json"
RUN_DIR = ROOT / "runs" / "s7_test"
SEEDS = (42, 123, 2026)
METRIC_PATHS = {
    "part_iou": ("part_iou",),
    "accuracy": ("classification_accuracy",),
    "head_iou": ("group_iou", "head"),
    "mid_iou": ("group_iou", "mid"),
    "tail_iou": ("group_iou", "tail"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metric_value(metrics: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = metrics
    for part in path:
        value = value[part]
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"test summary metric is non-finite: {path}={number}")
    return number


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} frozen seed reports, got {len(reports)}")
    summary: dict[str, Any] = {}
    for metric, path in METRIC_PATHS.items():
        values = [_metric_value(report["metrics"], path) for report in reports]
        summary[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
        }
    class_values: list[list[float | None]] = [
        list(report["metrics"]["per_class_iou"]) for report in reports
    ]
    per_class: list[dict[str, Any]] = []
    for class_index in range(NUM_PART_CLASSES):
        values = [row[class_index] for row in class_values]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        per_class.append({
            "class_id": class_index + 1,
            "values": values,
            "mean_over_defined_seed_metrics": float(np.mean(finite)) if finite else None,
            "defined_seed_count": len(finite),
        })
    summary["per_class_iou"] = per_class
    return summary


def validate_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if authorization.get("state") != "ARCHITECTURE_FROZEN_TEST_AUTHORIZED":
        raise RuntimeError("No frozen S7 test authorization is active")
    freeze = authorization["independent_gate_audit"]
    if freeze.get("all_gates_passed") is not True:
        raise RuntimeError("S7 promotion gates are not all passed")
    summary_path = ROOT / authorization["decision_basis"]["s7_paired_summary"]
    if sha256(summary_path) != authorization["decision_basis"]["s7_paired_summary_sha256"]:
        raise RuntimeError("S7 paired validation summary hash changed after architecture freeze")
    promotion = json.loads(summary_path.read_text(encoding="utf-8"))
    if promotion.get("promotion_passed") is not True or promotion.get("test_split_used") is not False:
        raise RuntimeError("S7 validation promotion artifact is not eligible for test")
    if tuple(authorization["frozen_test_protocol"]["vote_seeds"]) != (1103, 2207, 3301):
        raise RuntimeError("Frozen test voting seeds mismatch")
    if tuple(model["seed"] for model in authorization["frozen_test_protocol"]["models"]) != SEEDS:
        raise RuntimeError("Frozen test checkpoint seeds mismatch")
    evaluator_spec = authorization["frozen_test_protocol"]["evaluator"]
    evaluator_path = ROOT / evaluator_spec["path"]
    if evaluator_path.resolve() != Path(__file__).resolve() or sha256(evaluator_path) != evaluator_spec["sha256"]:
        raise RuntimeError("Frozen test evaluator source hash mismatch")
    for model in authorization["frozen_test_protocol"]["models"]:
        checkpoint = ROOT / model["checkpoint"]
        if not checkpoint.is_file() or sha256(checkpoint) != model["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen S7 checkpoint hash mismatch: {checkpoint}")
    return authorization, verification


def main() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S7 test evaluation requires BF16 CUDA")
    authorization, verification = validate_authorization()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "test_summary.json"
    if summary_path.exists():
        raise FileExistsError("Frozen test summary already exists; this campaign cannot be repeated")

    protocol = json.loads((PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text(encoding="utf-8"))
    class_groups = protocol["class_groups"]
    expected_vote_seeds = list(authorization["frozen_test_protocol"]["vote_seeds"])
    reports: list[dict[str, Any]] = []

    # This is the first code path in the project that constructs the test-split dataset.
    for model_spec in authorization["frozen_test_protocol"]["models"]:
        seed = int(model_spec["seed"])
        checkpoint_path = ROOT / model_spec["checkpoint"]
        output_path = RUN_DIR / f"test_full_cloud_seed{seed}.json"
        if output_path.exists():
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                report.get("split") != "test"
                or report.get("test_split_used") is not True
                or report.get("architecture") != "s7_mass_aware_mha"
                or int(report.get("training_seed", -1)) != seed
                or report.get("checkpoint_sha256") != model_spec["checkpoint_sha256"]
                or report.get("test_authorization_sha256") != sha256(AUTH_PATH)
                or report.get("vote_seeds") != expected_vote_seeds
                or report.get("evaluator_sha256") != sha256(Path(__file__).resolve())
            ):
                raise RuntimeError(f"Existing test output fails its frozen contract: {output_path}")
            reports.append(report)
            continue

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = CAMOBuildingNetDataset(
            ROOT, "test", 4096, 16, False, seed, build_neighbors=False
        )
        model = CAMOPyramidS7(width=64, global_mode="mass_aware_mha", attention_heads=4)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        device = torch.device("cuda")
        model.to(device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics, diagnostics = evaluate(
            model,
            dataset,
            device,
            architecture="s5",
            vote_seeds=expected_vote_seeds,
            class_groups=class_groups,
            chunk_size=4096,
            inference_batch_size=4,
            fine_k=16,
            proxies=256,
            support_k=8,
            proxy_k=16,
            max_models=None,
        )
        report = {
            "schema_version": "1.0",
            "stage": "S12",
            "split": "test",
            "metric_role": "one_time_final_test_evaluation",
            "architecture": "s7_mass_aware_mha",
            "training_seed": seed,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            "test_authorization": str(AUTH_PATH.relative_to(ROOT).as_posix()),
            "test_authorization_sha256": sha256(AUTH_PATH),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "base_voting_evaluator_sha256": sha256(ROOT / "evaluate_camo_full_cloud.py"),
            "vote_seeds": expected_vote_seeds,
            "chunk_size": 4096,
            "proxies": 256,
            "support_k": 8,
            "proxy_k": 16,
            "inference_batch_size": 4,
            "metrics": metrics,
            "diagnostics": {**diagnostics, "wall_seconds": time.perf_counter() - started},
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "amp_dtype": "bfloat16",
            },
            "test_split_used": True,
            "protocol_verification_passed": verification["passed"],
        }
        atomic_json(output_path, report)
        reports.append(json.loads(output_path.read_text(encoding="utf-8")))

    result = {
        "schema_version": "1.0",
        "stage": "S12",
        "split": "test",
        "architecture": "s7_mass_aware_mha",
        "authorization_sha256": sha256(AUTH_PATH),
        "validation_summary_sha256": authorization["decision_basis"]["s7_paired_summary_sha256"],
        "seeds": list(SEEDS),
        "evaluation_campaign_count": 1,
        "aggregation_is_descriptive_only": True,
        "metrics": summarize_reports(reports),
        "per_seed_artifacts": [
            {
                "path": (RUN_DIR / f"test_full_cloud_seed{seed}.json").relative_to(ROOT).as_posix(),
                "sha256": sha256(RUN_DIR / f"test_full_cloud_seed{seed}.json"),
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            for seed, report in zip(SEEDS, reports, strict=True)
        ],
        "test_split_used": True,
    }
    atomic_json(summary_path, result)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
