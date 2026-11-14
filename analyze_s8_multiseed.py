"""Independently audit the completed S8 three-seed internal validation gate."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "s8_multiseed"
PREREG = ROOT / "protocol" / "s8" / "multiseed_preregistration.json"
AUTH = ROOT / "protocol" / "s8" / "multiseed_authorization.json"
CONFIG = ROOT / "protocol" / "s8" / "multiseed_configuration.json"
EVALUATOR = ROOT / "evaluate_camo_s8_multiseed_full_cloud.py"
S7_SUMMARY = ROOT / "runs" / "s7_confirmatory" / "paired_summary.json"
OUTPUT = RUNS / "paired_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, label: str) -> None:
    require(math.isfinite(left) and math.isfinite(right), f"nonfinite {label}")
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), f"{label}: {left} != {right}")


def zero_confusion(classes: int) -> list[list[int]]:
    return [[0] * classes for _ in range(classes)]


def add_confusion(total: list[list[int]], other: list[list[int]]) -> None:
    require(len(other) == len(total), "confusion height mismatch")
    for index, row in enumerate(other):
        require(len(row) == len(total), "confusion width mismatch")
        for column, value in enumerate(row):
            require(isinstance(value, int) and value >= 0, "invalid confusion count")
            total[index][column] += value


def confusion_metrics(confusion: list[list[int]]) -> dict:
    classes = len(confusion)
    target = [sum(row) for row in confusion]
    predicted = [sum(confusion[row][column] for row in range(classes)) for column in range(classes)]
    intersection = [confusion[index][index] for index in range(classes)]
    union = [target[index] + predicted[index] - intersection[index] for index in range(classes)]
    iou = [intersection[index] / union[index] if union[index] else None for index in range(classes)]
    present = [value for value in iou if value is not None]
    shape_values = [float(torch.tensor(intersection[index]) / torch.tensor(union[index]))
                    for index in range(classes) if union[index]]
    valid = sum(target)
    require(valid > 0 and bool(present), "empty confusion")
    return {
        "classification_accuracy": sum(intersection) / valid,
        "part_iou": sum(present) / len(present),
        "shape_iou": sum(shape_values) / len(shape_values),
        "per_class_iou": iou,
        "valid_points": valid,
    }


def audit_seed(seed: int, prereg: dict, config: dict, groups: dict) -> dict:
    directory = RUNS / f"seed{seed}"
    status_path = directory / "status.json"
    metrics_path = directory / "metrics.csv"
    best_path = directory / "best.pth"
    last_path = directory / "last.pth"
    hardware_path = directory / "hardware.json"
    report_path = directory / "full_cloud_validation.json"
    status, hardware, report = (read_json(path) for path in (status_path, hardware_path, report_path))
    require(status["state"] == "TRAINING_COMPLETE" and status["test_split_used"] is False,
            f"seed {seed} training incomplete or test touched")
    require(report["split"] == "validation" and report["training_seed"] == seed,
            f"seed {seed} validation identity mismatch")
    require(report["test_split_access"] is False and report["full_cloud_test_evaluation_authorized"] is False,
            f"seed {seed} test access flag")
    require(report["headline_claim_authorized"] is False and report["protocol_verification_passed"] is True,
            f"seed {seed} interpretation/protocol flag")
    require(report["vote_seeds"] == config["evaluation"]["vote_seeds"] and
            report["diagnostics"]["votes"] == len(config["evaluation"]["vote_seeds"]),
            f"seed {seed} vote mismatch")
    require(report["diagnostics"]["buildings"] == 187 and report["diagnostics"]["points"] == 18_700_000,
            f"seed {seed} validation coverage mismatch")
    for field, path in (
        ("checkpoint_sha256", best_path), ("evaluator_sha256", EVALUATOR),
        ("preregistration_sha256", PREREG), ("authorization_sha256", AUTH),
        ("configuration_sha256", CONFIG),
    ):
        require(report[field] == sha256(path), f"seed {seed} {field} mismatch")

    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == config["training"]["epochs"], f"seed {seed} epoch count")
    require([int(row["epoch"]) for row in rows] == list(range(1, len(rows) + 1)),
            f"seed {seed} epoch sequence")
    require(all(math.isfinite(float(value)) for row in rows for key, value in row.items() if key != "epoch"),
            f"seed {seed} nonfinite training metric")
    best_epoch = max(range(1, len(rows) + 1), key=lambda epoch: float(rows[epoch - 1]["val_part_iou"]))
    best_iou = float(rows[best_epoch - 1]["val_part_iou"])
    require(status["best_epoch"] == best_epoch, f"seed {seed} earliest-best selection mismatch")
    close(float(status["best_validation_part_iou"]), best_iou, f"seed {seed} selected IoU")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    require(checkpoint["epoch"] == checkpoint["best_epoch"] == best_epoch,
            f"seed {seed} best checkpoint epoch mismatch")
    require(last_checkpoint["epoch"] == config["training"]["epochs"], f"seed {seed} last checkpoint epoch")
    close(float(checkpoint["best_iou"]), best_iou, f"seed {seed} checkpoint IoU")
    for field, path in (("config_sha256", CONFIG), ("authorization_sha256", AUTH),
                        ("preregistration_sha256", PREREG)):
        require(checkpoint[field] == last_checkpoint[field] == sha256(path),
                f"seed {seed} checkpoint {field} mismatch")
    require(set(checkpoint["rng"]) == {"python", "numpy", "torch", "cuda", "loader_generator"},
            f"seed {seed} incomplete resume checkpoint")

    per_building = report["per_building"]
    classes = len(report["metrics"]["confusion"])
    require(len(per_building) == report["diagnostics"]["buildings"], f"seed {seed} building count")
    require(len({building["model_name"] for building in per_building}) == len(per_building),
            f"seed {seed} duplicate building")
    total = zero_confusion(classes)
    building_ious = []
    for building in per_building:
        matrix = building["confusion"]
        add_confusion(total, matrix)
        result = confusion_metrics(matrix)
        require(building["valid_points"] == result["valid_points"],
                f"seed {seed} building valid point count")
        close(float(building["shape_iou"]), result["shape_iou"],
              f"seed {seed} building shape IoU")
        building_ious.append(result["shape_iou"])
    require(total == report["metrics"]["confusion"], f"seed {seed} aggregate confusion mismatch")
    recomputed = confusion_metrics(total)
    close(report["metrics"]["classification_accuracy"], recomputed["classification_accuracy"],
          f"seed {seed} accuracy")
    close(report["metrics"]["part_iou"], recomputed["part_iou"], f"seed {seed} part IoU")
    close(report["metrics"]["shape_iou"], sum(building_ious) / len(building_ious),
          f"seed {seed} shape IoU")
    for index, value in enumerate(recomputed["per_class_iou"]):
        reported = report["metrics"]["per_class_iou"][index]
        if value is None:
            require(reported is None, f"seed {seed} class {index} absent IoU")
        else:
            close(float(reported), value, f"seed {seed} class {index} IoU")
    for group_name, indices in groups.items():
        values = [recomputed["per_class_iou"][index] for index in indices
                  if recomputed["per_class_iou"][index] is not None]
        close(float(report["metrics"]["group_iou"][group_name]), sum(values) / len(values),
              f"seed {seed} {group_name} IoU")
    require(report["metrics"]["num_shapes"] == len(per_building), f"seed {seed} num_shapes")
    require(hardware["max_memory_allocated_mib"] <=
            config["comparison"]["maximum_allocated_vram_mib"], f"seed {seed} VRAM gate")
    require(report["diagnostics"]["elapsed_seconds"] > 0, f"seed {seed} elapsed time")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "training_metrics_sha256": sha256(metrics_path),
        "training_status_sha256": sha256(status_path),
        "best_checkpoint_sha256": sha256(best_path),
        "last_checkpoint_sha256": sha256(last_path),
        "full_cloud_validation_sha256": sha256(report_path),
        "full_cloud_part_iou": recomputed["part_iou"],
        "full_cloud_accuracy": recomputed["classification_accuracy"],
        "full_cloud_shape_iou_recomputed": sum(building_ious) / len(building_ious),
        "full_cloud_tail_iou": report["metrics"]["group_iou"]["tail"],
        "full_cloud_elapsed_seconds": report["diagnostics"]["elapsed_seconds"],
        "max_training_allocated_vram_mib": hardware["max_memory_allocated_mib"],
        "buildings_audited": len(per_building),
        "valid_points_audited": recomputed["valid_points"],
        "test_split_used": False,
    }


def main() -> None:
    require(not OUTPUT.exists(), "refusing to overwrite existing S8 summary")
    prereg, auth, config = (read_json(path) for path in (PREREG, AUTH, CONFIG))
    require(prereg["state"] == "FROZEN_BEFORE_MULTI_SEED_DATA_ACCESS", "preregistration state")
    require(auth["state"] == "AUTHORIZED_AFTER_INDEPENDENT_AUDIT", "authorization state")
    require(auth["preregistration_sha256"] == sha256(PREREG), "authorization prereg hash")
    require(prereg["locked_sha256"]["s7_paired_validation_summary"] == sha256(S7_SUMMARY),
            "S7 control hash changed")
    require(config["test_split_access"] is False and auth["test_split_access"] is False,
            "test split policy")
    orchestration = read_json(RUNS / "orchestration" / "status.json")
    require(orchestration["state"] == "COMPLETE" and orchestration["exit_code"] == 0,
            "orchestration not complete")
    groups = read_json(ROOT / "protocol" / "s0" / "camo_pyramid_protocol.json")["class_groups"]
    seeds = config["training"]["seeds"]
    require(seeds == prereg["training"]["seeds"] == auth["seeds"], "paired seed lock")
    s8 = [audit_seed(seed, prereg, config, groups) for seed in seeds]
    s7_reports = [read_json(ROOT / "runs" / "s7_confirmatory" /
                            f"full_cloud_mass_aware_mha_seed{seed}.json") for seed in seeds]
    s7 = []
    for seed, report in zip(seeds, s7_reports):
        require(report["training_seed"] == seed and report["split"] == "validation" and
                report["test_split_used"] is False, f"S7 seed {seed} identity")
        s7.append({
            "seed": seed,
            "full_cloud_validation_sha256": sha256(ROOT / "runs" / "s7_confirmatory" /
                                                   f"full_cloud_mass_aware_mha_seed{seed}.json"),
            "full_cloud_part_iou": report["metrics"]["part_iou"],
            "full_cloud_accuracy": report["metrics"]["classification_accuracy"],
            "full_cloud_tail_iou": report["metrics"]["group_iou"]["tail"],
            "full_cloud_elapsed_seconds": report["diagnostics"]["elapsed_seconds"],
        })
    def paired(name: str) -> list[float]:
        return [new[name] - old[name] for new, old in zip(s8, s7)]

    part_deltas = paired("full_cloud_part_iou")
    accuracy_deltas = paired("full_cloud_accuracy")
    tail_deltas = paired("full_cloud_tail_iou")
    mean = lambda values: sum(values) / len(values)
    latency_ratio = sum(row["full_cloud_elapsed_seconds"] for row in s8) / sum(
        row["full_cloud_elapsed_seconds"] for row in s7)
    rules = config["comparison"]
    gates = {
        "part_iou_gain": {
            "value": mean(part_deltas),
            "threshold": rules["mean_paired_part_iou_gain_minimum"],
            "passed": mean(part_deltas) >= rules["mean_paired_part_iou_gain_minimum"],
        },
        "accuracy_delta": {
            "value": mean(accuracy_deltas),
            "threshold": rules["mean_paired_accuracy_delta_minimum"],
            "passed": mean(accuracy_deltas) >= rules["mean_paired_accuracy_delta_minimum"],
        },
        "tail_iou_gain": {
            "value": mean(tail_deltas),
            "threshold": rules["mean_paired_tail_iou_gain_minimum"],
            "passed": mean(tail_deltas) >= rules["mean_paired_tail_iou_gain_minimum"],
        },
        "latency_ratio": {
            "value": latency_ratio,
            "threshold": rules["maximum_latency_ratio_vs_s7"],
            "passed": latency_ratio <= rules["maximum_latency_ratio_vs_s7"],
        },
        "training_vram": {
            "value": max(row["max_training_allocated_vram_mib"] for row in s8),
            "threshold": rules["maximum_allocated_vram_mib"],
            "passed": all(row["max_training_allocated_vram_mib"] <=
                          rules["maximum_allocated_vram_mib"] for row in s8),
        },
        "finite_training_metrics": {"passed": True},
        "test_split_unused": {"passed": True},
    }
    summary = {
        "schema_version": "1.0",
        "stage": "S8_MULTI_SEED_INTERNAL_GATE",
        "state": "AUDITED_COMPLETE",
        "metric_role": "internal_validation_gate_only",
        "preregistration_sha256": sha256(PREREG),
        "authorization_sha256": sha256(AUTH),
        "configuration_sha256": sha256(CONFIG),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "orchestration_status_sha256": sha256(RUNS / "orchestration" / "status.json"),
        "s8": s8,
        "s7_control": s7,
        "paired_deltas": {
            "part_iou": part_deltas,
            "accuracy": accuracy_deltas,
            "tail_iou": tail_deltas,
        },
        "gates": gates,
        "internal_promotion_passed": all(gate["passed"] for gate in gates.values()),
        "headline_claim_authorized": False,
        "test_split_used": False,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "gates": gates,
                      "internal_promotion_passed": summary["internal_promotion_passed"]}, indent=2))


if __name__ == "__main__":
    main()
