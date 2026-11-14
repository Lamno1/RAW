"""Paired S7 full-cloud summary and pre-registered promotion gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
ARMS = ("identity", "mass_aware_mha")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    out = ROOT / "runs" / "s7_confirmatory"
    records: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    artifact_audit = []
    for arm in ARMS:
        for seed in SEEDS:
            report_path = out / f"full_cloud_{arm}_seed{seed}.json"
            if not report_path.is_file():
                raise FileNotFoundError(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_mode = arm
            checks = {
                "split": report.get("split") == "validation",
                "architecture": report.get("architecture") == f"s7_{expected_mode}",
                "seed": int(report.get("training_seed", -1)) == seed,
                "dry_run": report.get("dry_run") is False,
                "test_unused": report.get("test_split_used") is False,
                "protocol": report.get("protocol_verification_passed") is True,
                "vote_seeds": report.get("vote_seeds") == [1103, 2207, 3301],
            }
            checkpoint = Path(report["checkpoint"])
            checks["checkpoint_hash"] = checkpoint.is_file() and sha256(checkpoint) == report.get("checkpoint_sha256")
            if not all(checks.values()):
                raise RuntimeError(f"S7 artifact failed audit: {report_path}: {checks}")
            metrics = report["metrics"]
            record = {
                "seed": seed,
                "part_iou": float(metrics["part_iou"]),
                "accuracy": float(metrics["classification_accuracy"]),
                "head_iou": float(metrics["group_iou"]["head"]),
                "mid_iou": float(metrics["group_iou"]["mid"]),
                "tail_iou": float(metrics["group_iou"]["tail"]),
                "artifact": report_path.relative_to(ROOT).as_posix(),
                "artifact_sha256": sha256(report_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            records[arm].append(record)
            artifact_audit.append({"path": report_path.relative_to(ROOT).as_posix(), "sha256": sha256(report_path), "checks": checks})

    paired: dict[str, object] = {}
    metric_map = {
        "part_iou": "part_iou", "accuracy": "accuracy",
        "head_iou": "head_iou", "mid_iou": "mid_iou", "tail_iou": "tail_iou",
    }
    for name, key in metric_map.items():
        control = np.asarray([x[key] for x in records["identity"]], dtype=np.float64)
        candidate = np.asarray([x[key] for x in records["mass_aware_mha"]], dtype=np.float64)
        delta = candidate - control
        statistic = ttest_rel(candidate, control)
        paired[name] = {
            "identity_values": control.tolist(),
            "mass_aware_values": candidate.tolist(),
            "delta_values": delta.tolist(),
            "identity_mean": float(control.mean()),
            "identity_sample_sd": float(control.std(ddof=1)),
            "mass_aware_mean": float(candidate.mean()),
            "mass_aware_sample_sd": float(candidate.std(ddof=1)),
            "delta_mean": float(delta.mean()),
            "delta_sample_sd": float(delta.std(ddof=1)),
            "paired_t": float(statistic.statistic),
            "paired_p_two_sided": float(statistic.pvalue),
        }
    delta_part = float(paired["part_iou"]["delta_mean"])
    delta_acc = float(paired["accuracy"]["delta_mean"])
    delta_middle_tail = float(paired["mid_iou"]["delta_mean"] + paired["tail_iou"]["delta_mean"])
    bench = json.loads((ROOT / "runs/s7_dry_run/benchmark.json").read_text(encoding="utf-8"))
    latency_ratio = float(bench["results"]["mass_aware_mha"]["latency_ratio_vs_identity"])
    memory_max = max(float(json.loads((out / f"{arm}_seed{seed}/hardware.json").read_text())["max_memory_allocated_mib"]) for arm in ARMS for seed in SEEDS)
    gates = {
        "part_iou_delta": {"value": delta_part, "threshold": 0.0044353341422272395, "passed": delta_part >= 0.0044353341422272395},
        "accuracy_delta": {"value": delta_acc, "threshold": -0.0043557722327992885, "passed": delta_acc >= -0.0043557722327992885},
        "not_head_only_mid_plus_tail_delta": {"value": delta_middle_tail, "threshold": 0.0, "passed": delta_middle_tail > 0.0},
        "latency_ratio": {"value": latency_ratio, "threshold": 1.25, "passed": latency_ratio <= 1.25},
        "allocated_memory_mib": {"value": memory_max, "threshold": 3200.0, "passed": memory_max <= 3200.0},
    }
    payload = {
        "schema_version": "1.0",
        "stage": "S7",
        "split": "validation",
        "seeds": list(SEEDS),
        "arms": records,
        "paired_statistics": paired,
        "promotion_gates": gates,
        "promotion_passed": all(value["passed"] for value in gates.values()),
        "artifact_audit": artifact_audit,
        "test_split_used": False,
    }
    summary = out / "paired_summary.json"
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

