import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "full_cloud_validation"
PROGRESS_PATH = RUN_DIR / "progress.json"
EVALUATOR = ROOT / "evaluate_camo_full_cloud.py"
VOTE_SEEDS = [1103, 2207, 3301]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_sd(values):
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def main() -> None:
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    if progress.get("state") != "COMPLETE":
        raise RuntimeError(f"confirmatory run not complete: {progress.get('state')}")
    evaluator_hash = sha256(EVALUATOR)
    if evaluator_hash != progress["evaluator_sha256"]:
        raise RuntimeError("evaluator hash mismatch")

    records = {"s5": [], "s2": []}
    checkpoint_audit = []
    for arm in ("s5", "s2"):
        for seed in (42, 123, 2026):
            key = f"{arm}_seed{seed}"
            step = progress["steps"][key]
            artifact_path = RUN_DIR / f"{arm}_seed{seed}.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            ckpt = Path(step["checkpoint"])
            checks = {
                "artifact_exists": artifact_path.exists(),
                "returncode_zero": step["returncode"] == 0,
                "artifact_architecture": artifact["architecture"] == arm,
                "split_validation": artifact["split"] == "validation",
                "dry_run_false": artifact["dry_run"] is False,
                "test_split_unused": artifact["test_split_used"] is False,
                "vote_seeds_exact": artifact["vote_seeds"] == VOTE_SEEDS,
                "evaluator_hash_match": artifact["evaluator_sha256"] == evaluator_hash,
                "checkpoint_hash_progress_match": artifact["checkpoint_sha256"] == step["checkpoint_sha256"],
                "checkpoint_hash_disk_match": sha256(ckpt) == step["checkpoint_sha256"],
                "protocol_verification_passed": artifact["protocol_verification_passed"] is True,
            }
            if not all(checks.values()) or step["state"] != "COMPLETE":
                raise RuntimeError(f"audit failed for {key}: {checks}")
            metric = artifact["metrics"]
            records[arm].append(
                {
                    "seed": seed,
                    "artifact": str(artifact_path),
                    "checkpoint": str(ckpt),
                    "checkpoint_sha256": sha256(ckpt),
                    "part_iou": float(metric["part_iou"]),
                    "accuracy": float(metric["classification_accuracy"]),
                    "shape_iou": float(metric["shape_iou"]),
                    "head_iou": float(metric["group_iou"]["head"]),
                    "mid_iou": float(metric["group_iou"]["mid"]),
                    "tail_iou": float(metric["group_iou"]["tail"]),
                    "total_points": int(artifact["diagnostics"]["total_points"]),
                    "votes": int(artifact["diagnostics"]["votes"]),
                    "max_memory_allocated_mib": float(artifact["hardware"]["max_memory_allocated_mib"]),
                    "checks": checks,
                }
            )
            checkpoint_audit.append({"arm": arm, "seed": seed, "path": str(ckpt), "sha256": sha256(ckpt)})

    s5_miou = np.array([r["part_iou"] for r in records["s5"]], dtype=np.float64)
    s2_miou = np.array([r["part_iou"] for r in records["s2"]], dtype=np.float64)
    s5_acc = np.array([r["accuracy"] for r in records["s5"]], dtype=np.float64)
    s2_acc = np.array([r["accuracy"] for r in records["s2"]], dtype=np.float64)
    d_miou = s5_miou - s2_miou
    d_acc = s5_acc - s2_acc

    def paired_summary(a, b, delta):
        test = ttest_rel(a, b)
        return {
            "s5_values": a.tolist(),
            "s2_values": b.tolist(),
            "delta_values": delta.tolist(),
            "s5_mean": float(a.mean()),
            "s5_sd": sample_sd(a),
            "s2_mean": float(b.mean()),
            "s2_sd": sample_sd(b),
            "delta_mean": float(delta.mean()),
            "delta_sd": sample_sd(delta),
            "paired_t": float(test.statistic),
            "paired_p_two_sided": float(test.pvalue),
        }

    output = {
        "schema_version": "1.0",
        "source_progress": str(PROGRESS_PATH),
        "progress_sha256": sha256(PROGRESS_PATH),
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": evaluator_hash,
        "vote_seeds": VOTE_SEEDS,
        "arms": records,
        "checkpoint_audit": checkpoint_audit,
        "paired_statistics": {
            "part_iou": paired_summary(s5_miou, s2_miou, d_miou),
            "classification_accuracy": paired_summary(s5_acc, s2_acc, d_acc),
        },
        "all_artifact_checks_passed": True,
        "notes": [
            "Statistics are recomputed from full-cloud artifact metrics, not fixed-crop training logs.",
            "This is a three-seed full-cloud validation; the S0 headline minimum of five seeds is not yet met.",
            "SROT is not included in this confirmatory comparison; S5 uses row-softmax coarsening.",
        ],
    }
    out_path = RUN_DIR / "paired_summary.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["paired_statistics"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
