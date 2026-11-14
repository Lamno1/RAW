"""Analyze Prototype 1 results according to Runbook §9."""
from __future__ import annotations
import json
import csv
from pathlib import Path
import numpy as np
import torch
from scipy import stats

from dataset import PART_LABELS, NUM_PART_CLASSES

def load_head_mid_tail(class_audit_path: Path):
    audit = json.loads(class_audit_path.read_text(encoding="utf-8"))
    # Sort classes by count descending
    sorted_classes = sorted(
        [(int(k) - 1, v["name"], v["count"], v["frequency"]) for k, v in audit.items()],
        key=lambda x: x[2],
        reverse=True
    )
    # Head: top 10, Mid: next 10, Tail: remaining 11
    head_indices = [x[0] for x in sorted_classes[:10]]
    mid_indices = [x[0] for x in sorted_classes[10:20]]
    tail_indices = [x[0] for x in sorted_classes[20:]]
    return sorted_classes, head_indices, mid_indices, tail_indices

def analyze():
    root = Path(".")
    checkpoint_dir = root / "checkpoints"
    log_dir = root / "logs"
    audit_path = root / "class_audit.json"
    
    sorted_classes, head_idx, mid_idx, tail_idx = load_head_mid_tail(audit_path)
    
    modes = ["none", "density"]
    seeds = [42, 123, 2026]
    
    results = {}
    
    for mode in modes:
        results[mode] = []
        for seed in seeds:
            ckpt_path = checkpoint_dir / f"best_camo_p1_{mode}_p4096_seed{seed}.pth"
            log_path = log_dir / f"camo_p1_{mode}_p4096_seed{seed}.csv"
            bench_path = root / f"benchmark_camo_{mode}.json"
            
            if not ckpt_path.exists():
                print(f"Waiting/missing checkpoint: {ckpt_path}")
                continue
                
            ckpt = torch.load(ckpt_path, map_location="cpu")
            best_epoch = ckpt["epoch"]
            best_miou = ckpt["val_miou"]
            per_class_iou = np.array(ckpt["val_per_class_iou"])
            
            bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}
            
            # Read CSV to get val_acc at best epoch
            val_acc = None
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if int(row["epoch"]) == best_epoch:
                            val_acc = float(row["val_acc"])
                            break
                            
            head_miou = np.nanmean(per_class_iou[head_idx])
            mid_miou = np.nanmean(per_class_iou[mid_idx])
            tail_miou = np.nanmean(per_class_iou[tail_idx])
            
            run_data = {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_miou": best_miou,
                "val_acc": val_acc,
                "head_miou": head_miou,
                "mid_miou": mid_miou,
                "tail_miou": tail_miou,
                "per_class_iou": per_class_iou.tolist(),
                "allocated_mib": bench.get("max_memory_allocated_mib"),
                "reserved_mib": bench.get("max_memory_reserved_mib"),
            }
            results[mode].append(run_data)

    # Paired inference is valid only because the same seeds are evaluated in
    # both modes. With n=3, report the p-value as descriptive evidence only;
    # do not label it statistically significant without a pre-registered rule.
    by_seed = {
        seed: (
            next((r["best_miou"] for r in results["none"] if r["seed"] == seed), np.nan),
            next((r["best_miou"] for r in results["density"] if r["seed"] == seed), np.nan),
        )
        for seed in seeds
    }
    paired = np.array(
        [density - none for none, density in by_seed.values() if np.isfinite(none + density)]
    )
    if paired.size >= 2:
        test = stats.ttest_1samp(paired, 0.0)
        results["paired_miou"] = {
            "seeds": [seed for seed, (none, density) in by_seed.items() if np.isfinite(none + density)],
            "deltas": paired.tolist(),
            "mean_delta": float(paired.mean()),
            "sample_std_delta": float(paired.std(ddof=1)),
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
            "interpretation": "descriptive; not conclusive with n<5",
        }
            
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    analyze()
