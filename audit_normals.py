"""Audit whether signed normal equations are admissible for S4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camo_build.assignment import audit_normal_orientation
from camo_dataset import CAMOBuildingNetDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-negative-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("normal_orientation_audit.json"))
    args = parser.parse_args()
    dataset = CAMOBuildingNetDataset(
        args.data_dir, args.split, num_points=32, k=8,
        augment=False, build_neighbors=False,
    )
    count = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    records: list[dict[str, object]] = []
    for index in range(count):
        sample = dataset.load_full(index)
        features = sample["features"].numpy()
        audit = audit_normal_orientation(
            features[:, :3], features[:, 3:6], k=args.k,
            max_negative_fraction=args.max_negative_fraction,
        )
        records.append({"model": sample["model_name"], **audit.to_dict()})
    signed_safe = bool(records) and all(bool(record["signed_safe"]) for record in records)
    report = {
        "split": args.split, "models_audited": len(records),
        "signed_normal_equations_allowed": signed_safe,
        "fallback_if_false": "use_abs_dot_and_abs_nz_consistently",
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

