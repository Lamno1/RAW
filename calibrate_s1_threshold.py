"""Lock the S1-derived practical architecture threshold before S5."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=Path("protocol/s0"))
    parser.add_argument("--control-values", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if len(args.control_values) < 3:
        parser.error("at least three frozen bridge-control values are required")
    target = args.protocol_dir / "delta_arch_calibration.json"
    if target.exists():
        raise FileExistsError(
            f"calibration is append-only and already exists: {target}"
        )
    protocol_path = args.protocol_dir / "camo_pyramid_protocol.json"
    status_path = args.protocol_dir / "status.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    values = np.asarray(args.control_values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("control values must be finite")
    sigma_pair = float(values.std(ddof=1))
    floor = float(protocol["threshold_rules"]["delta_floor_absolute_miou"])
    delta_arch = max(floor, 2.0 * sigma_pair)
    calibration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control_values": values.tolist(),
        "sigma_pair": sigma_pair,
        "delta_floor": floor,
        "delta_arch": delta_arch,
        "locked_before_s5": True,
    }
    target.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["delta_arch_calibration"] = "FROZEN"
    status["delta_arch"] = delta_arch
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
