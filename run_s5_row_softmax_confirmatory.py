"""Launch the frozen S5 row-softmax confirmatory schedule on one GPU.

Runs are sequential, fail-fast, and restartable at seed granularity.  This
launcher never enables SROT; the detached SROT arm remains stopped by its v1
hardware gate.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from camo_build.protocol import verify_protocol


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LOGS = RUNS / "launcher_logs"
PROGRESS = RUNS / "s5_row_softmax_progress.json"
GATE_STATUS = RUNS / "s5_detached_srot_gate" / "status.json"
ROW_AUTHORIZATION = ROOT / "protocol" / "s0" / "s5_row_softmax_authorization.json"
SEEDS = (42, 123, 2026)
EPOCHS = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_dir(seed: int) -> Path:
    return RUNS / f"s5_row_softmax_p4096_seed{seed}"


def metrics_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EPOCHS or int(rows[-1]["epoch"]) != EPOCHS:
        return False
    return all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "epoch"
    )


def seed_complete(seed: int) -> bool:
    directory = run_dir(seed)
    status_path = directory / "status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return (
        status.get("state") == "CONFIRMATORY_COMPLETE"
        and (directory / "best.pth").is_file()
        and (directory / "last.pth").is_file()
        and (directory / "hardware.json").is_file()
        and metrics_complete(directory / "metrics.csv")
    )


def assert_authorized() -> None:
    verification = verify_protocol(ROOT, ROOT / "protocol" / "s0")
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    gate = json.loads(GATE_STATUS.read_text(encoding="utf-8"))
    paired = gate["paired_latency_gate"]
    ratio = float(paired["row_softmax_vs_hard_median_ratio"])
    if not paired["row_softmax_passed"] or ratio > 1.25:
        raise RuntimeError(f"row-softmax hardware gate failed: {paired}")
    authorization = json.loads(ROW_AUTHORIZATION.read_text(encoding="utf-8"))
    if authorization.get("decision") != "AUTHORIZE_ROW_SOFTMAX_CONFIRMATORY_TRAINING":
        raise RuntimeError("row-softmax confirmatory authorization is absent")


def main() -> None:
    assert_authorized()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S5 confirmatory schedule requires BF16 CUDA")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 5.0:
        raise RuntimeError(f"insufficient free disk: {free_gib:.2f} GiB")
    RUNS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    state: dict[str, object] = {
        "stage": "S5",
        "arm": "row_softmax",
        "state": "RUNNING",
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(0),
        "protocol_verified": True,
        "seeds": list(SEEDS),
        "steps": {},
    }
    write_json(PROGRESS, state)
    for seed in SEEDS:
        name = f"s5_row_softmax_seed{seed}"
        if seed_complete(seed):
            state["steps"][name] = {
                "state": "SKIPPED_COMPLETE",
                "utc": utc_now(),
            }
            write_json(PROGRESS, state)
            continue
        command = (
            sys.executable,
            "train_camo_s5.py",
            "--run-dir",
            str(run_dir(seed)),
            "--seed",
            str(seed),
            "--points",
            "4096",
            "--proxies",
            "1024",
            "--fine-k",
            "16",
            "--proxy-k",
            "16",
            "--support-k",
            "8",
            "--width",
            "64",
            "--batch-size",
            "8",
            "--epochs",
            "50",
            "--lr",
            "0.001",
            "--amp-dtype",
            "bfloat16",
            "--num-workers",
            "0",
        )
        log_path = LOGS / f"{name}.log"
        state["steps"][name] = {
            "state": "RUNNING",
            "started_utc": utc_now(),
            "command": list(command),
            "log": str(log_path),
        }
        write_json(PROGRESS, state)
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONUNBUFFERED": "1"},
                check=False,
            )
        step = state["steps"][name]
        step["returncode"] = result.returncode
        step["finished_utc"] = utc_now()
        if result.returncode != 0 or not seed_complete(seed):
            step["state"] = "FAILED"
            state["state"] = "FAILED"
            write_json(PROGRESS, state)
            raise RuntimeError(f"{name} failed; inspect {log_path}")
        step["state"] = "COMPLETE"
        write_json(PROGRESS, state)
    state["state"] = "COMPLETE"
    state["finished_utc"] = utc_now()
    write_json(PROGRESS, state)


if __name__ == "__main__":
    main()
