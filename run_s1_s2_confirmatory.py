"""Run the frozen three-seed S1 bridge and paired S2 Lovasz study.

The schedule is intentionally single-GPU and fail-fast. Completed artifacts are
validated and skipped, so relaunching this file is idempotent at run granularity.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from camo_build.protocol import verify_protocol


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LAUNCH_LOGS = RUNS / "launcher_logs"
PROGRESS = RUNS / "s1_s2_confirmatory_progress.json"
SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    kind: str
    artifact: Path
    checkpoint: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csv_is_complete(path: Path, epochs: int = 50) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != epochs or int(rows[-1]["epoch"]) != epochs:
        return False
    return all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "epoch"
    )


def step_complete(step: Step) -> bool:
    if step.kind == "bridge":
        if not step.artifact.is_file() or not step.checkpoint.is_file():
            return False
        status = json.loads(step.artifact.read_text(encoding="utf-8"))
        return status.get("state") == "CONFIRMATORY_COMPLETE"
    return step.checkpoint.is_file() and csv_is_complete(step.artifact)


def build_steps() -> list[Step]:
    python = sys.executable
    steps: list[Step] = []
    for seed in SEEDS:
        run_dir = RUNS / f"s1_bridge_bce_p1024_seed{seed}"
        steps.append(
            Step(
                name=f"s1_bridge_seed{seed}",
                command=(
                    python, "train_bridge_baseline.py", "--points", "1024",
                    "--batch-size", "8", "--epochs", "50", "--seed", str(seed),
                    "--width", "96", "--lr", "0.001", "--balanced-ce",
                    "--num-workers", "0", "--inference-batch-size", "4",
                    "--run-dir", str(run_dir),
                ),
                kind="bridge",
                artifact=run_dir / "status.json",
                checkpoint=run_dir / "best.pth",
            )
        )
    for seed in SEEDS:
        for lovasz_weight, tag, benchmark in (
            (0.0, "bce", "benchmark_camo_density_bf16.json"),
            (0.5, "bce_lovasz0.5", "benchmark_p15_density_lovasz0.5_bf16.json"),
        ):
            name = f"camo_p15_density_{tag}_p4096_seed{seed}"
            command = [
                python, "train_camo_p1.py", "--benchmark-report", benchmark,
                "--points", "4096", "--batch-size", "8", "--width", "64",
                "--k", "16", "--epochs", "50", "--seed", str(seed),
                "--lr", "0.001", "--mass-mode", "density", "--num-workers", "0",
                "--amp-init-scale", "32768",
                "--amp-dtype", "bfloat16",
            ]
            if lovasz_weight > 0:
                command.extend(("--lovasz-weight", str(lovasz_weight)))
            steps.append(
                Step(
                    name=f"s2_{tag}_seed{seed}",
                    command=tuple(command),
                    kind="s2",
                    artifact=ROOT / "logs" / f"{name}.csv",
                    checkpoint=ROOT / "checkpoints" / f"best_{name}.pth",
                )
            )
    return steps


def main() -> None:
    verification = verify_protocol(ROOT, ROOT / "protocol" / "s0")
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if not torch.cuda.is_available():
        raise RuntimeError("confirmatory schedule requires CUDA")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 5.0:
        raise RuntimeError(f"insufficient free disk: {free_gib:.2f} GiB")
    RUNS.mkdir(exist_ok=True)
    LAUNCH_LOGS.mkdir(exist_ok=True)
    steps = build_steps()
    state: dict[str, object] = {
        "started_utc": utc_now(),
        "gpu": torch.cuda.get_device_name(0),
        "protocol_verified": True,
        "seeds": list(SEEDS),
        "steps": {},
    }
    write_json(PROGRESS, state)
    for step in steps:
        if step_complete(step):
            state["steps"][step.name] = {"state": "SKIPPED_COMPLETE", "utc": utc_now()}
            write_json(PROGRESS, state)
            continue
        log_path = LAUNCH_LOGS / f"{step.name}.log"
        state["steps"][step.name] = {
            "state": "RUNNING", "started_utc": utc_now(),
            "command": list(step.command), "log": str(log_path),
        }
        write_json(PROGRESS, state)
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(
                step.command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONUNBUFFERED": "1"}, check=False,
            )
        if result.returncode != 0 or not step_complete(step):
            state["steps"][step.name]["state"] = "FAILED"
            state["steps"][step.name]["returncode"] = result.returncode
            state["steps"][step.name]["finished_utc"] = utc_now()
            write_json(PROGRESS, state)
            raise RuntimeError(f"{step.name} failed; inspect {log_path}")
        state["steps"][step.name]["state"] = "COMPLETE"
        state["steps"][step.name]["returncode"] = result.returncode
        state["steps"][step.name]["finished_utc"] = utc_now()
        write_json(PROGRESS, state)
    state["state"] = "COMPLETE"
    state["finished_utc"] = utc_now()
    write_json(PROGRESS, state)


if __name__ == "__main__":
    main()
