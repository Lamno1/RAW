"""Sequential full-cloud validation for frozen S5 and paired S2 checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from camo_build.protocol import verify_protocol


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "full_cloud_validation"
LOG_DIR = OUTPUT_DIR / "logs"
PROGRESS = OUTPUT_DIR / "progress.json"
EVALUATOR = ROOT / "evaluate_camo_full_cloud.py"
SEEDS = (42, 123, 2026)
VOTES = (1103, 2207, 3301)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def output_path(architecture: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{architecture}_seed{seed}.json"


def checkpoint_path(architecture: str, seed: int) -> Path:
    if architecture == "s5":
        return ROOT / "runs" / f"s5_row_softmax_p4096_seed{seed}" / "best.pth"
    return (
        ROOT / "checkpoints"
        / f"best_camo_p15_density_bce_lovasz0.5_p4096_seed{seed}.pth"
    )


def complete(architecture: str, seed: int, evaluator_hash: str) -> bool:
    path = output_path(architecture, seed)
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return (
        report.get("architecture") == architecture
        and report.get("training_seed") == seed
        and report.get("vote_seeds") == list(VOTES)
        and report.get("dry_run") is False
        and report.get("evaluator_sha256") == evaluator_hash
        and report.get("checkpoint_sha256") == sha256(checkpoint_path(architecture, seed))
        and report.get("test_split_used") is False
        and report.get("protocol_verification_passed") is True
    )


def main() -> None:
    verification = verify_protocol(ROOT, ROOT / "protocol" / "s0")
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-cloud validation requires BF16 CUDA")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    evaluator_hash = sha256(EVALUATOR)
    state: dict[str, object] = {
        "state": "RUNNING",
        "started_utc": utc_now(),
        "protocol_verified": True,
        "evaluator_sha256": evaluator_hash,
        "vote_seeds": list(VOTES),
        "steps": {},
    }
    atomic_json(PROGRESS, state)
    steps = [("s5", seed) for seed in SEEDS] + [("s2", seed) for seed in SEEDS]
    for architecture, seed in steps:
        name = f"{architecture}_seed{seed}"
        if complete(architecture, seed, evaluator_hash):
            state["steps"][name] = {"state": "SKIPPED_COMPLETE", "utc": utc_now()}
            atomic_json(PROGRESS, state)
            continue
        checkpoint = checkpoint_path(architecture, seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = output_path(architecture, seed)
        command = (
            sys.executable,
            str(EVALUATOR),
            "--architecture", architecture,
            "--checkpoint", str(checkpoint),
            "--output", str(output),
            "--training-seed", str(seed),
        )
        log_path = LOG_DIR / f"{name}.log"
        state["steps"][name] = {
            "state": "RUNNING",
            "started_utc": utc_now(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "log": str(log_path),
        }
        atomic_json(PROGRESS, state)
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
        if result.returncode != 0 or not complete(architecture, seed, evaluator_hash):
            step["state"] = "FAILED"
            state["state"] = "FAILED"
            atomic_json(PROGRESS, state)
            raise RuntimeError(f"{name} failed; inspect {log_path}")
        step["state"] = "COMPLETE"
        atomic_json(PROGRESS, state)
    state["state"] = "COMPLETE"
    state["finished_utc"] = utc_now()
    atomic_json(PROGRESS, state)


if __name__ == "__main__":
    main()
