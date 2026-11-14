"""S0 protocol freezing and reproducibility manifests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_VERSION = "camo-pyramid-s0-v1"
OFFICIAL_EVALUATION_SOURCE = (
    "https://eval.ai/web/challenges/challenge-page/1938/evaluation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _class_groups(counts: np.ndarray) -> dict[str, list[int]]:
    if counts.shape != (31,) or np.any(counts < 0):
        raise ValueError("class_counts.npy must contain 31 non-negative counts")
    descending = np.argsort(-counts, kind="stable")
    return {
        "head": descending[:10].astype(int).tolist(),
        "mid": descending[10:20].astype(int).tolist(),
        "tail": descending[20:].astype(int).tolist(),
    }


def _source_manifest(root: Path, output: Path) -> list[tuple[str, str]]:
    included: list[Path] = []
    for suffix in ("*.py", "*.md"):
        included.extend(root.rglob(suffix))
    records: list[tuple[str, str]] = []
    for path in sorted(set(included)):
        relative_parts = path.relative_to(root).parts
        scratch_dir = any(
            part == "graphify-out"
            or part == ".pytest_cache"
            or part.startswith("pytest_basetemp")
            or part.startswith("pytest-of-")
            or part.startswith(".pytest")
            or part.startswith(".")
            for part in relative_parts[:-1]
        )
        if output in path.parents or scratch_dir:
            continue
        records.append((path.relative_to(root).as_posix(), _sha256(path)))
    return records


def assert_external_temp(root: str | Path, basetemp: str | Path) -> Path:
    """Reject pytest scratch directories nested below the protocol root."""
    root_path = Path(root).resolve()
    temp_path = Path(basetemp).resolve()
    if temp_path == root_path or root_path in temp_path.parents:
        raise ValueError(
            f"basetemp must be outside protocol root; got {temp_path} under {root_path}"
        )
    return temp_path


def freeze_protocol(
    root: str | Path,
    output: str | Path,
    *,
    crop_seeds: tuple[int, ...] = (1103, 2207, 3301),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze S0 without reading predictions or test metrics."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    if (output / "status.json").exists() and not overwrite:
        raise FileExistsError(
            f"protocol is already frozen at {output}; pass overwrite=True explicitly"
        )
    count_path = root / "class_counts.npy"
    if not count_path.is_file():
        raise FileNotFoundError(f"missing train-only counts: {count_path}")
    counts = np.load(count_path, allow_pickle=False).astype(np.int64, copy=False)

    split_manifest: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / "splits" / f"{split}_split.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing split: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        duplicates = sorted(name for name in names if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate names in {split}: {duplicates[:3]}")
        overlap = sorted(name for name in names if name in seen)
        if overlap:
            raise ValueError(f"split leakage between {split} and {seen[overlap[0]]}")
        seen.update({name: split for name in names})
        split_manifest[split] = {
            "count": len(names), "sha256": _sha256(path), "models": names
        }

    groups = _class_groups(counts)
    protocol: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_evaluation_source": OFFICIAL_EVALUATION_SOURCE,
        "official_point_metrics": {
            "part_iou": "mean global per-class IoU over classes with non-zero union",
            "shape_iou": "mean over buildings of mean per-class IoU over non-zero unions",
            "classification_accuracy": "correct valid points divided by all valid points",
        },
        "diagnostic_metrics": [
            "head_mid_tail_iou", "boundary_iou", "latency",
            "throughput", "max_memory_allocated_mib", "max_memory_reserved_mib",
        ],
        "primary_development_metric": "full_cloud_validation_part_iou",
        "checkpoint_selection_metric": "fixed_crop_validation_part_iou",
        "label_zero_policy": "geometry_context_ignore_loss_metric",
        "normal_geometry_policy": "sign_invariant_until_complete_orientation_audit_passes",
        "class_counts_sha256": _sha256(count_path),
        "ignore_index": -100,
        "class_groups": groups,
        "crop_seeds": list(crop_seeds),
        "full_cloud_votes": len(crop_seeds),
        "maximum_single_seed_exploratory_configs_per_component": 8,
        "confirmatory_seeds": [42, 123, 2026],
        "headline_minimum_seeds": 5,
        "threshold_rules": {
            "delta_floor_absolute_miou": 0.002,
            "delta_arch": "max(0.002, 2*sigma_pair); populate after S1 before S5",
            "miou_noninferiority_margin": 0.001,
            "tail_absolute_gain": 0.001,
            "maximum_latency_ratio": 1.25,
            "maximum_allocated_vram_mib": 3200.0,
            "relative_mass_residual": 1e-6,
            "adjoint_relative_error": 1e-5,
        },
        "test_policy": "hash/name audit allowed; predictions and metrics forbidden until S12",
    }
    checkpoint_rule = {
        "selection_split": "validation",
        "selection_metric": protocol["checkpoint_selection_metric"],
        "mode": "max",
        "tie_break": "earliest_epoch",
        "test_selection_forbidden": True,
    }
    crop_manifest = {
        "seeds": list(crop_seeds),
        "coverage": "each vote permutes the full cloud and chunks it without omission",
        "padding": "last chunk repeats points but padded logits are excluded from accumulation",
    }
    source_records = _source_manifest(root, output)
    source_text = "".join(f"{digest}  {relative}\n" for relative, digest in source_records)
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    environment = "\n".join(
        [
            f"python={sys.version.replace(chr(10), ' ')}",
            f"platform={platform.platform()}",
            f"torch={torch.__version__}",
            f"cuda_runtime={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
            "", "[packages]", *packages,
        ]
    ) + "\n"

    _atomic_json(output / "camo_pyramid_protocol.json", protocol)
    _atomic_json(output / "split_manifest.json", split_manifest)
    _atomic_json(output / "crop_manifest.json", crop_manifest)
    _atomic_json(output / "checkpoint_rule.json", checkpoint_rule)
    _atomic_text(output / "source_manifest.sha256", source_text)
    _atomic_text(output / "environment.txt", environment)
    calibration_path = output / "delta_arch_calibration.json"
    status: dict[str, Any] = {
        "stage": "S0", "state": "FROZEN", "protocol_version": PROTOCOL_VERSION,
        "delta_arch_calibration": "PENDING_S1",
    }
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        delta_arch = float(calibration["delta_arch"])
        if not np.isfinite(delta_arch) or delta_arch <= 0.0:
            raise ValueError("existing delta_arch calibration is invalid")
        status["delta_arch_calibration"] = "FROZEN"
        status["delta_arch"] = delta_arch
    _atomic_json(output / "status.json", status)
    return protocol


def verify_protocol(root: str | Path, protocol_dir: str | Path) -> dict[str, Any]:
    """Verify source, split and train-count hashes against a frozen S0."""
    root = Path(root).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    protocol = json.loads(
        (protocol_dir / "camo_pyramid_protocol.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (protocol_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    missing: list[str] = []
    changed: list[str] = []
    unreadable: list[str] = []
    manifest_path = protocol_dir / "source_manifest.sha256"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        try:
            if not path.is_file():
                missing.append(relative)
            elif _sha256(path) != expected:
                changed.append(relative)
        except OSError:
            unreadable.append(relative)
    split_changed: list[str] = []
    for split, record in split_manifest.items():
        path = root / "splits" / f"{split}_split.txt"
        if not path.is_file() or _sha256(path) != record["sha256"]:
            split_changed.append(split)
    count_path = root / "class_counts.npy"
    counts_match = (
        count_path.is_file()
        and _sha256(count_path) == protocol["class_counts_sha256"]
    )
    passed = not missing and not changed and not unreadable and not split_changed and counts_match
    return {
        "passed": passed,
        "missing_sources": missing,
        "changed_sources": changed,
        "unreadable_sources": unreadable,
        "changed_splits": split_changed,
        "class_counts_match": counts_match,
        "protocol_version": protocol["version"],
    }
