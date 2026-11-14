"""Authorized single-seed S8 feasibility pilot with resumable checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from camo_build.losses import BalancedCEWithLovasz
from camo_build.metrics import SegmentationMeter
from camo_build.protocol import verify_protocol
from camo_build.pyramid_s8 import CAMOPyramidS8
from camo_s8_dataset import CAMOS8Dataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol" / "s0"
CONFIG = ROOT / "protocol" / "s8" / "pilot_configuration.json"
AUTHORIZATION = ROOT / "protocol" / "s8" / "pilot_authorization.json"
PREREG = ROOT / "protocol" / "s0" / "mutations" / "s8_architecture_preregistration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_authorization() -> tuple[dict[str, object], dict[str, object]]:
    """Hash and protocol checks happen before constructing any dataset."""
    verification = verify_protocol(ROOT, PROTOCOL)
    if not verification["passed"]:
        raise RuntimeError(f"frozen S0 verification failed: {verification}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    auth = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if auth.get("state") != "SINGLE_SEED_PILOT_AUTHORIZED_BEFORE_DATA_ACCESS":
        raise RuntimeError("S8 pilot authorization is absent or not in authorized state")
    if auth.get("test_split_access") is not False:
        raise RuntimeError("S8 pilot authorization must forbid test split access")
    expected = auth.get("locked_sha256", {})
    paths = {
        "pilot_configuration": CONFIG,
        "architecture_preregistration": PREREG,
        "s8_model": ROOT / "camo_build" / "pyramid_s8.py",
        "s8_dataset": ROOT / "camo_s8_dataset.py",
        "training_runner": Path(__file__).resolve(),
        "dry_run_runner": ROOT / "benchmark_s8_dry_run.py",
        "dry_run_result": ROOT / "runs" / "s8_dry_run" / "benchmark.json",
        "s2_objective": PROTOCOL / "s2_objective_decision.json",
    }
    for key, path in paths.items():
        actual = sha256(path)
        if expected.get(key) != actual:
            raise RuntimeError(f"S8 authorization hash mismatch for {key}: {path}")
    if auth.get("seed") != config["training"]["seed"] or auth.get("epochs") != config["training"]["epochs"]:
        raise RuntimeError("S8 pilot config and authorization disagree")
    if config["data_and_leakage"]["test_split_access"] is not False:
        raise RuntimeError("S8 configuration unexpectedly permits test split")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S8 pilot requires CUDA with native BF16 support")
    return config, {"authorization": auth, "s0_verification": verification}


def _move(batch: dict[str, object], key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch field {key} is not a tensor")
    return value.to(device, non_blocking=True)


def run_epoch(
    model: CAMOPyramidS8,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    class_groups: dict[str, list[int]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, object], float]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    loss_weighted, valid_points, grad_norm_sum, grad_steps = 0.0, 0, 0.0, 0
    for batch in loader:
        features, mass, target = (_move(batch, key, device) for key in ("features", "mass", "label"))
        valid = target != IGNORE_INDEX
        count = int(valid.sum())
        if count <= 0:
            raise RuntimeError("all-ignore S8 batch")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda", dtype=torch.bfloat16
        ):
            logits = model(
                features, mass,
                *(_move(batch, key, device) for key in (
                    "neighbors0", "anchor1", "support01", "neighbors1",
                    "anchor2", "support12", "neighbors2",
                )),
            )
            loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite S8 loss")
        if training:
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"missing/non-finite gradient at {name}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            grad_norm_sum += float(grad_norm.detach())
            grad_steps += 1
            optimizer.step()
        meter.update(logits.detach().argmax(-1), target.detach(), model_names=batch["model_name"])
        loss_weighted += float(loss.detach()) * count
        valid_points += count
    metrics = meter.compute(class_groups)
    if not valid_points:
        raise RuntimeError("empty S8 epoch")
    return loss_weighted / valid_points, metrics, grad_norm_sum / max(grad_steps, 1)


def source_manifest(path: Path, *, verify: bool) -> None:
    sources = [
        ROOT / "train_camo_s8.py", ROOT / "camo_s8_dataset.py",
        ROOT / "camo_dataset.py", ROOT / "dataset.py",
        ROOT / "camo_build" / "pyramid_s8.py",
        ROOT / "camo_build" / "coarsening.py",
        ROOT / "camo_build" / "operator.py",
        ROOT / "camo_build" / "transport.py",
        ROOT / "camo_build" / "sparse_ops.py",
        ROOT / "camo_build" / "s7_attention.py",
        ROOT / "camo_build" / "losses.py",
        ROOT / "camo_build" / "metrics.py",
        ROOT / "camo_build" / "assignment.py",
        ROOT / "camo_build" / "neighbors.py",
    ]
    current = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in sorted(sources)
    )
    if verify and path.read_text(encoding="utf-8") != current:
        raise RuntimeError("S8 source changed since checkpoint; refusing resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "s8_pilot" / "mass_aware_mha_seed42")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Do not construct train/validation datasets before this complete hash audit.
    config, authorization_context = validate_authorization()
    cfg = config["training"]
    frozen_protocol = json.loads((PROTOCOL / "camo_pyramid_protocol.json").read_text(encoding="utf-8"))
    class_groups = frozen_protocol["class_groups"]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pth", run_dir / "best.pth"
    manifest_path = run_dir / "source_manifest.sha256"
    if args.resume:
        if not last_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--resume requires last.pth and source manifest")
        source_manifest(manifest_path, verify=True)
    elif any(run_dir.iterdir()):
        previous_status = run_dir / "status.json"
        if not previous_status.is_file():
            raise FileExistsError(f"run directory is not empty: {run_dir}; use --resume")
        prior = json.loads(previous_status.read_text(encoding="utf-8"))
        prior_metrics = run_dir / "metrics.csv"
        if (
            prior.get("state") != "FAILED"
            or last_path.exists()
            or best_path.exists()
            or not prior_metrics.is_file()
            or len(prior_metrics.read_text(encoding="utf-8").splitlines()) > 1
        ):
            raise FileExistsError(f"run directory has prior artifacts: {run_dir}; use --resume")
        attempt_record = run_dir / "attempt_001_failed.json"
        if attempt_record.exists():
            raise FileExistsError(f"failed attempt is already preserved: {attempt_record}")
        shutil.copy2(previous_status, attempt_record)
        shutil.copy2(prior_metrics, run_dir / "attempt_001_metrics_header.csv")
        prior_metrics.unlink()
        previous_status.unlink()
    else:
        source_manifest(manifest_path, verify=False)

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda")
    data_root = ROOT
    train_dataset = CAMOS8Dataset(data_root, "train", int(cfg["points"]), 16, True, int(cfg["seed"]))
    val_dataset = CAMOS8Dataset(data_root, "val", int(cfg["points"]), 16, False, int(cfg["seed"]))
    loader_generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=loader_generator, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    model = CAMOPyramidS8().to(device)
    class_weights = torch.from_numpy(np.load(ROOT / "class_weights.npy", allow_pickle=False)).float().to(device)
    objective = BalancedCEWithLovasz(
        class_weights, lovasz_weight=0.5, ignore_index=IGNORE_INDEX
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(cfg["epochs"]), eta_min=1e-6
    )
    start_epoch, best_iou, best_epoch = 1, -1.0, 0
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_iou, best_epoch = float(state["best_iou"]), int(state["best_epoch"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        loader_generator.set_state(state["rng"]["loader_generator"])
        if start_epoch > int(cfg["epochs"]):
            raise RuntimeError("pilot already completed")

    for name in ("environment.txt", "split_manifest.json", "crop_manifest.json", "checkpoint_rule.json"):
        shutil.copy2(PROTOCOL / name, run_dir / name)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "authorization_record.json", {
        "authorization_sha256": sha256(AUTHORIZATION),
        "pilot_configuration_sha256": sha256(CONFIG),
        "verified_utc": utc_now(),
        "s0_verification": authorization_context["s0_verification"],
        "test_split_access": False,
    })
    write_json(run_dir / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "packages": sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in metadata.distributions() if d.metadata.get("Name")
        ),
    })
    write_json(run_dir / "status.json", {
        "stage": "S8", "phase": "pilot", "state": "RUNNING",
        "started_utc": utc_now(), "seed": cfg["seed"],
        "test_split_used": False,
    })
    metrics_path = run_dir / "metrics.csv"
    append = args.resume and start_epoch > 1 and metrics_path.is_file()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with metrics_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not append:
                writer.writerow([
                    "epoch", "train_loss", "train_part_iou", "train_accuracy",
                    "train_head_iou", "train_mid_iou", "train_tail_iou", "train_gradient_norm",
                    "val_loss", "val_part_iou", "val_accuracy", "val_head_iou",
                    "val_mid_iou", "val_tail_iou", "lr",
                ])
            for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
                train_dataset.set_epoch(epoch)
                val_dataset.set_epoch(0)
                train_loss, train_metric, grad_norm = run_epoch(
                    model, train_loader, objective, device, class_groups, optimizer=optimizer
                )
                val_loss, val_metric, _ = run_epoch(
                    model, val_loader, objective, device, class_groups
                )
                scheduler.step()
                tg, vg = train_metric["group_iou"], val_metric["group_iou"]
                writer.writerow([
                    epoch, train_loss, train_metric["part_iou"], train_metric["classification_accuracy"],
                    tg["head"], tg["mid"], tg["tail"], grad_norm,
                    val_loss, val_metric["part_iou"], val_metric["classification_accuracy"],
                    vg["head"], vg["mid"], vg["tail"], optimizer.param_groups[0]["lr"],
                ])
                stream.flush()
                if float(val_metric["part_iou"]) > best_iou:
                    best_iou, best_epoch = float(val_metric["part_iou"]), epoch
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_iou": best_iou, "best_epoch": best_epoch,
                    "validation": val_metric, "config_sha256": sha256(CONFIG),
                    "authorization_sha256": sha256(AUTHORIZATION),
                    "rng": {
                        "python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "loader_generator": loader_generator.get_state(),
                    },
                }
                torch.save(state, last_path.with_suffix(".pth.tmp"))
                os.replace(last_path.with_suffix(".pth.tmp"), last_path)
                if best_epoch == epoch:
                    torch.save(state, best_path.with_suffix(".pth.tmp"))
                    os.replace(best_path.with_suffix(".pth.tmp"), best_path)
                print(
                    f"S8 pilot epoch={epoch}/{cfg['epochs']} val_part_iou={best_iou:.6f} "
                    f"val_accuracy={float(val_metric['classification_accuracy']):.6f} "
                    f"grad_norm={grad_norm:.5f}", flush=True,
                )
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "amp_dtype": "bfloat16",
        }
        if hardware["max_memory_allocated_mib"] > 3200.0:
            raise RuntimeError("S8 pilot exceeded the frozen VRAM limit")
        write_json(run_dir / "hardware.json", hardware)
        write_json(run_dir / "status.json", {
            "stage": "S8", "phase": "pilot", "state": "PILOT_COMPLETE",
            "finished_utc": utc_now(), "best_epoch": best_epoch,
            "best_validation_part_iou": best_iou,
            "test_split_used": False, "training_authorized": True,
        })
    except Exception as error:
        write_json(run_dir / "status.json", {
            "stage": "S8", "phase": "pilot", "state": "FAILED",
            "failed_utc": utc_now(), "error_type": type(error).__name__,
            "error": str(error), "test_split_used": False,
        })
        raise
    finally:
        train_dataset and getattr(train_loader, "_iterator", None) and train_loader._iterator._shutdown_workers()
        val_dataset and getattr(val_loader, "_iterator", None) and val_loader._iterator._shutdown_workers()


if __name__ == "__main__":
    main()
