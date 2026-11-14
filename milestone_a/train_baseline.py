"""Milestone A: budget-matched PointNeXt-S baseline training runner.

Mirrors ``train_camo_s7.py`` (same data class, loss, optimizer, schedule,
gradient clipping, checkpoint rule and VRAM gate).  It refuses to start unless
the Milestone A preregistration is frozen and every locked file hash matches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from camo_build.losses import BalancedCEWithLovasz  # noqa: E402
from camo_build.metrics import SegmentationMeter  # noqa: E402
from camo_build.protocol import verify_protocol  # noqa: E402
from camo_dataset import CAMOBuildingNetDataset  # noqa: E402
from dataset import IGNORE_INDEX, NUM_PART_CLASSES  # noqa: E402
from pointnext_baseline import PointNeXtBaseline, parameter_count  # noqa: E402

PROTOCOL_DIR = ROOT / "protocol" / "s0"
MILESTONE_DIR = ROOT / "protocol" / "milestone_a"
PREREG = MILESTONE_DIR / "baseline_budget_matched_preregistration.json"
LR_SELECTION = MILESTONE_DIR / "lr_selection.json"
VRAM_LIMIT_MIB = 3200.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Files whose hashes are locked in the preregistration under these keys.
LOCKED_FILES = {
    "baseline_model_source": HERE / "pointnext_baseline.py",
    "pointnet2_torch_shim": HERE / "pointnet2_torch_shim.py",
    "baseline_config": HERE / "cfgs" / "pointnext-s_buildingnet.yaml",
    "baseline_training_runner": HERE / "train_baseline.py",
    "baseline_evaluator_adapter": HERE / "evaluate_baseline_full_cloud.py",
    "environment_lockfile": HERE / "environment_lock.txt",
    "s7_paired_validation_summary": ROOT / "runs" / "s7_confirmatory" / "paired_summary.json",
    "s0_protocol": PROTOCOL_DIR / "camo_pyramid_protocol.json",
    "validation_split": PROTOCOL_DIR / "split_manifest.json",
    "metrics_implementation": ROOT / "camo_build" / "metrics.py",
    "losses": ROOT / "camo_build" / "losses.py",
}


def validate_frozen_preregistration(args: argparse.Namespace) -> dict[str, object]:
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    if not PREREG.is_file():
        raise RuntimeError("Milestone A preregistration is not frozen (file missing)")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("state") != "FROZEN_BEFORE_MILESTONE_A_DATA_ACCESS":
        raise RuntimeError("Milestone A preregistration is not in the frozen state")
    locked = prereg["locked_sha256"]
    for key, path in LOCKED_FILES.items():
        expected = locked.get(key)
        if expected is None:
            raise RuntimeError(f"locked hash missing for {key}")
        if sha256(path) != expected:
            raise RuntimeError(f"locked file changed after freeze: {key} ({path})")
    matching = prereg["matching_conditions"]
    if args.points != matching["points_per_crop"]:
        raise RuntimeError("points per crop differ from preregistration")
    allowance = prereg["hyperparameter_allowance"]
    if args.phase == "search":
        if args.seed != allowance["search_seed"] or args.epochs != allowance["search_epochs"]:
            raise RuntimeError("search seed/epochs differ from preregistration")
        if args.lr not in allowance["learning_rates"]:
            raise RuntimeError("learning rate is not in the preregistered search set")
    elif args.phase == "confirmatory":
        if args.seed not in matching["training_seeds"] or args.epochs != matching["epochs"]:
            raise RuntimeError("confirmatory seed/epochs differ from preregistration")
        if not LR_SELECTION.is_file():
            raise RuntimeError("learning-rate selection has not been recorded")
        selection = json.loads(LR_SELECTION.read_text(encoding="utf-8"))
        if selection.get("state") != "FROZEN_BEFORE_CONFIRMATORY_METRICS":
            raise RuntimeError("learning-rate selection is not frozen")
        if selection.get("preregistration_sha256") != sha256(PREREG):
            raise RuntimeError("learning-rate selection refers to a different preregistration")
        if args.lr != selection["selected_learning_rate"]:
            raise RuntimeError("learning rate differs from the frozen selection")
    return {"verification": verification, "preregistration": prereg}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    class_groups: dict[str, list[int]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, object], float]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    weighted_loss, valid_points = 0.0, 0
    gradient_norm_sum, gradient_steps = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        count = int((target != IGNORE_INDEX).sum())
        if count <= 0:
            raise RuntimeError("all-ignore batch reached the baseline runner")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
        ):
            logits = model(features)
            loss = loss_fn(logits, target)
        if training:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite baseline gradient norm")
            gradient_norm_sum += float(gradient_norm.detach())
            gradient_steps += 1
            optimizer.step()
        meter.update(logits.detach().argmax(-1), target.detach(), model_names=batch["model_name"])
        weighted_loss += float(loss.detach()) * count
        valid_points += count
    if valid_points == 0:
        raise RuntimeError("baseline epoch produced no valid points")
    metrics = meter.compute(class_groups)
    scalars = (weighted_loss / valid_points, metrics, gradient_norm_sum / max(gradient_steps, 1))
    finite = (scalars[0], metrics["part_iou"], metrics["shape_iou"],
              metrics["classification_accuracy"], scalars[2])
    if not all(math.isfinite(float(value)) for value in finite):
        raise FloatingPointError("baseline epoch produced a non-finite metric")
    return scalars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "search", "confirmatory"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.phase == "smoke":
        # Smoke runs measure nothing and support no claim, so a failed S0 check is
        # recorded in authorization.json instead of stopping the mechanics test.
        verification = verify_protocol(ROOT, PROTOCOL_DIR)
        authorization = {"verification": verification, "preregistration": None}
    else:
        authorization = validate_frozen_preregistration(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the baseline requires BF16 CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    protocol = json.loads((PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text())
    class_groups = protocol["class_groups"]
    train_dataset = CAMOBuildingNetDataset(
        args.data_dir, "train", args.points, 16, True, args.seed, build_neighbors=False
    )
    validation_dataset = CAMOBuildingNetDataset(
        args.data_dir, "val", args.points, 16, False, args.seed, build_neighbors=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = PointNeXtBaseline().to(device)
    class_weights = torch.from_numpy(
        np.load(args.data_dir / "class_weights.npy", allow_pickle=False)
    ).float().to(device)
    loss_fn = BalancedCEWithLovasz(class_weights, lovasz_weight=0.5, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 1), eta_min=1e-6
    )

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data_dir": str(args.data_dir.resolve()), "run_dir": str(run_dir),
              "objective": "BCE_PLUS_0.5_LOVASZ", "architecture": "pointnext_s_reduced_to_budget",
              "parameters": parameter_count(model), "protocol_version": protocol["version"]}
    write_json(run_dir / "config.yaml", config)
    for name in ("environment.txt", "split_manifest.json", "crop_manifest.json", "checkpoint_rule.json"):
        shutil.copy2(PROTOCOL_DIR / name, run_dir / name)
    locked_manifest = "".join(
        f"{sha256(path)}  {key}\n" for key, path in sorted(LOCKED_FILES.items()) if path.is_file()
    )
    manifest_path = run_dir / "source_manifest.sha256"
    if args.resume and manifest_path.is_file() and manifest_path.read_text(encoding="utf-8") != locked_manifest:
        raise RuntimeError("source changed since checkpoint; refusing an inexact resume")
    manifest_path.write_text(locked_manifest, encoding="utf-8")
    write_json(run_dir / "authorization.json", {
        "preregistration_sha256": sha256(PREREG) if PREREG.is_file() else None,
        "protocol_verification": authorization["verification"], "verified_utc": utc_now(),
    })
    write_json(run_dir / "status.json", {
        "stage": "MILESTONE_A", "phase": args.phase, "state": f"{args.phase.upper()}_RUNNING",
        "started_utc": utc_now(),
    })

    metrics_path, best_path, last_path = run_dir / "metrics.csv", run_dir / "best.pth", run_dir / "last.pth"
    start_epoch, best_iou, best_epoch = 1, -1.0, 0
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else state
        best_iou = float(best_checkpoint.get("validation", {}).get("part_iou", -1.0))
        best_epoch = int(best_checkpoint.get("epoch", 0))
        rng = state.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "loader_generator" in rng:
            generator.set_state(rng["loader_generator"])
        if start_epoch > args.epochs:
            raise RuntimeError("run has already completed all epochs")
    torch.cuda.reset_peak_memory_stats()
    step_seconds: list[float] = []
    try:
        append = args.resume and start_epoch > 1 and metrics_path.is_file()
        with metrics_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not append:
                writer.writerow([
                    "epoch", "train_loss", "train_part_iou", "train_shape_iou", "train_accuracy",
                    "train_head_iou", "train_mid_iou", "train_tail_iou", "train_gradient_norm",
                    "val_loss", "val_part_iou", "val_shape_iou", "val_accuracy",
                    "val_head_iou", "val_mid_iou", "val_tail_iou", "lr", "train_epoch_seconds",
                ])
            for epoch in range(start_epoch, args.epochs + 1):
                train_dataset.set_epoch(epoch)
                validation_dataset.set_epoch(0)
                maximum = 2 if args.phase == "smoke" else None
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                train_loss, train_metrics, gradient_norm = run_epoch(
                    model, train_loader, loss_fn, device, class_groups,
                    optimizer=optimizer, max_batches=maximum,
                )
                finished.record()
                torch.cuda.synchronize()
                epoch_seconds = started.elapsed_time(finished) / 1000.0
                val_loss, val_metrics, _ = run_epoch(
                    model, validation_loader, loss_fn, device, class_groups, max_batches=maximum
                )
                scheduler.step()
                tg, vg = train_metrics["group_iou"], val_metrics["group_iou"]
                writer.writerow([
                    epoch, train_loss, train_metrics["part_iou"], train_metrics["shape_iou"],
                    train_metrics["classification_accuracy"], tg["head"], tg["mid"], tg["tail"],
                    gradient_norm, val_loss, val_metrics["part_iou"], val_metrics["shape_iou"],
                    val_metrics["classification_accuracy"], vg["head"], vg["mid"], vg["tail"],
                    optimizer.param_groups[0]["lr"], epoch_seconds,
                ])
                stream.flush()
                step_seconds.append(epoch_seconds)
                if float(val_metrics["part_iou"]) > best_iou:
                    best_iou, best_epoch = float(val_metrics["part_iou"]), epoch
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_iou": best_iou, "best_epoch": best_epoch,
                    "validation": val_metrics, "config": config,
                    "rng": {
                        "python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "loader_generator": generator.get_state(),
                    },
                }
                atomic_torch_save(state, last_path)
                if best_epoch == epoch:
                    atomic_torch_save(state, best_path)
                print(f"phase={args.phase} seed={args.seed} lr={args.lr} epoch={epoch:03d} "
                      f"val_piou={best_iou:.6f} val_acc={float(val_metrics['classification_accuracy']):.6f} "
                      f"epoch_s={epoch_seconds:.1f}", flush=True)
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "amp_dtype": "bfloat16", "parameters": parameter_count(model),
            "median_train_epoch_seconds": float(np.median(step_seconds)) if step_seconds else None,
        }
        if hardware["max_memory_allocated_mib"] > VRAM_LIMIT_MIB:
            raise RuntimeError("baseline VRAM limit exceeded")
        write_json(run_dir / "hardware.json", hardware)
        write_json(run_dir / "status.json", {
            "stage": "MILESTONE_A", "phase": args.phase, "state": f"{args.phase.upper()}_COMPLETE",
            "finished_utc": utc_now(), "best_epoch": best_epoch,
            "best_validation_part_iou": best_iou, "test_split_used": False,
        })
    except Exception as exc:
        write_json(run_dir / "status.json", {
            "stage": "MILESTONE_A", "phase": args.phase, "state": "FAILED", "failed_utc": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
