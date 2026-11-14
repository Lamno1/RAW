"""Frozen S5 row-softmax confirmatory training runner.

This runner is intentionally limited to the already-authorized row-softmax
control.  Detached SROT remains forbidden after its hardware-gate failure.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from camo_build.losses import BalancedCEWithLovasz
from camo_build.metrics import SegmentationMeter
from camo_build.protocol import verify_protocol
from camo_build.pyramid import CAMOPyramidS5
from camo_s5_dataset import CAMOS5Dataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES


ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "protocol" / "s0"
GATE_STATUS = ROOT / "runs" / "s5_detached_srot_gate" / "status.json"
OBJECTIVE_ARTIFACT = PROTOCOL_DIR / "s2_objective_decision.json"
ACCURACY_GUARDRAIL = PROTOCOL_DIR / "s5_accuracy_guardrail.json"
PREREGISTRATION = PROTOCOL_DIR / "s5_experiment_preregistration.json"
ROW_SOFTMAX_AUTHORIZATION = PROTOCOL_DIR / "s5_row_softmax_authorization.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_run_source_manifest(path: Path) -> None:
    """Freeze every Python source that can affect this S5 run."""
    sources = {
        ROOT / "train_camo_s5.py",
        ROOT / "camo_s5_dataset.py",
        ROOT / "camo_dataset.py",
        ROOT / "dataset.py",
        *set((ROOT / "camo_build").glob("*.py")),
    }
    records = [
        f"{sha256(source)}  {source.relative_to(ROOT).as_posix()}\n"
        for source in sorted(sources)
    ]
    path.write_text("".join(records), encoding="utf-8")


def validate_authorization(args: argparse.Namespace) -> dict[str, object]:
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    gate = json.loads(GATE_STATUS.read_text(encoding="utf-8"))
    paired = gate.get("paired_latency_gate", {})
    ratio = float(paired.get("row_softmax_vs_hard_median_ratio", math.inf))
    passed = bool(paired.get("row_softmax_passed", False))
    if not passed or ratio > 1.25:
        raise RuntimeError(f"row-softmax latency gate is not authorized: {paired}")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    frozen = prereg["configuration"]
    expected = {
        "points": args.points,
        "proxies": args.proxies,
        "fine_knn": args.fine_k,
        "proxy_knn": args.proxy_k,
        "support_k": args.support_k,
        "width": args.width,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "amp_dtype": args.amp_dtype,
    }
    mismatch = {key: (frozen.get(key), value) for key, value in expected.items()
                if frozen.get(key) != value}
    if mismatch and not args.dry_run:
        raise RuntimeError(f"S5 preregistration mismatch: {mismatch}")
    objective = json.loads(OBJECTIVE_ARTIFACT.read_text(encoding="utf-8"))
    if objective.get("decision") != "BCE_PLUS_0.5_LOVASZ":
        raise RuntimeError("locked S2 objective is not BCE_PLUS_0.5_LOVASZ")
    objective_hash = prereg["locked_dependencies"]["objective_sha256"]
    if sha256(OBJECTIVE_ARTIFACT) != objective_hash:
        raise RuntimeError("S2 objective artifact hash no longer matches S5 preregistration")
    guardrail = json.loads(ACCURACY_GUARDRAIL.read_text(encoding="utf-8"))
    row_authorization = json.loads(
        ROW_SOFTMAX_AUTHORIZATION.read_text(encoding="utf-8")
    )
    if (
        row_authorization.get("decision")
        != "AUTHORIZE_ROW_SOFTMAX_CONFIRMATORY_TRAINING"
    ):
        raise RuntimeError("row-softmax confirmatory authorization is absent")
    return {
        "protocol_verification": verification,
        "row_softmax_latency_ratio": ratio,
        "preregistration": prereg,
        "objective": objective,
        "accuracy_guardrail": guardrail,
        "row_softmax_authorization": row_authorization,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: CAMOPyramidS5,
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
    weighted_loss = 0.0
    valid_points = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        fine_neighbors = batch["neighbors"].to(device, non_blocking=True)
        anchor_index = batch["anchor_index"].to(device, non_blocking=True)
        support_target = batch["support_target"].to(device, non_blocking=True)
        proxy_neighbors = batch["proxy_neighbors"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        valid = target != IGNORE_INDEX
        count = int(valid.sum())
        if count <= 0:
            raise RuntimeError("all-ignore batch reached S5 runner")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(
                features, mass, fine_neighbors, anchor_index,
                support_target, proxy_neighbors,
            )
            loss = loss_fn(logits, target)
        if training:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite S5 gradient norm")
            gradient_norm_sum += float(gradient_norm.detach())
            gradient_steps += 1
            optimizer.step()
        prediction = logits.detach().argmax(-1)
        meter.update(prediction, target.detach(), model_names=batch["model_name"])
        weighted_loss += float(loss.detach()) * count
        valid_points += count
    if valid_points == 0:
        raise RuntimeError("S5 epoch produced no valid points")
    metrics = meter.compute(class_groups)
    scalars = (
        weighted_loss / valid_points,
        metrics,
        gradient_norm_sum / max(gradient_steps, 1),
    )
    if not all(math.isfinite(float(value)) for value in (
        scalars[0], metrics["part_iou"], metrics["shape_iou"],
        metrics["classification_accuracy"], scalars[2],
    )):
        raise FloatingPointError("S5 epoch produced a non-finite metric")
    return scalars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 123, 2026))
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--proxies", type=int, default=1024)
    parser.add_argument("--fine-k", type=int, default=16)
    parser.add_argument("--proxy-k", type=int, default=16)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from run-dir/last.pth and append to metrics.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    authorization = validate_authorization(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S5 confirmatory training requires BF16 CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    class_groups = authorization["preregistration"]["class_groups"] if "class_groups" in authorization["preregistration"] else json.loads(
        (PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text(encoding="utf-8")
    )["class_groups"]

    train_dataset = CAMOS5Dataset(
        args.data_dir, "train", args.points, args.fine_k, True, args.seed,
        proxies=args.proxies, support_k=args.support_k, proxy_k=args.proxy_k,
    )
    validation_dataset = CAMOS5Dataset(
        args.data_dir, "val", args.points, args.fine_k, False, args.seed,
        proxies=args.proxies, support_k=args.support_k, proxy_k=args.proxy_k,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    model = CAMOPyramidS5(
        width=args.width,
        coarsening_mode="row_softmax",
        transport_epsilon=0.2,
        transport_tau=0.0,
        transport_iterations=100,
        transport_backend="reference",
    ).to(device)
    class_weights = torch.from_numpy(
        np.load(args.data_dir / "class_weights.npy", allow_pickle=False)
    ).float().to(device)
    loss_fn = BalancedCEWithLovasz(
        class_weights, lovasz_weight=0.5, ignore_index=IGNORE_INDEX
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 1), eta_min=1e-6
    )

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "data_dir": str(args.data_dir.resolve()),
        "run_dir": str(run_dir),
        "coarsening_mode": "row_softmax",
        "transport_backend": "reference",
        "mass_mode": "density",
        "objective": "BCE_PLUS_0.5_LOVASZ",
        "device": str(device),
        "protocol_version": authorization["preregistration"]["protocol_version"],
    }
    write_json(run_dir / "config.yaml", config)
    for name in (
        "environment.txt", "split_manifest.json",
        "crop_manifest.json", "checkpoint_rule.json",
    ):
        shutil.copy2(PROTOCOL_DIR / name, run_dir / name)
    shutil.copy2(
        PROTOCOL_DIR / "source_manifest.sha256",
        run_dir / "s0_source_manifest.sha256",
    )
    write_run_source_manifest(run_dir / "source_manifest.sha256")
    write_json(run_dir / "authorization.json", {
        "row_softmax_latency_ratio": authorization["row_softmax_latency_ratio"],
        "objective_sha256": sha256(OBJECTIVE_ARTIFACT),
        "accuracy_guardrail_sha256": sha256(ACCURACY_GUARDRAIL),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "row_softmax_authorization_sha256": sha256(ROW_SOFTMAX_AUTHORIZATION),
        "verified_utc": utc_now(),
    })
    write_json(run_dir / "status.json", {
        "stage": "S5", "arm": "row_softmax",
        "state": "DRY_RUN_RUNNING" if args.dry_run else "CONFIRMATORY_RUNNING",
        "started_utc": utc_now(),
    })

    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    best_iou = -1.0
    best_epoch = 0
    epoch_count = 1 if args.dry_run else args.epochs
    start_epoch = 1
    if args.resume and last_path.is_file():
        resume_state = torch.load(last_path, map_location=device, weights_only=False)
        required = {"model", "optimizer", "scheduler", "epoch"}
        missing = required.difference(resume_state)
        if missing:
            raise RuntimeError(f"resume checkpoint missing keys: {sorted(missing)}")
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > epoch_count:
            raise RuntimeError(f"run already reached epoch {start_epoch - 1}")
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            best_iou = float(best_state.get("validation", {}).get("part_iou", -1.0))
            best_epoch = int(best_state.get("epoch", 0))
        rng = resume_state.get("rng", {})
        if "python" in rng: random.setstate(rng["python"])
        if "numpy" in rng: np.random.set_state(rng["numpy"])
        if "torch" in rng: torch.set_rng_state(rng["torch"])
        if "cuda" in rng: torch.cuda.set_rng_state_all(rng["cuda"])
        if "loader_generator" in rng: generator.set_state(rng["loader_generator"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        append_metrics = args.resume and start_epoch > 1 and metrics_path.is_file()
        with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not append_metrics:
                writer.writerow([
                    "epoch", "train_loss", "train_part_iou", "train_shape_iou",
                    "train_accuracy", "train_head_iou", "train_mid_iou",
                    "train_tail_iou", "train_gradient_norm", "val_loss",
                    "val_part_iou", "val_shape_iou", "val_accuracy",
                    "val_head_iou", "val_mid_iou", "val_tail_iou", "lr",
                ])
            for epoch in range(start_epoch, epoch_count + 1):
                train_dataset.set_epoch(epoch)
                validation_dataset.set_epoch(0)
                maximum = 1 if args.dry_run else None
                train_loss, train_metrics, gradient_norm = run_epoch(
                    model, train_loader, loss_fn, device, class_groups,
                    optimizer=optimizer, max_batches=maximum,
                )
                val_loss, val_metrics, _ = run_epoch(
                    model, validation_loader, loss_fn, device, class_groups,
                    max_batches=maximum,
                )
                scheduler.step()
                train_groups = train_metrics["group_iou"]
                val_groups = val_metrics["group_iou"]
                writer.writerow([
                    epoch, train_loss, train_metrics["part_iou"],
                    train_metrics["shape_iou"], train_metrics["classification_accuracy"],
                    train_groups["head"], train_groups["mid"], train_groups["tail"],
                    gradient_norm, val_loss, val_metrics["part_iou"],
                    val_metrics["shape_iou"], val_metrics["classification_accuracy"],
                    val_groups["head"], val_groups["mid"], val_groups["tail"],
                    optimizer.param_groups[0]["lr"],
                ])
                stream.flush()
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "validation": val_metrics,
                    "config": config,
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                        "loader_generator": generator.get_state(),
                    },
                }
                torch.save(state, last_path)
                if float(val_metrics["part_iou"]) > best_iou:
                    best_iou = float(val_metrics["part_iou"])
                    best_epoch = epoch
                    torch.save(state, best_path)
                print(
                    f"seed={args.seed} epoch={epoch:03d} "
                    f"train_piou={float(train_metrics['part_iou']):.6f} "
                    f"val_piou={float(val_metrics['part_iou']):.6f} "
                    f"val_acc={float(val_metrics['classification_accuracy']):.6f}",
                    flush=True,
                )
        hardware = {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "bf16": True,
        }
        write_json(run_dir / "hardware.json", hardware)
        write_json(run_dir / "status.json", {
            "stage": "S5", "arm": "row_softmax",
            "state": "DRY_RUN_COMPLETE" if args.dry_run else "CONFIRMATORY_COMPLETE",
            "finished_utc": utc_now(), "best_epoch": best_epoch,
            "best_validation_part_iou": best_iou,
            "segmentation_metrics_inspected": True,
        })
    except Exception as exc:
        write_json(run_dir / "status.json", {
            "stage": "S5", "arm": "row_softmax", "state": "FAILED",
            "failed_utc": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
