"""Train CAMO-Build Prototype 1 after its independent hardware gate passes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from camo_build.losses import BalancedCEWithLovasz, StaticBalancedCrossEntropy
from camo_build.model import CAMOPrototype1
from camo_dataset import CAMOBuildingNetDataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES


def validate_benchmark(path: Path, args: argparse.Namespace) -> dict[str, object]:
    """Refuse training when the exact configuration lacks a passing VRAM gate."""
    report = json.loads(path.read_text(encoding="utf-8"))
    config = report["config"]
    expected = {
        "batch_size": args.batch_size,
        "points": args.points,
        "width": args.width,
        "k": args.k,
        "mass_mode": args.mass_mode,
        "amp_dtype": args.amp_dtype,
    }
    if args.lovasz_weight > 0.0:
        expected["lovasz_weight"] = args.lovasz_weight
    expected["amp_init_scale"] = args.amp_init_scale
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"benchmark configuration mismatch: {mismatches}")
    allocated = float(report["max_memory_allocated_mib"])
    if allocated > 3200.0:
        raise RuntimeError(f"benchmark failed VRAM gate: {allocated:.1f} MiB")
    return report


def update_confusion(
    confusion: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor
) -> None:
    valid = target != IGNORE_INDEX
    encoded = target[valid] * NUM_PART_CLASSES + prediction[valid]
    confusion += torch.bincount(
        encoded.cpu(), minlength=NUM_PART_CLASSES**2
    ).reshape(NUM_PART_CLASSES, NUM_PART_CLASSES)


def metrics(confusion: torch.Tensor) -> tuple[float, float, list[float]]:
    confusion = confusion.double()
    intersection = confusion.diag()
    target_count = confusion.sum(dim=1)
    prediction_count = confusion.sum(dim=0)
    union = target_count + prediction_count - intersection
    present = union > 0
    per_class = torch.full((NUM_PART_CLASSES,), float("nan"), dtype=torch.double)
    per_class[present] = intersection[present] / union[present]
    accuracy = float(intersection.sum() / target_count.sum().clamp_min(1.0))
    mean_iou = float(per_class[present].mean()) if present.any() else 0.0
    return accuracy, mean_iou, per_class.tolist()


def run_epoch(
    model: CAMOPrototype1,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float, float, list[float], float]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(NUM_PART_CLASSES, NUM_PART_CLASSES, dtype=torch.int64)
    weighted_loss = 0.0
    valid_points = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0
    for batch in tqdm(loader, desc="train" if training else "val", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        neighbors = batch["neighbors"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        valid = target != IGNORE_INDEX
        count = int(valid.sum())
        if count == 0:
            raise RuntimeError("all-ignore batch reached training loop")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype
        ):
            logits = model(features, mass, neighbors)
            loss = loss_fn(logits, target)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite gradient norm")
            gradient_norm_sum += float(gradient_norm.detach())
            gradient_steps += 1
            scaler.step(optimizer)
            scaler.update()
        prediction = logits.argmax(dim=-1)
        update_confusion(confusion, prediction.detach(), target.detach())
        weighted_loss += float(loss.detach()) * count
        valid_points += count
    accuracy, mean_iou, per_class = metrics(confusion)
    mean_gradient_norm = gradient_norm_sum / max(gradient_steps, 1)
    return (
        weighted_loss / max(valid_points, 1), accuracy, mean_iou,
        per_class, mean_gradient_norm,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--amp-init-scale", type=float, default=32768.0)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--mass-mode", choices=("none", "uniform", "density"), default="density"
    )
    args = parser.parse_args()
    if args.lovasz_weight < 0.0:
        parser.error("lovasz-weight must be non-negative")
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0.0:
        parser.error("amp-init-scale must be finite and positive")
    benchmark = validate_benchmark(args.benchmark_report, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp_dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 requested but unsupported by this CUDA device")
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    train_dataset = CAMOBuildingNetDataset(
        args.data_dir, "train", args.points, args.k, True, args.seed
    )
    validation_dataset = CAMOBuildingNetDataset(
        args.data_dir, "val", args.points, args.k, False, args.seed
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )

    model = CAMOPrototype1(
        width=args.width, mass_mode=args.mass_mode
    ).to(device)
    class_weights = torch.from_numpy(
        np.load(args.data_dir / "class_weights.npy", allow_pickle=False)
    ).float().to(device)
    if args.lovasz_weight > 0.0:
        loss_fn = BalancedCEWithLovasz(
            class_weights,
            lovasz_weight=args.lovasz_weight,
            ignore_index=IGNORE_INDEX,
        )
    else:
        loss_fn = StaticBalancedCrossEntropy(
            class_weights, ignore_index=IGNORE_INDEX
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp_dtype == "float16",
        init_scale=args.amp_init_scale,
    )

    loss_tag = (
        f"bce_lovasz{args.lovasz_weight:g}"
        if args.lovasz_weight > 0.0 else "bce"
    )
    run_name = (
        f"camo_p15_{args.mass_mode}_{loss_tag}_p{args.points}_seed{args.seed}"
    )
    checkpoint_dir = args.data_dir / "checkpoints"
    log_dir = args.data_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{run_name}.pth"
    log_path = log_dir / f"{run_name}.csv"
    best_iou = -1.0

    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "epoch", "train_loss", "train_acc", "train_miou",
                "train_gradient_norm", "val_loss", "val_acc", "val_miou",
            ]
        )
        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch)
            train_result = run_epoch(
                model, train_loader, loss_fn, device, optimizer, scaler,
                amp_dtype=autocast_dtype,
            )
            validation_result = run_epoch(
                model, validation_loader, loss_fn, device,
                amp_dtype=autocast_dtype,
            )
            scheduler.step()
            writer.writerow([
                epoch, *train_result[:3], train_result[4],
                *validation_result[:3],
            ])
            stream.flush()
            print(
                f"epoch={epoch:03d} train_mIoU={train_result[2]:.4f} "
                f"val_mIoU={validation_result[2]:.4f}"
            )
            if validation_result[2] > best_iou:
                best_iou = validation_result[2]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_miou": best_iou,
                        "val_per_class_iou": validation_result[3],
                        "config": vars(args) | {"data_dir": str(args.data_dir), "benchmark_report": str(args.benchmark_report)},
                        "benchmark": benchmark,
                    },
                    checkpoint_path,
                )
    print(f"best_val_mIoU={best_iou:.4f} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
