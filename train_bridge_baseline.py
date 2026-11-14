"""S1 corrected GeometryAwarePCT bridge baseline.

This runner uses full-building normalization, epoch-varying train sampling,
fixed validation crops and optional chunked full-cloud voting. Dense PCT input
is hard-capped at 4,000 points by the project protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from camo_build.losses import BalancedCEWithLovasz, StaticBalancedCrossEntropy
from camo_build.metrics import LogitVoteAccumulator, SegmentationMeter, full_cloud_chunks
from camo_dataset import CAMOBuildingNetDataset
from dataset import IGNORE_INDEX, NUM_PART_CLASSES
from model import GeometryAwarePCT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _run_epoch(
    model: GeometryAwarePCT,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    weighted_loss = 0.0
    valid_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        valid = target != IGNORE_INDEX
        count = int(valid.sum())
        if count == 0:
            raise RuntimeError("all-ignore batch reached bridge runner")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(features)
            loss = loss_fn(logits, target)
        if training:
            if scaler is None:
                raise RuntimeError("training requires a GradScaler")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        prediction = logits.detach().argmax(-1)
        meter.update(prediction, target.detach(), model_names=batch["model_name"])
        weighted_loss += float(loss.detach()) * count
        valid_count += count
    return weighted_loss / max(valid_count, 1), meter.compute()


@torch.no_grad()
def evaluate_full_cloud(
    model: GeometryAwarePCT,
    dataset: CAMOBuildingNetDataset,
    device: torch.device,
    *,
    chunk_size: int,
    vote_seeds: list[int],
    inference_batch_size: int,
) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(NUM_PART_CLASSES, IGNORE_INDEX)
    for sample_index in range(len(dataset)):
        full = dataset.load_full(sample_index)
        features = full["features"]
        target = full["label"]
        accumulator = LogitVoteAccumulator(len(features), NUM_PART_CLASSES)
        for seed in vote_seeds:
            chunks = full_cloud_chunks(len(features), chunk_size, seed + sample_index)
            for start in range(0, len(chunks), inference_batch_size):
                group = chunks[start : start + inference_batch_size]
                batch_features = torch.stack(
                    [features[index] for index, _ in group]
                ).to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(batch_features)
                for row, (index, valid) in enumerate(group):
                    valid_index = torch.from_numpy(index[:valid]).long()
                    accumulator.update(valid_index, logits[row, :valid])
        prediction = accumulator.prediction()
        meter.update(
            prediction.unsqueeze(0), target.unsqueeze(0),
            model_names=[str(full["model_name"])],
        )
    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balanced-ce", action="store_true")
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--skip-full-cloud", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not 0 < args.points <= 4000:
        parser.error("dense GeometryAwarePCT requires 0 < points <= 4000")
    if args.lovasz_weight < 0:
        parser.error("lovasz-weight must be non-negative")
    protocol_dir = args.protocol_dir or args.data_dir / "protocol" / "s0"
    protocol_path = protocol_dir / "camo_pyramid_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("S0 is not frozen; run freeze_camo_protocol.py first")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    vote_seeds = [int(seed) for seed in protocol["crop_seeds"]]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CAMOBuildingNetDataset(
        args.data_dir, "train", args.points, augment=True,
        seed=args.seed, build_neighbors=False,
    )
    validation_dataset = CAMOBuildingNetDataset(
        args.data_dir, "val", args.points, augment=False,
        seed=args.seed, build_neighbors=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_args
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)

    model = GeometryAwarePCT(width=args.width).to(device)
    class_weights = torch.from_numpy(
        np.load(args.data_dir / "class_weights.npy", allow_pickle=False)
    ).float().to(device)
    base_weights = class_weights if args.balanced_ce else torch.ones_like(class_weights)
    if args.lovasz_weight > 0:
        loss_fn: torch.nn.Module = BalancedCEWithLovasz(
            base_weights, lovasz_weight=args.lovasz_weight, ignore_index=IGNORE_INDEX
        )
    else:
        loss_fn = StaticBalancedCrossEntropy(base_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    run_name = f"bridge_p{args.points}_seed{args.seed}"
    run_dir = args.run_dir or args.data_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "data_dir": str(args.data_dir), "protocol_dir": str(protocol_dir),
        "run_dir": str(run_dir), "device": str(device),
    }
    # JSON is a valid YAML 1.2 document and avoids an additional dependency.
    _write_json(run_dir / "config.yaml", config)
    for name in ("source_manifest.sha256", "environment.txt", "split_manifest.json", "crop_manifest.json", "checkpoint_rule.json"):
        shutil.copy2(protocol_dir / name, run_dir / name)
    _write_json(run_dir / "status.json", {"stage": "S1", "state": "DRY_RUN" if args.dry_run else "PILOT"})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_iou = -1.0
    checkpoint_path = run_dir / "best.pth"
    metrics_path = run_dir / "metrics.csv"
    epochs = 1 if args.dry_run else args.epochs
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "train_piou", "val_loss", "val_piou", "val_siou", "val_ca"])
        for epoch in range(1, epochs + 1):
            train_dataset.set_epoch(epoch)
            validation_dataset.set_epoch(0)
            maximum = 1 if args.dry_run else None
            train_loss, train_metrics = _run_epoch(
                model, train_loader, loss_fn, device,
                optimizer=optimizer, scaler=scaler, max_batches=maximum,
            )
            val_loss, val_metrics = _run_epoch(
                model, validation_loader, loss_fn, device, max_batches=maximum
            )
            scheduler.step()
            writer.writerow([
                epoch, train_loss, train_metrics["part_iou"], val_loss,
                val_metrics["part_iou"], val_metrics["shape_iou"],
                val_metrics["classification_accuracy"],
            ])
            stream.flush()
            if float(val_metrics["part_iou"]) > best_iou:
                best_iou = float(val_metrics["part_iou"])
                torch.save(
                    {
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "validation": val_metrics, "config": config,
                        "rng": {
                            "python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    },
                    checkpoint_path,
                )

    full_metrics: dict[str, object] | None = None
    if not args.skip_full_cloud and not args.dry_run:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        full_metrics = evaluate_full_cloud(
            model, validation_dataset, device, chunk_size=args.points,
            vote_seeds=vote_seeds, inference_batch_size=args.inference_batch_size,
        )
        _write_json(run_dir / "full_cloud_validation.json", full_metrics)

    hardware = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0.0,
    }
    _write_json(run_dir / "hardware.json", hardware)
    final_state = (
        "DRY_RUN" if args.dry_run
        else "CONFIRMATORY_COMPLETE" if full_metrics is not None
        else "TRAINING_COMPLETE_EVAL_PENDING"
    )
    _write_json(run_dir / "status.json", {
        "stage": "S1", "state": final_state,
        "best_fixed_crop_part_iou": best_iou,
        "full_cloud_complete": full_metrics is not None,
    })
    print(json.dumps({"run_dir": str(run_dir), "best_part_iou": best_iou, "hardware": hardware}, indent=2))


if __name__ == "__main__":
    main()
