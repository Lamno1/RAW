"""Prospective S7 dry-run, pilot, and confirmatory training runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from camo_build.losses import BalancedCEWithLovasz
from camo_build.protocol import verify_protocol
from camo_build.s7 import CAMOPyramidS7
from camo_s5_dataset import CAMOS5Dataset
from dataset import IGNORE_INDEX
from train_camo_s5 import run_epoch


ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "protocol" / "s0"
PREREG = ROOT / "protocol" / "mutations" / "s7_global_proxy_preregistration.json"
CONFIRMATORY_AUTH = ROOT / "protocol" / "s7" / "confirmatory_authorization.json"
MODES = ("identity", "unweighted_mha", "mass_aware_mha")


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_manifest(path: Path, *, require_match: bool = False) -> None:
    sources = {
        ROOT / "train_camo_s7.py", ROOT / "train_camo_s5.py",
        ROOT / "camo_s5_dataset.py", ROOT / "camo_dataset.py",
        ROOT / "dataset.py", *set((ROOT / "camo_build").glob("*.py")),
    }
    contents = "".join(
        f"{sha256(source)}  {source.relative_to(ROOT).as_posix()}\n"
        for source in sorted(sources)
    )
    if require_match and path.is_file() and path.read_text(encoding="utf-8") != contents:
        raise RuntimeError("source changed since checkpoint; refusing an inexact resume")
    path.write_text(contents, encoding="utf-8")


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_phase(args: argparse.Namespace) -> dict[str, object]:
    verification = verify_protocol(ROOT, PROTOCOL_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"S0 verification failed: {verification}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg.get("state") != "FROZEN_BEFORE_ANY_S7_METRIC":
        raise RuntimeError("S7 preregistration is not frozen")
    dependencies = prereg["locked_dependencies"]
    dependency_keys = (
        ("s2_objective_decision", "s2_objective_sha256"),
        ("s5_headline_result", "s5_headline_sha256"),
        ("s5_accuracy_guardrail", "s5_accuracy_guardrail_sha256"),
    )
    for path_key, hash_key in dependency_keys:
        path = ROOT / dependencies[path_key]
        expected = dependencies[hash_key]
        if sha256(path) != expected:
            raise RuntimeError(f"locked S7 dependency changed: {path}")
    expected = prereg["configuration"]
    actual = {
        "points": args.points, "proxies": args.proxies,
        "fine_knn": args.fine_k, "proxy_knn": args.proxy_k,
        "support_k": args.support_k, "width": args.width,
        "attention_heads": args.attention_heads,
        "batch_size": args.batch_size, "amp_dtype": args.amp_dtype,
    }
    mismatch = {key: (expected[key], value) for key, value in actual.items()
                if expected[key] != value}
    if mismatch:
        raise RuntimeError(f"S7 configuration mismatch: {mismatch}")
    if args.phase == "pilot":
        if args.seed != expected["pilot_seed"] or args.epochs != expected["pilot_epochs"]:
            raise RuntimeError("pilot seed/epochs differ from preregistration")
    elif args.phase == "confirmatory":
        if args.seed not in expected["confirmatory_seeds"] or args.epochs != expected["confirmatory_epochs"]:
            raise RuntimeError("confirmatory seed/epochs differ from preregistration")
        if not CONFIRMATORY_AUTH.is_file():
            raise RuntimeError("S7 confirmatory training has not been authorized")
        authorization = json.loads(CONFIRMATORY_AUTH.read_text(encoding="utf-8"))
        if authorization.get("state") != "AUTHORIZED_BEFORE_CONFIRMATORY_METRICS":
            raise RuntimeError("S7 authorization is not in the authorized state")
        decision_path = ROOT / authorization["decision_artifact"]
        if sha256(decision_path) != authorization.get("decision_artifact_sha256"):
            raise RuntimeError("S7 pilot decision artifact hash mismatch")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            decision.get("state") != "FROZEN_BEFORE_CONFIRMATORY_METRICS"
            or decision.get("decision", {}).get("selected_candidate")
            != authorization.get("selected_arm")
            or decision.get("preregistration", {}).get("sha256") != sha256(PREREG)
        ):
            raise RuntimeError("S7 confirmatory authorization disagrees with pilot decision")
        if args.global_mode not in authorization.get("arms_to_run", []):
            raise RuntimeError("requested arm is not authorized for S7 confirmatory")
    return {"verification": verification, "preregistration": prereg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("dry_run", "pilot", "confirmatory"), required=True)
    parser.add_argument("--global-mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--proxies", type=int, default=256)
    parser.add_argument("--fine-k", type=int, default=16)
    parser.add_argument("--proxy-k", type=int, default=16)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    authorization = validate_phase(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S7 requires BF16 CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    protocol = json.loads((PROTOCOL_DIR / "camo_pyramid_protocol.json").read_text())
    class_groups = protocol["class_groups"]
    train_dataset = CAMOS5Dataset(
        args.data_dir, "train", args.points, args.fine_k, True, args.seed,
        proxies=args.proxies, support_k=args.support_k, proxy_k=args.proxy_k,
    )
    validation_dataset = CAMOS5Dataset(
        args.data_dir, "val", args.points, args.fine_k, False, args.seed,
        proxies=args.proxies, support_k=args.support_k, proxy_k=args.proxy_k,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = CAMOPyramidS7(
        width=args.width, global_mode=args.global_mode,
        attention_heads=args.attention_heads,
    ).to(device)
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
    config = {**vars(args), "data_dir": str(args.data_dir.resolve()),
              "run_dir": str(run_dir), "objective": "BCE_PLUS_0.5_LOVASZ",
              "coarsening": "row_softmax", "mass_mode": "density",
              "protocol_version": protocol["version"]}
    write_json(run_dir / "config.yaml", config)
    for name in ("environment.txt", "split_manifest.json", "crop_manifest.json", "checkpoint_rule.json"):
        shutil.copy2(PROTOCOL_DIR / name, run_dir / name)
    source_manifest(run_dir / "source_manifest.sha256", require_match=args.resume)
    write_json(run_dir / "authorization.json", {
        "s7_preregistration_sha256": sha256(PREREG),
        "confirmatory_authorization_sha256": sha256(CONFIRMATORY_AUTH) if CONFIRMATORY_AUTH.is_file() else None,
        "protocol_verification": authorization["verification"], "verified_utc": utc_now(),
    })
    write_json(run_dir / "status.json", {
        "stage": "S7", "phase": args.phase, "arm": args.global_mode,
        "state": f"{args.phase.upper()}_RUNNING", "started_utc": utc_now(),
    })

    metrics_path, best_path, last_path = (
        run_dir / "metrics.csv", run_dir / "best.pth", run_dir / "last.pth"
    )
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
    try:
        append = args.resume and start_epoch > 1 and metrics_path.is_file()
        with metrics_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not append:
                writer.writerow([
                    "epoch", "train_loss", "train_part_iou", "train_shape_iou",
                    "train_accuracy", "train_head_iou", "train_mid_iou", "train_tail_iou",
                    "train_gradient_norm", "val_loss", "val_part_iou", "val_shape_iou",
                    "val_accuracy", "val_head_iou", "val_mid_iou", "val_tail_iou", "lr",
                ])
            for epoch in range(start_epoch, args.epochs + 1):
                train_dataset.set_epoch(epoch)
                validation_dataset.set_epoch(0)
                maximum = 1 if args.phase == "dry_run" else None
                train_loss, train_metrics, gradient_norm = run_epoch(
                    model, train_loader, loss_fn, device, class_groups,
                    optimizer=optimizer, max_batches=maximum,
                )
                val_loss, val_metrics, _ = run_epoch(
                    model, validation_loader, loss_fn, device, class_groups,
                    max_batches=maximum,
                )
                scheduler.step()
                tg, vg = train_metrics["group_iou"], val_metrics["group_iou"]
                writer.writerow([
                    epoch, train_loss, train_metrics["part_iou"], train_metrics["shape_iou"],
                    train_metrics["classification_accuracy"], tg["head"], tg["mid"], tg["tail"],
                    gradient_norm, val_loss, val_metrics["part_iou"], val_metrics["shape_iou"],
                    val_metrics["classification_accuracy"], vg["head"], vg["mid"], vg["tail"],
                    optimizer.param_groups[0]["lr"],
                ])
                stream.flush()
                if float(val_metrics["part_iou"]) > best_iou:
                    best_iou, best_epoch = float(val_metrics["part_iou"]), epoch
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_iou": best_iou, "best_epoch": best_epoch,
                    "validation": val_metrics, "config": config,
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                        "loader_generator": generator.get_state(),
                    },
                }
                atomic_torch_save(state, last_path)
                if best_epoch == epoch:
                    atomic_torch_save(state, best_path)
                print(f"phase={args.phase} arm={args.global_mode} seed={args.seed} "
                      f"epoch={epoch:03d} val_piou={best_iou:.6f} "
                      f"val_acc={float(val_metrics['classification_accuracy']):.6f}", flush=True)
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "amp_dtype": "bfloat16",
        }
        if hardware["max_memory_allocated_mib"] > 3200.0:
            raise RuntimeError("S7 VRAM Limit Exceeded")
        write_json(run_dir / "hardware.json", hardware)
        write_json(run_dir / "status.json", {
            "stage": "S7", "phase": args.phase, "arm": args.global_mode,
            "state": f"{args.phase.upper()}_COMPLETE", "finished_utc": utc_now(),
            "best_epoch": best_epoch, "best_validation_part_iou": best_iou,
            "test_split_used": False,
        })
    except Exception as exc:
        write_json(run_dir / "status.json", {
            "stage": "S7", "phase": args.phase, "arm": args.global_mode,
            "state": "FAILED", "failed_utc": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
