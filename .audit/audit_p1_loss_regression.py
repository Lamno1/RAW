"""Locate the first non-finite P1 gradient under two equivalent CE forms."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camo_build.losses import StaticBalancedCrossEntropy
from camo_build.model import CAMOPrototype1
from camo_dataset import CAMOBuildingNetDataset
from dataset import IGNORE_INDEX


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(loss_kind: str, *, strict: bool) -> dict[str, object]:
    seed_all(42)
    device = torch.device("cuda")
    dataset = CAMOBuildingNetDataset(ROOT, "train", 4096, 16, True, 42)
    dataset.set_epoch(1)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0, pin_memory=True,
    )
    model = CAMOPrototype1(width=64, mass_mode="density").to(device)
    weights = torch.from_numpy(np.load(ROOT / "class_weights.npy")).float().to(device)
    current = StaticBalancedCrossEntropy(weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    skipped = 0
    max_loss_delta = 0.0
    first_nonfinite: dict[str, object] | None = None

    for batch_index, batch in enumerate(loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        neighbors = batch["neighbors"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(features, mass, neighbors)
        valid = target != IGNORE_INDEX
        current_loss = current(logits, target)
        reference_loss = F.cross_entropy(
            logits[valid].float(), target[valid], weight=weights, reduction="mean"
        )
        max_loss_delta = max(
            max_loss_delta, float((current_loss - reference_loss).abs().detach())
        )
        loss = current_loss if loss_kind == "current" else reference_loss
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad_parameters = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if bad_parameters and first_nonfinite is None:
            first_nonfinite = {
                "batch": batch_index,
                "loss": float(loss.detach()),
                "current_loss": float(current_loss.detach()),
                "reference_loss": float(reference_loss.detach()),
                "logit_abs_max": float(logits.detach().abs().max()),
                "scale": scale_before,
                "bad_parameter_count": len(bad_parameters),
                "bad_parameters_first10": bad_parameters[:10],
            }
            if strict:
                break
        if not bad_parameters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1

    return {
        "loss_kind": loss_kind,
        "strict": strict,
        "max_loss_delta": max_loss_delta,
        "first_nonfinite": first_nonfinite,
        "skipped_steps": skipped,
        "final_scale": scaler.get_scale(),
    }


def main() -> None:
    result = {
        "current_strict": run("current", strict=True),
        "reference_strict": run("reference", strict=True),
        "legacy_gradscaler": run("current", strict=False),
    }
    output = ROOT / ".audit" / "p1_loss_regression.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
