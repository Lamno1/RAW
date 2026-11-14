"""Prepare immutable full-building normalization and mass metadata.

This script never overwrites the existing baseline cache. It writes compact,
point-aligned metadata under ``camo_cache/<split>``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from camo_build.measure import estimate_surface_mass
from dataset import load_ply


def normalization_parameters(xyz: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, float]:
    """Return full-building centroid and maximum radial scale."""
    center = xyz.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = xyz - center
    scale = max(float(np.linalg.norm(centered, axis=1).max()), epsilon)
    return center, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        split_file = args.data_dir / "splits" / f"{split}_split.txt"
        names = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit is not None:
            names = names[: args.limit]
        output_dir = args.data_dir / "camo_cache" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in tqdm(names, desc=f"CAMO metadata {split}"):
            destination = output_dir / f"{name}.npz"
            if destination.exists() and not args.overwrite:
                continue
            cached = args.data_dir / "cache_npz" / split / f"{name}.npz"
            if cached.exists():
                with np.load(cached, allow_pickle=False) as archive:
                    features = archive["features"].astype(np.float32, copy=False)
            else:
                features = load_ply(
                    args.data_dir / "POINT_CLOUDS" / f"{name}.ply"
                )
            xyz = features[:, :3]
            center, scale = normalization_parameters(xyz)
            normalized_xyz = (xyz - center) / scale
            mass = estimate_surface_mass(normalized_xyz, k=args.k)
            np.savez(
                destination,
                center=center,
                scale=np.asarray(scale, dtype=np.float32),
                mass=mass,
                k=np.asarray(args.k, dtype=np.int32),
                num_points=np.asarray(len(xyz), dtype=np.int64),
            )


if __name__ == "__main__":
    main()

