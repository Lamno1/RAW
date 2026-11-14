"""Dataset adapter for the frozen CAMO-Build Prototype 1 protocol."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from camo_build.neighbors import scipy_knn_indices
from dataset import IGNORE_INDEX, NUM_PART_CLASSES, load_ply


class CAMOBuildingNetDataset(Dataset):
    """BuildingNet samples with cached full-cloud mass and normalization."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        num_points: int = 4096,
        k: int = 16,
        augment: bool | None = None,
        seed: int = 42,
        build_neighbors: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.num_points = int(num_points)
        self.k = int(k)
        self.augment = split == "train" if augment is None else augment
        self.seed = int(seed)
        self.build_neighbors = bool(build_neighbors)
        self.epoch = 0
        split_file = self.root / "splits" / f"{split}_split.txt"
        self.archive_path = (
            self.root / "model_data" / "point_cloud" / "point_labels.zip"
        )
        names = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        available = {
            path.stem: path for path in (self.root / "POINT_CLOUDS").glob("*.ply")
        }
        self.samples = [(name, available[name]) for name in names]
        missing_metadata = [
            name
            for name, _ in self.samples
            if not (self.root / "camo_cache" / split / f"{name}.npz").exists()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                "CAMO metadata missing; run prepare_camo_cache.py. "
                f"Examples: {missing_metadata[:3]}"
            )

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic training sampling between epochs."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _labels(self, name: str) -> np.ndarray:
        member = f"point_labels/{name}_label.json"
        with zipfile.ZipFile(self.archive_path) as archive, archive.open(member) as stream:
            mapping = json.load(stream)
        labels = np.empty(len(mapping), dtype=np.int64)
        for index, value in mapping.items():
            labels[int(index)] = int(value)
        return labels

    def _load_full_arrays(
        self, index: int
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Load and normalize one complete building before sampling."""
        name, ply_path = self.samples[index]
        base_cache = self.root / "cache_npz" / self.split / f"{name}.npz"
        if base_cache.exists():
            with np.load(base_cache, allow_pickle=False) as archive:
                data = archive["features"].astype(np.float32, copy=True)
                raw_labels = archive["labels"].astype(np.int64, copy=False)
        else:
            data = load_ply(ply_path)
            raw_labels = self._labels(name)

        metadata_path = self.root / "camo_cache" / self.split / f"{name}.npz"
        with np.load(metadata_path, allow_pickle=False) as metadata:
            center = metadata["center"].astype(np.float32, copy=False)
            scale = float(metadata["scale"])
            full_mass = metadata["mass"].astype(np.float32, copy=True)
            metadata_points = int(metadata["num_points"])
        if len(data) != len(raw_labels) or len(data) != len(full_mass):
            raise ValueError(f"point-aligned arrays disagree for {name}")
        if metadata_points != len(data):
            raise ValueError(f"stale CAMO metadata for {name}")

        labels = np.where(raw_labels == 0, IGNORE_INDEX, raw_labels - 1)
        xyz = data[:, :3]
        normals = data[:, 3:6]
        rgb = data[:, 6:9]
        xyz -= center
        xyz /= max(scale, 1e-6)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
        if rgb.max(initial=0.0) > 1.0:
            rgb /= 255.0
        if not np.isfinite(data).all() or not np.isfinite(full_mass).all():
            raise FloatingPointError(f"non-finite full cloud in {name}")
        return name, data, labels.astype(np.int64, copy=False), full_mass

    def load_full(self, index: int) -> dict[str, torch.Tensor | str]:
        """Return a normalized full cloud for chunked validation voting."""
        name, data, labels, mass = self._load_full_arrays(index)
        return {
            "features": torch.from_numpy(data).float(),
            "mass": torch.from_numpy(mass).float(),
            "label": torch.from_numpy(labels).long(),
            "point_index": torch.arange(len(data), dtype=torch.long),
            "model_name": name,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name, data, labels, full_mass = self._load_full_arrays(index)
        rng_seed = self.seed + index + self.epoch * 1_000_003
        rng = np.random.default_rng(rng_seed)
        selection = rng.choice(
            len(data), self.num_points, replace=len(data) < self.num_points
        )
        sample = data[selection].copy()
        labels = labels[selection].copy()
        mass = full_mass[selection].copy()

        xyz = sample[:, :3]
        normals = sample[:, 3:6]
        rgb = sample[:, 6:9]
        # `_load_full_arrays` already applied the full-building transform
        # before sampling, preserving the frozen normalization parameters.

        if self.augment:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            xyz[:] = xyz @ rotation.T
            normals[:] = normals @ rotation.T
            xyz += rng.normal(0.0, 0.005, xyz.shape).astype(np.float32)

        mass /= max(float(mass.mean()), 1e-6)
        neighbors = scipy_knn_indices(xyz, self.k) if self.build_neighbors else None
        valid = labels != IGNORE_INDEX
        if valid.any() and (
            labels[valid].min() < 0 or labels[valid].max() >= NUM_PART_CLASSES
        ):
            raise ValueError(f"invalid label in {name}")
        features = np.concatenate((xyz, normals, rgb), axis=1)
        if not np.isfinite(features).all() or not np.isfinite(mass).all():
            raise FloatingPointError(f"non-finite CAMO sample in {name}")
        result: dict[str, torch.Tensor | str] = {
            "features": torch.from_numpy(features).float(),
            "mass": torch.from_numpy(mass).float(),
            "label": torch.from_numpy(labels).long(),
            "point_index": torch.from_numpy(selection.astype(np.int64, copy=False)).long(),
            "model_name": name,
        }
        if neighbors is not None:
            result["neighbors"] = torch.from_numpy(neighbors).long()
        return result
