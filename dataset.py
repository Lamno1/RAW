"""BuildingNet dataset for point-wise semantic segmentation."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

PART_LABELS = (
    "wall", "window", "vehicle", "roof", "plant", "door", "tower",
    "furniture", "ground", "beam", "stairs", "column", "banister", "floor",
    "chimney", "ceiling", "fence", "pool", "corridor", "balcony", "garage",
    "dome", "road", "gate", "parapet", "buttress", "dormer", "lighting",
    "arch", "awning", "shutters",
)
NUM_PART_CLASSES = len(PART_LABELS)
IGNORE_INDEX = -100

def load_ply(path):
    path = Path(path); count = None; skip = 0
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            skip += 1
            if line.startswith("element vertex"): count = int(line.split()[-1])
            if line.strip() == "end_header": break
    if count is None: raise ValueError(f"PLY không có vertex: {path}")
    data = np.loadtxt(path, dtype=np.float32, skiprows=skip, max_rows=count)
    if data.shape != (count, 9): raise ValueError(f"PLY phải có shape ({count},9), nhận {data.shape}")
    return data

class BuildingNetDataset(Dataset):
    def __init__(self, root_dir, split="train", num_points=1024, augment=None, seed=42):
        self.root = Path(root_dir); self.split = split; self.num_points = num_points
        self.augment = split == "train" if augment is None else augment; self.seed = seed
        split_file = self.root / "splits" / f"{split}_split.txt"
        self.archive_path = self.root / "model_data" / "point_cloud" / "point_labels.zip"
        self.cache_dir = self.root / "cache_npz" / split
        for item in (split_file, self.archive_path, self.root / "POINT_CLOUDS"):
            if not item.exists(): raise FileNotFoundError(f"Thiếu dữ liệu: {item}")
        names = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        available = {p.stem: p for p in (self.root / "POINT_CLOUDS").glob("*.ply")}
        missing = sorted(set(names) - set(available))
        if missing: raise FileNotFoundError(f"Thiếu {len(missing)} PLY; ví dụ {missing[:3]}")
        self.samples = [(name, available[name]) for name in names]

    def __len__(self): return len(self.samples)

    def _labels(self, name):
        member = f"point_labels/{name}_label.json"
        with zipfile.ZipFile(self.archive_path) as archive, archive.open(member) as stream:
            mapping = json.load(stream)
        labels = np.empty(len(mapping), dtype=np.int64)
        for index, value in mapping.items(): labels[int(index)] = int(value)
        return labels

    def __getitem__(self, index):
        name, path = self.samples[index]
        cache_path = self.cache_dir / f"{name}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                data, raw_labels = cached["features"], cached["labels"].astype(np.int64)
        else:
            data, raw_labels = load_ply(path), self._labels(name)
        labels = np.where(raw_labels == 0, IGNORE_INDEX, raw_labels - 1)
        if len(data) != len(labels): raise ValueError(f"Lệch point/label ở {name}")
        rng = np.random.default_rng(self.seed + index + 999983) if self.augment else np.random.default_rng(self.seed + index)
        choice = rng.choice(len(data), self.num_points, replace=len(data) < self.num_points)
        data, labels = data[choice].copy(), labels[choice].copy()
        valid = labels != IGNORE_INDEX
        if valid.any() and (labels[valid].min() < 0 or labels[valid].max() >= NUM_PART_CLASSES):
            raise ValueError(f"Nhãn sai ở {name}")
        xyz, normal, rgb = data[:, :3], data[:, 3:6], data[:, 6:9]
        xyz -= xyz.mean(0, keepdims=True); xyz /= max(np.linalg.norm(xyz, axis=1).max(), 1e-6)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        if rgb.max() > 1: rgb /= 255.0
        if self.augment:
            a = rng.uniform(0, 2*np.pi); c, s = np.cos(a), np.sin(a)
            rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            xyz[:] = xyz @ rotation.T; normal[:] = normal @ rotation.T
            xyz += rng.normal(0, .005, xyz.shape).astype(np.float32)
        features = np.concatenate((xyz, normal, rgb), axis=1)
        return {"features": torch.from_numpy(features).float(), "label": torch.from_numpy(labels).long(), "model_name": name}

if __name__ == "__main__":
    sample = BuildingNetDataset(Path(__file__).parent, augment=False)[0]
    print(sample["model_name"], sample["features"].shape, sample["label"].shape, torch.unique(sample["label"]))
