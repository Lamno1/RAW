"""Full-cloud to working-measure assignment for CAMO-Pyramid S4."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .transport import SparseTransportPlan


@dataclass(frozen=True)
class NormalOrientationAudit:
    finite_fraction: float
    unit_length_mean_error: float
    negative_neighbor_fraction: float
    signed_safe: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def audit_normal_orientation(
    xyz: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 8,
    max_negative_fraction: float = 0.1,
) -> NormalOrientationAudit:
    """Audit local sign consistency; it does not silently reorient normals."""
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if xyz.shape != normals.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz and normals must have shape [N,3]")
    finite = np.isfinite(normals).all(axis=1)
    lengths = np.linalg.norm(normals[finite], axis=1)
    finite_fraction = float(finite.mean())
    mean_error = float(np.mean(np.abs(lengths - 1.0))) if lengths.size else float("inf")
    if len(xyz) <= k:
        raise ValueError("normal audit requires N > k")
    _, neighbor = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    neighbor = neighbor[:, 1:]
    safe_normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    dot = np.einsum("nd,nkd->nk", safe_normals, safe_normals[neighbor])
    negative_fraction = float(np.mean(dot < 0.0))
    signed_safe = (
        finite_fraction == 1.0
        and mean_error <= 1e-2
        and negative_fraction <= max_negative_fraction
    )
    return NormalOrientationAudit(
        finite_fraction, mean_error, negative_fraction, signed_safe
    )


def select_spatial_anchors(
    xyz: np.ndarray,
    mass: np.ndarray,
    num_anchors: int,
    *,
    max_candidate_factor: int = 4,
) -> np.ndarray:
    """Select deterministic voxel candidates followed by weighted FPS.

    This is an offline/CPU coverage initializer. Its finite grid is not claimed
    to be exactly SO(2)-equivariant; azimuth robustness remains an empirical
    test as specified by the architecture document.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    mass = np.asarray(mass, dtype=np.float32)
    n = len(xyz)
    if xyz.shape != (n, 3) or mass.shape != (n,):
        raise ValueError("xyz [N,3] and mass [N] are required")
    if not np.isfinite(xyz).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("anchor inputs must be finite with positive masses")
    if not 0 < num_anchors <= n:
        raise ValueError("num_anchors must satisfy 0 < M <= N")
    if num_anchors == n:
        return np.arange(n, dtype=np.int64)

    lower = xyz.min(0)
    span = np.maximum(xyz.max(0) - lower, 1e-6)
    normalized = (xyz - lower) / span
    resolution = max(2, int(np.ceil(num_anchors ** (1.0 / 3.0))))
    representatives = np.empty(0, dtype=np.int64)
    codes = np.empty(0, dtype=np.int64)
    while resolution <= 4096:
        cell = np.minimum((normalized * resolution).astype(np.int64), resolution - 1)
        code = cell[:, 0] + resolution * (cell[:, 1] + resolution * cell[:, 2])
        order = np.argsort(code, kind="stable")
        sorted_code = code[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_code)) + 1]
        ends = np.r_[starts[1:], n]
        reps = []
        rep_codes = []
        for start, end in zip(starts, ends, strict=True):
            members = order[start:end]
            reps.append(int(members[np.argmax(mass[members])]))
            rep_codes.append(int(sorted_code[start]))
        representatives = np.asarray(reps, dtype=np.int64)
        codes = np.asarray(rep_codes, dtype=np.int64)
        if representatives.size >= num_anchors:
            break
        resolution *= 2

    if representatives.size < num_anchors:
        selected = np.zeros(n, dtype=bool)
        selected[representatives] = True
        fill = np.flatnonzero(~selected)
        fill = fill[np.argsort(-mass[fill], kind="stable")]
        representatives = np.concatenate(
            (representatives, fill[: num_anchors - representatives.size])
        )
        codes = np.arange(representatives.size, dtype=np.int64)

    maximum_candidates = max(num_anchors, max_candidate_factor * num_anchors)
    if representatives.size > maximum_candidates:
        order = np.argsort(codes, kind="stable")
        positions = np.linspace(0, order.size - 1, maximum_candidates, dtype=np.int64)
        representatives = representatives[order[positions]]

    candidate_xyz = xyz[representatives]
    candidate_mass = mass[representatives]
    selected_local = np.empty(num_anchors, dtype=np.int64)
    selected_mask = np.zeros(representatives.size, dtype=bool)
    first = int(np.argmax(candidate_mass))
    minimum_distance = np.full(representatives.size, np.inf, dtype=np.float32)
    for step in range(num_anchors):
        if step == 0:
            current = first
        else:
            weight = np.sqrt(
                candidate_mass / max(float(candidate_mass.mean()), 1e-6)
            )
            score = np.where(selected_mask, -1.0, minimum_distance * weight)
            current = int(np.argmax(score))
        selected_local[step] = current
        selected_mask[current] = True
        squared = np.sum((candidate_xyz - candidate_xyz[current]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, squared)
        minimum_distance[selected_mask] = -1.0
    return representatives[selected_local]


def build_full_cloud_assignment(
    xyz: np.ndarray,
    mass: np.ndarray,
    *,
    anchor_xyz: np.ndarray | None = None,
    num_anchors: int | None = None,
    k: int = 2,
    epsilon: float = 1e-6,
) -> SparseTransportPlan:
    """Construct ``Q_is=m_i A_is`` with exact row-stochastic ``A``."""
    xyz = np.asarray(xyz, dtype=np.float32)
    mass = np.asarray(mass, dtype=np.float32)
    n = len(xyz)
    if xyz.shape != (n, 3) or mass.shape != (n,):
        raise ValueError("xyz [N,3] and mass [N] are required")
    if not np.isfinite(xyz).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("assignment inputs must be finite with positive masses")
    if anchor_xyz is None:
        if num_anchors is None:
            raise ValueError("anchor_xyz or num_anchors is required")
        anchor_index = select_spatial_anchors(xyz, mass, num_anchors)
        anchors = xyz[anchor_index]
    else:
        anchors = np.asarray(anchor_xyz, dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or not np.isfinite(anchors).all():
            raise ValueError("anchor_xyz must be finite [M,3]")
    m = len(anchors)
    if not 0 < k <= m:
        raise ValueError("k must satisfy 0 < k <= number of anchors")

    distance, target = cKDTree(anchors).query(xyz, k=k, workers=-1)
    if k == 1:
        distance = distance[:, None]
        target = target[:, None]
    source = np.repeat(np.arange(n, dtype=np.int64), k)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    squared = np.asarray(distance, dtype=np.float32).reshape(-1) ** 2

    present = np.bincount(target, minlength=m) > 0
    if not present.all():
        missing = np.flatnonzero(~present)
        reciprocal_distance, reciprocal_source = cKDTree(xyz).query(
            anchors[missing], k=1, workers=-1
        )
        source = np.concatenate((source, np.asarray(reciprocal_source, dtype=np.int64)))
        target = np.concatenate((target, missing.astype(np.int64)))
        squared = np.concatenate(
            (squared, np.asarray(reciprocal_distance, dtype=np.float32) ** 2)
        )

    encoded = source * m + target
    order = np.argsort(encoded, kind="stable")
    encoded, source, target, squared = (
        encoded[order], source[order], target[order], squared[order]
    )
    unique, first = np.unique(encoded, return_index=True)
    source = source[first]
    target = target[first]
    squared = np.minimum.reduceat(squared, first)
    if unique.size != source.size:
        raise RuntimeError("assignment edge coalescing failed")

    row_scale = np.zeros(n, dtype=np.float32)
    np.maximum.at(row_scale, source, squared)
    row_scale = np.maximum(row_scale, epsilon)
    score = np.clip(-squared / row_scale[source], -50.0, 0.0)
    row_max = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(row_max, source, score)
    unnormalized = np.exp(score - row_max[source]).astype(np.float32)
    row_sum = np.zeros(n, dtype=np.float32)
    np.add.at(row_sum, source, unnormalized)
    conditional = unnormalized / np.maximum(row_sum[source], epsilon)
    # Correct FP32 summation drift without perturbing the mathematical rows.
    conditional_sum = np.zeros(n, dtype=np.float32)
    np.add.at(conditional_sum, source, conditional)
    conditional /= conditional_sum[source]
    source_coverage = np.bincount(source, minlength=n)
    target_coverage = np.bincount(target, minlength=m)
    if np.any(source_coverage == 0):
        missing = np.flatnonzero(source_coverage == 0)[:5].tolist()
        raise RuntimeError(f"assignment lost source rows: {missing}")
    if np.any(target_coverage == 0):
        missing = np.flatnonzero(target_coverage == 0)[:5].tolist()
        raise RuntimeError(f"assignment lost anchor coverage: {missing}")
    if not np.isfinite(conditional).all() or np.any(conditional <= 0.0):
        raise FloatingPointError("assignment produced invalid conditional weights")
    value = mass[source] * conditional

    source_tensor = torch.from_numpy(source).long()
    target_tensor = torch.from_numpy(target).long()
    value_tensor = torch.from_numpy(value).float()
    source_mass = torch.from_numpy(mass).float()
    target_mass = torch.zeros(m, dtype=torch.float32)
    target_mass.scatter_add_(0, target_tensor, value_tensor)
    return SparseTransportPlan(
        source_tensor, target_tensor, value_tensor, source_mass, target_mass
    )
