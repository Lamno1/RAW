"""Frozen Prototype 1 surface-mass estimator."""
from __future__ import annotations

import math

import numpy as np

from .neighbors import scipy_knn_indices


def estimate_surface_mass(
    xyz: np.ndarray,
    k: int = 16,
    epsilon: float = 1e-6,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    r"""Estimate mean-one quadrature masses on an intrinsic 2-D surface.

    The frozen estimator is

        rho_i = mean_j exp(-d_ij^2 / (2(h_i^2 + eps)))
                / (2*pi*(h_i^2 + eps))
        m_i = clip(1/(rho_i + eps), Q_0.01, Q_0.99)
        m_i_op = m_i / (mean(m_i) + eps).

    Masses are preprocessing data and never part of autograd.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    indices = scipy_knn_indices(xyz, k)
    offsets = xyz[indices] - xyz[:, None, :]
    squared_distance = np.einsum("nkd,nkd->nk", offsets, offsets)
    h2 = np.maximum(squared_distance.mean(axis=1), epsilon)
    kernel = np.exp(-squared_distance / (2.0 * (h2[:, None] + epsilon)))
    rho = kernel.mean(axis=1) / (2.0 * math.pi * (h2 + epsilon))
    inverse_density = 1.0 / (rho + epsilon)

    low, high = np.quantile(inverse_density, [lower_quantile, upper_quantile])
    clipped = np.clip(inverse_density, low, high)
    mass = clipped / max(float(clipped.mean()), epsilon)
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise FloatingPointError("surface-mass estimator produced invalid values")
    return mass.astype(np.float32, copy=False)

