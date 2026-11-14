"""Mass-conservative detached coarsening for the S5 ablation.

For a fixed sparse support ``E`` the three controls are

``hard``
    ``P_ij = m_i 1[j = a(i)]`` with spatial nearest-anchor assignment.
``row_softmax``
    ``P_ij = m_i exp(-C_ij/eps) / sum_r exp(-C_ir/eps)``.
``srot``
    ``argmin_P <P,C> + eps KL(P||R) + tau KL(P^T1||nu)`` subject to
    the exact source marginal ``P1=m``.

All plans are detached.  Feature pushforward remains differentiable with
respect to the source features, but never through anchor/support selection or
the transport solver.  No dense source-by-proxy tensor is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .sparse_ops import segment_sum
from .transport import (
    SparseTransportPlan,
    TransportDiagnostics,
    geometry_transport_cost,
    pushforward_axial_normals,
    semi_relaxed_sparse_transport,
)
from .triton_transport import (
    prepare_target_csr,
    pushforward_triton,
    pushforward_triton_autograd,
    semi_relaxed_sparse_transport_triton,
)
from .warmstart_transport import semi_relaxed_sparse_transport_warm


COARSENING_MODES = frozenset({"hard", "row_softmax", "srot"})


@dataclass(frozen=True)
class CoarseningResult:
    """Batched tensors produced by one detached sparse transport plan."""

    proxy_xyz: torch.Tensor
    proxy_normals: torch.Tensor
    proxy_features: torch.Tensor
    proxy_mass: torch.Tensor
    decode_index: torch.Tensor
    plan: SparseTransportPlan
    diagnostics: TransportDiagnostics | None


def build_fixed_proxy_support(
    xyz: np.ndarray,
    anchor_index: np.ndarray,
    *,
    k: int = 8,
) -> np.ndarray:
    """Return a fixed-size source-to-proxy kNN support ``[N,k]``.

    Every selected anchor is forced into its own source row.  This bounded
    reciprocal edge gives every proxy positive in-degree without adding a
    variable number of edges, so ordinary ``DataLoader`` collation remains
    possible.  Labels are intentionally absent from the interface.
    """

    points = np.asarray(xyz, dtype=np.float32)
    anchors = np.asarray(anchor_index, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if anchors.ndim != 1 or anchors.size == 0:
        raise ValueError("anchor_index must be a non-empty vector")
    if not np.isfinite(points).all():
        raise ValueError("xyz contains NaN or Inf")
    if np.unique(anchors).size != anchors.size:
        raise ValueError("anchor indices must be unique")
    if anchors.min() < 0 or anchors.max() >= len(points):
        raise ValueError("anchor index is outside the source cloud")
    if not 0 < k <= anchors.size:
        raise ValueError("k must satisfy 0 < k <= number of anchors")

    _, target = cKDTree(points[anchors]).query(points, k=k, workers=-1)
    if k == 1:
        target = target[:, None]
    target = np.asarray(target, dtype=np.int64)
    # Duplicate coordinates can cause a different coincident anchor to be
    # returned.  The forced self edge makes target coverage a hard invariant.
    for proxy, source in enumerate(anchors):
        if proxy not in target[source]:
            target[source, -1] = proxy
    if np.any(np.bincount(target.reshape(-1), minlength=anchors.size) == 0):
        raise RuntimeError("fixed support left a proxy without an incoming edge")
    return target


def _validate_batched_inputs(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    features: torch.Tensor,
    mass: torch.Tensor,
    anchor_index: torch.Tensor,
    support_target: torch.Tensor,
) -> tuple[int, int, int, int]:
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or normals.shape != xyz.shape:
        raise ValueError("xyz and normals must have shape [B,N,3]")
    if features.ndim != 3 or features.shape[:2] != xyz.shape[:2]:
        raise ValueError("features must have shape [B,N,D]")
    if mass.shape != xyz.shape[:2]:
        raise ValueError("mass must have shape [B,N]")
    if anchor_index.ndim != 2 or anchor_index.shape[0] != xyz.shape[0]:
        raise ValueError("anchor_index must have shape [B,M]")
    if support_target.ndim != 3 or support_target.shape[:2] != xyz.shape[:2]:
        raise ValueError("support_target must have shape [B,N,k]")
    if anchor_index.dtype != torch.long or support_target.dtype != torch.long:
        raise TypeError("anchor and support indices must be int64")
    if not torch.isfinite(xyz).all() or not torch.isfinite(normals).all():
        raise FloatingPointError("coarsening geometry contains NaN or Inf")
    if not torch.isfinite(features).all() or not torch.isfinite(mass).all():
        raise FloatingPointError("coarsening features or masses contain NaN or Inf")
    if torch.any(mass <= 0):
        raise ValueError("coarsening requires strictly positive source masses")
    batch, points, _ = xyz.shape
    proxies = anchor_index.shape[1]
    support_k = support_target.shape[2]
    if proxies <= 0 or support_k <= 0 or support_k > proxies:
        raise ValueError("invalid proxy or support count")
    if int(anchor_index.min()) < 0 or int(anchor_index.max()) >= points:
        raise ValueError("anchor index is outside [0,N)")
    if int(support_target.min()) < 0 or int(support_target.max()) >= proxies:
        raise ValueError("support target is outside [0,M)")
    return batch, points, proxies, support_k


class SparseMeasureCoarsener(nn.Module):
    """Construct and apply one of the frozen S5 detached coarseners."""

    def __init__(
        self,
        mode: str = "srot",
        *,
        epsilon: float = 0.2,
        tau: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_weight: float = 1.0,
        normal_weight: float = 0.25,
        warm_start: bool = False,
        transport_backend: str = "reference",
    ) -> None:
        super().__init__()
        if mode not in COARSENING_MODES:
            raise ValueError(f"mode must be one of {sorted(COARSENING_MODES)}")
        if epsilon <= 0 or tau < 0 or max_iterations <= 0 or tolerance <= 0:
            raise ValueError("invalid transport hyperparameter")
        self.mode = mode
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.distance_weight = float(distance_weight)
        self.normal_weight = float(normal_weight)
        self.warm_start = bool(warm_start)
        if transport_backend not in {"reference", "triton", "triton_hybrid"}:
            raise ValueError(
                "transport_backend must be 'reference', 'triton', or 'triton_hybrid'"
            )
        self.transport_backend = transport_backend
        self._warm_log_v: torch.Tensor | None = None

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        features: torch.Tensor,
        mass: torch.Tensor,
        anchor_index: torch.Tensor,
        support_target: torch.Tensor,
    ) -> CoarseningResult:
        batch, points, proxies, support_k = _validate_batched_inputs(
            xyz, normals, features, mass, anchor_index, support_target
        )
        device = xyz.device
        anchor_index = anchor_index.to(device=device)
        support_target = support_target.to(device=device)
        batch_index = torch.arange(batch, device=device)[:, None]
        seed_xyz = xyz[batch_index, anchor_index]
        seed_normals = normals[batch_index, anchor_index]

        source_local = torch.arange(points, device=device)[:, None].expand(
            points, support_k
        )
        edge_source_parts: list[torch.Tensor] = []
        edge_target_parts: list[torch.Tensor] = []
        edge_cost_parts: list[torch.Tensor] = []
        for item in range(batch):
            local_source = source_local.reshape(-1)
            local_target = support_target[item].reshape(-1)
            local_cost = geometry_transport_cost(
                xyz[item], seed_xyz[item], local_source, local_target,
                source_normals=normals[item], proxy_normals=seed_normals[item],
                distance_weight=self.distance_weight,
                normal_weight=self.normal_weight,
                sign_invariant_normals=True,
            )
            edge_source_parts.append(local_source + item * points)
            edge_target_parts.append(local_target + item * proxies)
            edge_cost_parts.append(local_cost)
        edge_source = torch.cat(edge_source_parts)
        edge_target = torch.cat(edge_target_parts)
        edge_cost = torch.cat(edge_cost_parts).float()
        flat_mass = mass.detach().reshape(-1).float()

        diagnostics: TransportDiagnostics | None
        if self.mode == "hard":
            # The hard weighted-FPS/voxel control is a spatial Voronoi
            # partition.  Normal terms are not allowed to change its cells.
            candidates = seed_xyz[
                torch.arange(batch, device=device)[:, None, None], support_target
            ]
            squared = (xyz[:, :, None, :].float() - candidates.float()).square().sum(-1)
            choice = torch.gather(
                support_target, 2, squared.argmin(dim=-1, keepdim=True)
            ).squeeze(-1)
            # Force each anchor's own positive mass into its proxy.  This is a
            # hard coverage constraint, not an epsilon mass patch.
            choice[batch_index, anchor_index] = torch.arange(
                proxies, device=device
            )[None, :]
            hard_source = torch.arange(batch * points, device=device)
            hard_target = choice + torch.arange(batch, device=device)[:, None] * proxies
            hard_target = hard_target.reshape(-1)
            target_mass = segment_sum(flat_mass, hard_target, batch * proxies)
            if torch.any(target_mass <= 0):
                raise FloatingPointError("hard coarsening produced an inactive proxy")
            plan = SparseTransportPlan(
                hard_source, hard_target, flat_mass.detach(), flat_mass.detach(),
                target_mass.detach(),
            )
            diagnostics = None
        elif self.mode == "row_softmax":
            plan, diagnostics = semi_relaxed_sparse_transport(
                flat_mass,
                edge_source,
                edge_target,
                edge_cost,
                num_targets=batch * proxies,
                epsilon=self.epsilon,
                tau=0.0,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                detach_plan=True,
            )
        else:
            if self.transport_backend in {"triton", "triton_hybrid"}:
                if self.warm_start:
                    raise ValueError("warm_start is not supported by Triton backend")
                values, target_mass, iterations, converged, final_change = (
                    semi_relaxed_sparse_transport_triton(
                        flat_mass,
                        edge_source,
                        edge_target,
                        edge_cost,
                        num_targets=batch * proxies,
                        epsilon=self.epsilon,
                        tau=self.tau,
                        max_iterations=self.max_iterations,
                        tolerance=self.tolerance,
                        target_backend=(
                            "native" if self.transport_backend == "triton_hybrid" else "triton"
                        ),
                    )
                )
                plan = SparseTransportPlan(
                    edge_source,
                    edge_target,
                    values,
                    flat_mass.detach(),
                    target_mass.detach(),
                )
                diagnostics = TransportDiagnostics(
                    iterations=iterations,
                    converged=converged,
                    # The Triton solver performs the FP32 row correction and
                    # validates finite/positive target mass before returning.
                    # Keep this hot path scalar-free; the full diagnostics are
                    # recomputed by the offline gate from the returned plan.
                    absolute_mass_residual=0.0,
                    relative_mass_residual=0.0,
                    max_relative_row_residual=0.0,
                    target_kl=0.0,
                    entropy=0.0,
                    objective=0.0,
                    max_log_scaling_change=float(final_change),
                )
            elif self.warm_start:
                plan, diagnostics, dual = semi_relaxed_sparse_transport_warm(
                    flat_mass,
                    edge_source,
                    edge_target,
                    edge_cost,
                    num_targets=batch * proxies,
                    initial_log_v=self._warm_log_v,
                    epsilon=self.epsilon,
                    tau=self.tau,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                )
                self._warm_log_v = dual
            else:
                plan, diagnostics = semi_relaxed_sparse_transport(
                    flat_mass,
                    edge_source,
                    edge_target,
                    edge_cost,
                    num_targets=batch * proxies,
                    epsilon=self.epsilon,
                    tau=self.tau,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    detach_plan=True,
                )
            if not diagnostics.converged:
                raise FloatingPointError(
                    "SROT did not reach the frozen convergence tolerance: "
                    f"{diagnostics}"
                )

        if self.transport_backend in {"triton", "triton_hybrid"}:
            target_csr = prepare_target_csr(
                edge_source, edge_target, batch * proxies
            )
            proxy_xyz = pushforward_triton(
                xyz.reshape(batch * points, 3), edge_source, edge_target,
                plan.value, plan.target_mass, num_targets=batch * proxies,
                prepared=target_csr,
            ).reshape(batch, proxies, 3)
            proxy_features = pushforward_triton_autograd(
                features.reshape(batch * points, features.shape[-1]),
                edge_source, edge_target, plan.value, plan.target_mass,
                num_targets=batch * proxies,
                prepared=target_csr,
            ).reshape(batch, proxies, features.shape[-1])
            normal = torch.nn.functional.normalize(
                normals.reshape(batch * points, 3).float(), dim=-1, eps=1e-6
            )
            second_moment = normal[:, :, None] * normal[:, None, :]
            target_moment = pushforward_triton(
                second_moment.reshape(batch * points, 9), edge_source, edge_target,
                plan.value, plan.target_mass, num_targets=batch * proxies,
                prepared=target_csr,
            ).reshape(batch * proxies, 3, 3)
            target_moment = 0.5 * (target_moment + target_moment.transpose(-1, -2))
            _, eigenvectors = torch.linalg.eigh(target_moment)
            proxy_axis = torch.nn.functional.normalize(eigenvectors[..., -1], dim=-1)
            proxy_axis = proxy_axis.reshape(batch, proxies, 3)
        else:
            proxy_xyz = plan.pushforward(xyz.reshape(batch * points, 3)).reshape(
                batch, proxies, 3
            )
            proxy_features = plan.pushforward(
                features.reshape(batch * points, features.shape[-1])
            ).reshape(batch, proxies, features.shape[-1])
            proxy_axis = pushforward_axial_normals(
                plan, normals.reshape(batch * points, 3)
            ).reshape(batch, proxies, 3)
        # eigh returns an unoriented axis.  Align it to the selected source
        # normal so downstream frozen signed LMAO descriptors are deterministic.
        alignment = (proxy_axis * seed_normals.float()).sum(-1, keepdim=True)
        proxy_normals = torch.where(alignment < 0, -proxy_axis, proxy_axis)
        proxy_mass = plan.target_mass.reshape(batch, proxies)

        # Conventional S5 decoder: nearest barycentric proxy restricted to the
        # same sparse support.  Plan pullback is deliberately deferred to S6.
        decoded_candidates = proxy_xyz[
            torch.arange(batch, device=device)[:, None, None], support_target
        ]
        decoded_distance = (
            xyz[:, :, None, :].float() - decoded_candidates.float()
        ).square().sum(-1)
        decode_index = torch.gather(
            support_target, 2, decoded_distance.argmin(-1, keepdim=True)
        ).squeeze(-1)

        outputs = (proxy_xyz, proxy_normals, proxy_features, proxy_mass)
        if any(not torch.isfinite(value).all() for value in outputs):
            raise FloatingPointError("coarsening output contains NaN or Inf")
        return CoarseningResult(
            proxy_xyz=proxy_xyz,
            proxy_normals=proxy_normals,
            proxy_features=proxy_features,
            proxy_mass=proxy_mass,
            decode_index=decode_index,
            plan=plan,
            diagnostics=diagnostics,
        )
