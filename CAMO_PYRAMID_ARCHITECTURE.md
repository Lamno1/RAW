# CAMO-Pyramid: Measure-Preserving Proxy Operator Network for Building Segmentation

**Status:** proposed architecture for the next paper track  
**Date:** 2026-09-21  
**Target:** BuildingNet, 31-class point-wise architectural-part segmentation  
**Hardware envelope:** NVIDIA RTX 3050 Ti Laptop GPU, 4 GB VRAM; hard allocation gate 3.2 GB

## Executive decision

The next architecture should not be a larger PCT and should not stack OT, Mamba, Lovász, hierarchy, and routing in one training run. The proposed architecture is **CAMO-Pyramid**, whose single organizing principle is that every resolution is a discrete measure and every resolution change is a mass pushforward or pullback.

The core architecture has five components:

1. a full-cloud-to-working-measure assignment that accounts for every input mass;
2. gravity-preserving local measure operators at each resolution;
3. semi-relaxed sparse optimal-transport coarsening;
4. mass-aware global reasoning only on a small proxy measure;
5. a transport-adjoint decoder that reuses the encoder plans.

Long-tail losses, pretraining, and detail routing remain controlled extensions, not mandatory parts of the first end-to-end model.

## 1. Evidence that shapes the design

### Current project evidence

- The historical GeometryAwarePCT pilot reached about 14.18% sampled-cloud mIoU, but its fixed 1,024-point sampling, post-sampling normalization, and non-official metric prevent direct use as the final paper baseline.
- CAMO Prototype 1 ran reliably at 4,096 points under the 4 GB limit.
- Density-derived mass improved validation mIoU in all three paired seeds, but the mean gain was only +0.00070 and the paired test was inconclusive (`p ≈ 0.067`). Therefore mass awareness is retained as a prior, not treated as a proven headline contribution.
- Tail mIoU remains near zero. A better loss alone cannot recover tail parts that disappear during coarsening or sampling.

### Public research evidence

- Point Transformer V3 shows that efficient serialization and scale can matter more than increasingly intricate attention, but serialization is not inherently measure preserving. <https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html>
- Superpoint Transformer demonstrates that adaptive hierarchical partitions can achieve a strong accuracy-efficiency trade-off with a compact model. <https://openaccess.thecvf.com/content/ICCV2023/papers/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.pdf>
- SP2T supports a dual-stream local/proxy design and sparse proxy interactions, while identifying unreliable proxy sampling as a central failure mode. <https://openaccess.thecvf.com/content/ICCV2025/html/Wan_SP2T_Sparse_Proxy_Attention_for_Dual-stream_Point_Transformer_ICCV_2025_paper.html>
- Geometric operator learning with OT supports using instance-dependent transport to move computation to a simpler latent geometry, but it does not establish a ready-made segmentation architecture for BuildingNet. <https://jmlr.org/beta/papers/v26/25-1380.html>
- PointMamba and Voxel Mamba show that SSMs offer linear sequence processing, but their own motivation acknowledges that 3D-to-1D serialization can damage spatial proximity. Mamba is therefore an optional coarse-level comparator, not the main inductive bias. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html> and <https://proceedings.neurips.cc/paper_files/paper/2024/hash/947b63838c90f1485188b9c673bc3a14-Abstract-Conference.html>
- Sonata demonstrates that self-supervised 3D representations can be useful in data-limited regimes, but domain transfer from scans to mesh-derived building clouds must be measured rather than assumed. <https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.pdf>
- Lovász-Softmax provides a tractable surrogate for IoU; it is an objective ablation, not a replacement for detail-preserving representation. <https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html>
- Balancing Logit Variation shows that long-tailed segmentation has a feature-distribution problem beyond raw frequency weighting. <https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Balancing_Logit_Variation_for_Long-Tailed_Semantic_Segmentation_CVPR_2023_paper.html>

## 2. Mathematical object

At resolution level `l`, the point cloud is a discrete measure

\[
\mu_l=\sum_{i=1}^{N_l}m_i^{(l)}\,\delta_{(x_i^{(l)},a_i^{(l)},h_i^{(l)})},
\]

where `x` is position, `a` contains normal/RGB/geometric attributes, `h` is the learned feature, and `m` is positive quadrature mass.

The architecture must maintain four invariants:

1. \(m_i^{(l)}>0\);
2. total mass is preserved across each encoder pushforward;
3. constant features remain constant under decoder pullback;
4. local geometry is invariant to translation and rotation around gravity, not to arbitrary 3D rotations.

## 3. Architecture overview

```text
Full building: 100k points on CPU + cached mass/normalization
                         |
  streamed sparse assignment A: 100k -> 4,096 working atoms
                         |
             invariant attribute stem -> 64
                         |
            Level 0: 2 x LMAO, N0=4,096
                         |
          Sparse semi-relaxed OT P0: 4096 -> 1024
                         |
            Level 1: 2 x LMAO, N1=1,024
                         |
          Sparse semi-relaxed OT P1: 1024 -> 256
                         |
     Level 2: proxy global block, N2=256, width=128
                         |
       Pullback with P1 -> fuse Level-1 residual
                         |
       Pullback with P0 -> fuse Level-0 residual
                         |
                  linear 31-class head
                         |
     streamed pullback/refinement A -> full-cloud prediction
```

Initial deployment remains at `N0=4096, batch=8`. Scaling to `N0=8192, batch=4` is allowed only after a new hardware gate.

## 4. Stage A — full-building measure initialization and coverage assignment

Use the frozen Prototype 1 estimator and full-building normalization. Mass is computed once on the original 100k points, cached, and detached:

\[
\rho_i=
\frac{1}{2\pi(h_i^2+\epsilon)}
\frac1k\sum_{j\in\mathcal N_k(i)}
\exp\left[-\frac{d_{ij}^2}{2(h_i^2+\epsilon)}\right],
\]

\[
\hat m_i=(\rho_i+\epsilon)^{-1},\qquad
m_i=\frac{\operatorname{clip}(\hat m_i,Q_{.01},Q_{.99})}
{\operatorname{mean}[\operatorname{clip}(\hat m,Q_{.01},Q_{.99})]+\epsilon}.
\]

Selecting 4,096 points and discarding the remainder would destroy the full-building measure. Instead, choose 4,096 anchors without labels and construct a sparse row-stochastic assignment

\[
A_{is}\ge 0,\qquad \sum_s A_{is}=1,
\]

then push the complete input measure to the working measure:

\[
m_s^{(0)}=\sum_{i=1}^{N_{full}}m_iA_{is},\qquad
x_s^{(0)}=\frac{\sum_i m_iA_{is}x_i}{m_s^{(0)}}.
\]

This assignment is built on CPU or in bounded chunks and cached per crop/anchor manifest. Every full-cloud point must have at least one edge, and each anchor receives a self/coverage edge so every working atom has positive mass. Renormalizing masses after dropping points is explicitly forbidden because it repairs only the scalar total, not the spatial distribution.

Candidate anchors use a geometry-only coverage method such as weighted FPS or a tested sparse partition; labels are never used. Multiple precomputed anchor manifests may be rotated across epochs. The assignment `A` is retained for full-cloud reconstruction and voting.

Raw `XYZ, Nx, Ny` are not passed through a generic point MLP if an invariance claim is made. The initial point/atom stem uses RGB and gravity-preserving scalars such as normalized height, `n_z`, horizontal-normal magnitude, density, and curvature. Positions remain available only through invariant relative geometry. Because finite anchor selection and tie breaking need not commute exactly with azimuth rotation, the formal claim is limited to the local kernels and positional biases; whole-model azimuth robustness is measured empirically.

The initial implementation requires an explicit orientation audit before signed normals are used. The 2026-09-22 S4 pilot audit did not establish sign consistency, so the implementation default is now the coherent sign-invariant variant (`n_i^T n_j` becomes `|n_i^T n_j|`, `n_z` becomes `|n_z|`, and the normal cost changes accordingly). A naïve average of opposing normals is forbidden, and any reconstructed normal is renormalized only on a positive-norm mask. An unsigned second moment must never be converted back to an arbitrarily signed eigenvector and fed to signed equations.

## 5. Stage B — gravity-preserving Local Measure-Aware Operator

At every level, use the validated local operator:

\[
\alpha_{ij}^{(c)}=
\frac{m_j\kappa_\theta^{(c)}(g_{ij})}
{\sum_{r\in\mathcal N_k(i)}m_r\kappa_\theta^{(c)}(g_{ir})},
\]

\[
h_i^{l+1}=h_i^l+
\operatorname{SiLU}\left(W_1h_i^l+
\sum_j\alpha_{ij}\odot W_2h_j^l\right).
\]

The edge invariant is

\[
g_{ij}=[r_{ij}^{xy},\Delta z_{ij},d_{ij},n_i^Tn_j,n_i^Te_z,n_j^Te_z].
\]

Masses are positive and `kappa` is evaluated in FP32 with a positive floor, so the denominator is strictly positive on every valid neighborhood. Empty neighborhoods use an explicit identity fallback and increment an error counter. The normalized aggregation reproduces constant neighbor features in exact arithmetic; the complete residual nonlinear block maps a spatially constant input to another spatially constant output, not necessarily to the same value. An unconditional `+epsilon` in the denominator is not used because it silently breaks the normalized-aggregation identity. The block is accurately described as **measure aware and constant-reproducing at aggregation**. It is not claimed to conserve the mass-weighted feature integral.

## 6. Stage C — Semi-Relaxed Sparse OT Coarsening (SROT)

### 6.1 Why semi-relaxed OT

Balanced sparse OT can be infeasible when the kNN support cannot realize both marginals. Fully unbalanced OT weakens mass accounting. CAMO-Pyramid therefore fixes the source marginal and relaxes only the target capacity:

\[
P_l^*=\arg\min_{P\ge0,\,P\mathbf1=m^{(l)}}
\sum_{(i,j)\in E_l}P_{ij}C_{ij}
+\varepsilon\operatorname{KL}(P\|R)
+\tau\operatorname{KL}(P^T\mathbf1\|\nu).
\]

Here

\[
\operatorname{KL}(p\|q)=\sum_r p_r\log\frac{p_r}{q_r}-p_r+q_r,
\]

`nu_j>0` has the same total mass as the source, and the positive sparse reference is

\[
R_{ij}=m_i\frac{\nu_j}{\sum_{q:(i,q)\in E_l}\nu_q},\qquad (i,j)\in E_l.
\]

Every source must have at least one proxy edge. The hard row constraint guarantees

\[
\sum_jP_{ij}=m_i^{(l)},
\qquad
\sum_{ij}P_{ij}=\sum_i m_i^{(l)}.
\]

This avoids claiming exact target marginal matching while still preserving total source mass.

### 6.2 Sparse support

Candidate proxies are initialized by deterministic spatial stratification or weighted FPS. Each source connects to `k_p` nearest proxies plus reciprocal coverage edges:

\[
|E_l|\approx N_lk_p,
\qquad k_p\in\{4,8\}.
\]

No dense \([N_l,N_{l+1}]\) cost matrix is allowed. Duplicate edges are coalesced. Reciprocal coverage adds at most a documented constant number of edges per proxy so that `|E_l|=O(N_l k_p+N_{l+1})`.

For `K_ij=R_ij exp(-C_ij/epsilon)`, the semi-relaxed scaling iterations are implemented in the log domain:

\[
u\leftarrow m/(Kv),\qquad
v\leftarrow \left(\nu/(K^Tu)\right)^{\tau/(\varepsilon+\tau)}.
\]

The solver has both a maximum-iteration budget and convergence diagnostics; ten iterations is an initial benchmark setting, not a convergence claim. After the last iteration, an exact masked row normalization restores `P 1=m`. Log relative row residual, target KL, entropy, objective change, and iteration count. A dustbin is not part of the core path; if used for diagnostics, its mass is retained and reported rather than discarded.

### 6.3 Cost curriculum

Prototype 2a uses geometry only:

\[
C_{ij}=\alpha\,\widetilde{\|x_i-q_j\|^2}
+\beta\,\widetilde{(1-n_i^T\bar n_j)},
\]

where each term is robustly normalized by median/MAD and clamped. Feature-dependent cost is forbidden until geometry-only transport beats weighted FPS.

### 6.4 Pushforward

\[
m_j^{(l+1)}=\sum_iP_{ij},
\]

\[
x_j^{(l+1)}=
\frac{\sum_iP_{ij}x_i^{(l)}}{m_j^{(l+1)}},
\qquad
h_j^{(l+1)}=
\frac{\sum_iP_{ij}h_i^{(l)}}{m_j^{(l+1)}}.
\]

These divisions are evaluated only for active proxies with validated positive mass. Zero/near-zero proxies are repaired by the bounded coverage rule or marked as a failed transport plan; adding epsilon is not allowed to conceal them. All dual updates, segmented LogSumExp reductions, marginal residuals, and divisions run in FP32.

## 7. Stage D — coarse proxy global reasoning

Global reasoning is applied only at `N2=256`. At this size, mass-aware multi-head attention is affordable and easy to audit:

\[
s_{ij}=q_i^Tk_j/\sqrt d+B_{SO(2)}(i,j),\qquad
\alpha_{ij}=\operatorname{softmax}_j(s_{ij}+\log m_j),
\]

\[
H_2'=H_2+\sum_j\alpha_{ij}V(H_{2,j}).
\]

Masses are positive-masked and the score reduction runs in FP32. Positional bias uses gravity-preserving pair features. Its \(O(N_2^2)\) cost is bounded by design and does not scale with the original 100k points. Unweighted MHA remains a required control.

This is preferred over a Mamba dependency in the main model. A serialized SSM may be evaluated later as a comparator on the same 256 proxies.

## 8. Stage E — transport-adjoint decoder

Reuse the same transport plan rather than learning an unrelated interpolation. Define conditional pullback weights

\[
U_{ij}=\frac{P_{ij}}{m_i^{(l)}}.
\]

Because rows of `P` sum to source mass,

\[
\sum_jU_{ij}=1.
\]

Decode by

\[
\hat h_i^{(l)}=
\sum_jU_{ij}h_j^{(l+1)},
\]

then fuse with the saved local residual:

\[
h_{i,dec}^{(l)}=
\operatorname{MLP}([h_{i,enc}^{(l)},\hat h_i^{(l)}]).
\]

Let the encoder averaging operator be `A_l=D_{m^{l+1}}^{-1}P_l^T`. Under the mass-weighted inner products, `U_l=D_{m^l}^{-1}P_l` satisfies

\[
\langle A_lh,g\rangle_{m^{l+1}}=\langle h,U_lg\rangle_{m^l}.
\]

The equality holds on positive-mass active entries; implementation uses masked exact reciprocals and fails on invalid mass instead of perturbing the identity with epsilon. The decoder therefore preserves constants and is the weighted adjoint of the encoder average.

For the initial assignment, define `Q_is=m_i A_is`. Its encoder average is `D_{m^(0)}^{-1}Q^T`, and its weighted adjoint is exactly the conditional assignment `A`. After the final level-0 decoder, reconstruct full-cloud features as

\[
\hat h_i^{full}=\sum_s A_{is}h_s^{(0)}.
\]

An adjoint unit test is mandatory. Final logits may be refined by a small per-point invariant MLP using the pulled feature and local residual descriptors, streamed in bounded chunks. Nearest-anchor broadcasting is only a control.

## 9. Stage F — decision head and objective ladder

The base head is the existing linear classifier with no prior adjustment. This keeps the architecture control separate from long-tail decision rules.

Do not enable every loss simultaneously. The objective ladder is:

1. `B-CE` only for architecture validation;
2. `B-CE + Lovász` after the isolated P1.5 ablation;
3. optional cosine classifier, logit adjustment, boundary, or hierarchy objective only after the pyramid is stable.

Lovász skips absent classes in the current batch. Logit adjustment and the cosine classifier must each be tested separately from Lovász and from one another. The sign and train-versus-inference convention for prior correction must be pre-registered; no generic `-tau log(pi)` formula is treated as universally correct.

## 10. Complexity and 4 GB configuration

For batch `B`, local width `D`, local kNN `k`, proxy edge count `E`, Sinkhorn iterations `T`, and final proxy count `M=256`:

\[
T_{model}=O(BND^2+BNk\,C_\kappa(D)+BT|E|+B|E|D+BM^2D),
\]

where `C_kappa(D)=Theta(D²)` for the current channel-generating edge MLP. A future fixed-width or factorized kernel may reduce this term, but that reduction cannot be claimed for the present operator.

\[
S_{train}=O(BND+BNkD+B|E|+BM^2),
\]

unless the local kernel is fused or recomputed. Saved backward activations and optimizer state add implementation-dependent constants and must be measured. Graph construction time/memory is reported separately. The `M²` term is bounded and small. There is no dense attention or OT at `N>4000`.

Mandatory 100k stress calculation: `[B=8,N=100000,D=9]` FP32 input is about 27.5 MiB, but one `[8,100000,16,64]` edge tensor is 1,562.5 MiB in FP16 and 3,125 MiB in FP32. A gather plus kernel plus saved backward tensors would exceed 4 GB. Therefore 100k processing is restricted to CPU/streamed sparse assignment and chunked final reconstruction; LMAO never runs on all 100k points at batch 8.

### Numerical stability contract

- KDE bandwidths, logarithm arguments, and robust scale estimates use `epsilon=1e-6` or a scale-calibrated floor.
- Sparse Sinkhorn uses segmented LogSumExp with per-segment maxima, finite cost clipping, and FP32 duals.
- Mass divisions use `clamp(min=epsilon)` only as an implementation guard after a positive-mass mask/assertion. If the clamp activates on an active node, the plan is marked failed; it is not accepted as satisfying an exact conservation or adjoint identity.
- `NaN`, `Inf`, empty segments, non-convergence, and anomalous dustbin mass are fatal in dry-run/confirmatory modes. `nan_to_num` may not hide a core operator failure.
- AMP is limited to feature projections. Mass, costs, OT reductions, normalization denominators, Lovász sorting, and diagnostic residuals remain FP32.

Initial frozen hardware candidate:

| Item | Value |
|---|---:|
| Input representatives | 4,096 |
| Batch size | 8 |
| Pyramid | 4,096 → 1,024 → 256 |
| Widths | 64 → 96 → 128 |
| Local neighbors | 16, 16, 12 |
| Proxy neighbors | 8 |
| Sinkhorn iterations | max 100 with early stopping; S4 synthetic gate converged at 66 |
| Precision | AMP backbone; FP32 mass/OT/reductions |
| Allocation gate | ≤ 3.2 GB |

If this fails, reduce batch size before reducing point count, because point survival is a scientific variable.

## 11. Paper hypotheses

### Primary hypothesis

Sparse semi-relaxed transport produces a more stable multiresolution representation than FPS/voxel/random coarsening under equal proxy and compute budgets.

### Secondary hypotheses

1. Reusing the encoder transport plan for decoder pullback improves boundary and tail-part recovery over nearest-neighbor interpolation.
2. Gravity-preserving measure operators are more stable to sampling density and azimuth rotation than token-only local attention.
3. Global reasoning at a small proxy measure yields a better accuracy-memory Pareto point than global point attention.

Density weighting alone is not the headline claim because current evidence is directional but inconclusive.

## 12. Required ablations

| Axis | Control | Treatment |
|---|---|---|
| Local operator | uniform mass | density mass |
| Coarsening | weighted FPS/voxel | same-support row-softmax, then SROT |
| Decoder | nearest/inverse-distance interpolation | transport pullback |
| Global context | none/unweighted MHA | mass-aware proxy MHA |
| Objective | B-CE | B-CE + Lovász |
| Prior correction | none | logit adjustment |
| Resolution | 1k/4k | 8k if hardware passes |

Each scientific claim must change only one axis.

The paper comparison set must also include: the corrected historical bridge baseline; at least one modern point/serialization baseline; and one sparse voxel or superpoint baseline when a reproducible 4 GB configuration exists. Publicly reported official numbers and locally reproduced budget-matched numbers are shown in separate table columns and are never treated as directly interchangeable.

## 13. Falsification and stop rules

Abandon or simplify SROT if any condition holds:

- its relative mass residual `|sum(m_out)-sum(m_in)|/sum(m_in)` exceeds `1e-6` in FP32, or its absolute residual exceeds the scale-aware threshold frozen in S0;
- more than 0.1% of proxies receive near-zero mass after bounded reciprocal coverage;
- it exceeds weighted FPS by more than 25% wall time without improving proxy coverage or validation mIoU;
- it exceeds 3.2 GB allocated memory;
- gains disappear under repeated crops or the confirmatory seed set;
- transport pullback harms boundary/tail IoU despite similar global mIoU.

Do not backpropagate through Sinkhorn until detached-plan P2a demonstrates a forward benefit.

## 14. What is deliberately excluded from the core

- Dense Sinkhorn or dense pairwise attention above 4,000 points.
- Mamba as the primary backbone.
- Seven simultaneous losses.
- A hand-crafted semantic hierarchy before the flat classifier is stable.
- Dynamic scatter/gather routing before a static packed-patch control exists.
- Claims of Riemann convergence without the required sampling assumptions.

## 15. Intended paper narrative

> Building part segmentation is simultaneously a discretization problem and a long-tail recognition problem. Existing point token pipelines can lose small architectural parts when they subsample without tracking the surface measure. CAMO-Pyramid represents every resolution as a discrete measure, coarsens it with sparse semi-relaxed transport, reasons globally only over a compact proxy measure, and reconstructs point features using the adjoint of the same transport plan.

This narrative is narrower, more falsifiable, and more coherent than claiming a universal new backbone that is stronger than all existing models. Until the same-support soft-pooling control and modern budget-matched baselines are defeated, CAMO-Pyramid remains a proposed architecture rather than an established improvement.
