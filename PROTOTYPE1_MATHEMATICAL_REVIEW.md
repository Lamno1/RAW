# CAMO-Build Prototype 1 — Mathematical and Implementation Review

**Status:** **FROZEN — all nine freeze conditions formally accepted by the project owner.**  
**Date:** 2026-09-20  
**Scope:** Mass estimator, local measure-aware operator, loss, and hardware protocol.

## Freeze record

- **Decision:** Accepted without exception.
- **Effective date:** 2026-09-20.
- **Implementation authorization:** Prototype 1 may now be implemented exactly against this specification.
- **Change control:** Any modification to the mass estimator, operator equation, kernel bounds, neighborhood construction, loss controls, normalization order, or hardware gates requires a versioned amendment and a new explicit acceptance. Silent mathematical changes in code are prohibited.
- **Isolation rule:** Initial implementation and benchmarking must be performed in standalone modules/tests before integration into the existing training pipeline.

## 1. Decision

The proposed implementation must not be merged or trained as written. It contains four blocking issues:

1. the adaptive KDE omits the \(h_i^{-2}\) normalization and therefore largely cancels the density signal it is intended to estimate;
2. the local normalized kernel preserves constant features but does not, in general, conserve the \(m\)-weighted feature integral;
3. dense kNN at \(N=4096\) violates the project's no-dense-\(O(N^2)\)-above-4000 rule and its quoted peak VRAM excludes major allocations;
4. applying \(W_2\) after gathering neighbors introduces avoidable \(O(NkD^2)\) work and a large autograd tensor.

## 2. Mass estimator correction

### 2.1. Why the proposed KDE is not accepted

The proposed estimator is

\[
\rho_i=\frac1k\sum_{j\in\mathcal N_k(i)}
\exp\left(-\frac{d_{ij}^2}{2h_i^2+\epsilon}\right),
\qquad h_i^2=\frac1k\sum_jd_{ij}^2.
\]

If a neighborhood is locally scaled by a factor \(s\), both \(d_{ij}^2\) and \(h_i^2\) scale by \(s^2\). Their ratio is approximately unchanged. Consequently \(\rho_i\) is almost scale invariant and cannot reliably distinguish dense from sparse neighborhoods.

Duplicate points do not make this \(\rho_i\) explode: the exponential is bounded by one. The actual failure is loss of density information and possible collapse of \(h_i\) to its clamp.

### 2.2. Freeze candidate

Assume the building surface has intrinsic dimension \(d=2\). Exclude the query point itself from \(\mathcal N_k(i)\). Define

\[
h_i^2=\frac1k\sum_{j\in\mathcal N_k(i)}d_{ij}^2,
\]

\[
\rho_i=
\frac{1}{2\pi(h_i^2+\epsilon)}
\frac1k\sum_{j\in\mathcal N_k(i)}
\exp\left(-\frac{d_{ij}^2}{2(h_i^2+\epsilon)}\right).
\]

Then

\[
\hat m_i=(\rho_i+\epsilon)^{-1}.
\]

To prevent a single isolated point from dominating, apply robust clipping using train-independent per-building statistics:

\[
\tilde m_i=\operatorname{clip}(\hat m_i,
Q_{0.01}(\hat m),Q_{0.99}(\hat m)).
\]

For numerical conditioning inside the local operator, use mean-one weights

\[
m_i^{op}=\frac{\tilde m_i}{N^{-1}\sum_n\tilde m_n+\epsilon}.
\]

Mean-one scaling does not change a normalized kernel but avoids local partition values of order \(k/N\).

### 2.3. Where masses must be computed

Masses must be estimated once on the full cached building cloud, outside autograd, and cached point-aligned with the original data. The sampled/cropped points inherit their cached masses. Estimating normalization geometry after random sampling is not accepted because the current dataset centers and scales each sampled subset independently.

The full-building coordinate transform must be computed before sampling and cached. At present, `dataset.py` samples first and derives center/scale from the selected subset; two samples of the same building therefore define different geometry and different invariants.

## 3. Operator semantics correction

### 3.1. What the proposed operator actually guarantees

For

\[
K_{ij}=\frac{m_j\kappa_{ij}}
{\sum_r m_r\kappa_{ir}+\epsilon},
\]

the rows approximately sum to one, so \(K\mathbf1\approx\mathbf1\). It preserves constant signals. It does **not** generally satisfy

\[
\sum_i m_i(Kh)_i=\sum_i m_ih_i.
\]

Thus it is a measure-aware normalized integral operator, not a conservative feature-transport operator. Prototype 1 must use the accurate name **Local Measure-Aware Operator (LMAO)** unless a reversible or mass-stochastic construction is introduced and tested.

The point masses themselves remain unchanged in Prototype 1 because there is no pooling. That is the only literal mass-conservation claim permitted at this stage.

### 3.2. Accepted forward equation candidate

\[
v_i=W_2h_i,
\]

\[
\alpha_{ij}^{(c)}=
\frac{m_j^{op}\,\kappa_\theta^{(c)}(g_{ij})}
{\sum_{r\in\mathcal N_k(i)}m_r^{op}\,
\kappa_\theta^{(c)}(g_{ir})+\epsilon},
\]

\[
h_i^{l+1}=h_i^l+
\operatorname{SiLU}\left(W_1h_i^l+
\sum_{j\in\mathcal N_k(i)}\alpha_{ij}\odot v_j\right).
\]

\(W_2\) is evaluated before neighbor gather. This changes projection cost from \(O(NkD^2)\) to \(O(ND^2)\); gather and aggregation remain \(O(NkD)\).

Kernel output must be strictly positive and bounded during Prototype 1:

\[
\kappa_\theta(g)=\kappa_{min}+
(\kappa_{max}-\kappa_{min})\sigma(f_\theta(g)).
\]

An unbounded Softplus kernel is not frozen because its scale is unidentifiable after normalization and can create poorly conditioned gradients.

All weighted reductions and division by \(Z_i\) must execute in FP32. AMP may be used for the MLP and linear projections.

## 4. Geometry invariants audit

The proposed six-dimensional edge vector is invariant to translation and rotation around \(z\):

\[
g_{ij}=[r_{ij},\Delta z_{ij},d_{ij},
n_i^Tn_j,n_i^Te_z,n_j^Te_z].
\]

Before freezing it, the following data assumptions must be audited:

- normal vectors are unit length;
- normal orientations are consistent across each building;
- vertical axis is consistent across all samples;
- duplicate points and zero-length edges are counted;
- self-neighbors are explicitly excluded.

Clamp dot products to \([-1,1]\) to contain floating-point drift. Distances use \(\sqrt{d^2+\epsilon}\), but squared distances should remain unclamped at zero for neighbor ranking; the diagonal must instead be masked to \(+\infty\).

## 5. Complexity and hardware correction

### 5.1. Accepted asymptotic cost

Assuming the kNN graph is supplied by a sparse backend:

\[
T=O(ND^2+NkD_{geom}D+NkD),
\]

\[
M=O(ND+Nk+NkD_{geom}+NkD)
\]

before fused-kernel optimization. The implementation must not materialize multiple \([B,N,k,D]\) tensors simultaneously.

### 5.2. Dense kNN is rejected at 4096

`[8,4096,4096]` FP32 alone is 512 MiB, not a complete peak estimate. Matrix multiplication output, top-k workspace, indices, gathered activations, kernel outputs, saved tensors for backward, gradients, optimizer states, CUDA context and allocator fragmentation are additional.

Therefore the claim that the module peaks near 650 MB is unsupported. The mandatory backend from the first Prototype 1 benchmark is `torch_cluster.knn_graph`, `pytorch3d.ops.knn_points`, FAISS/CUDA, or another measured sparse/chunked kNN implementation. A dense fallback is permitted only for unit tests with \(N\le1024\).

## 6. Loss and experimental control

The static weighted multiclass CE equation is accepted:

\[
\mathcal L_{B\text{-}CE}=
-\frac{\sum_{i\in\mathcal V}w_{y_i}\log p_{i,y_i}}
{\max(\sum_{i\in\mathcal V}w_{y_i},\epsilon)}.
\]

However, an all-ignore batch must raise a data error or be skipped explicitly; silently returning zero hides pipeline failures.

The current weights are finite and nonzero, but range only from approximately 0.465 to 1.097 despite a class-count ratio above 1100:1. They are mild frequency smoothing, not strong balancing.

To isolate the mass hypothesis, Prototype 1 requires three controls under the exact same loss and training schedule:

1. local kernel without \(m_j\);
2. local kernel with uniform \(m_j=1\);
3. local kernel with cached density-derived \(m_j^{op}\).

Comparison only against the existing global PCT cannot attribute a gain specifically to measure weighting because locality, kernel form and architecture all change simultaneously.

## 7. Benchmark protocol correction

Correctness and performance tests must be separate.

### Debug gate

- small \(N\);
- anomaly detection enabled;
- finite-value assertions after mass, kernel, partition and output;
- gradient checks on learnable parameters;
- no expected gradient for cached masses;
- equivariance/invariance unit tests.

### Performance gate

- anomaly detection disabled;
- fixed tensor shapes and production sparse kNN backend;
- CUDA synchronization around timing;
- warm-up followed by measured forward, backward and optimizer steps;
- report both `max_memory_allocated` and `max_memory_reserved` after resetting peak statistics;
- repeat in a fresh process to avoid allocator-history bias.

The 50,000 points/s threshold is not frozen because it is arbitrary. The threshold must be derived from measured full-step time and the required wall-clock budget, with data loading and validation included separately.

### Memory gate

The 3.2 GB allocated ceiling remains accepted. Reserved memory must also remain below the physical 4 GB margin in the actual environment. Passing ten dummy batches does not prove absence of fragmentation or validation-time peaks; both train and validation paths must be benchmarked.

## 8. Frozen conditions

The following accepted conditions are now binding:

- full-cloud normalization is cached before sampling;
- mass estimator uses the \(h_i^{-2}\) density factor or an explicitly chosen kNN-cell-area estimator;
- self-neighbors are excluded;
- masses are cached and detached;
- operator is named measure-aware, not feature-mass-conservative;
- kernel range \([\kappa_{min},\kappa_{max}]\) is fixed;
- sparse kNN backend is available;
- the three attribution controls are included;
- debug and performance gates are separate.

These conditions were formally accepted on 2026-09-20. PyTorch implementation is authorized subject to exact equation-to-code traceability and the standalone debug, memory, and performance gates above.
