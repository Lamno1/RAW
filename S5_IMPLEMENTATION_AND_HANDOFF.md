# CAMO-Pyramid S5 — Implementation and Handoff

## Current decision

The single-level S5 architecture and all three controlled coarseners are implemented and mathematically tested. Training is intentionally **not authorized** yet: detached SROT fails the frozen latency gate while passing numerical and VRAM gates.

- Unit suite: **37 passed, 0 failed**.
- S0 verification: **passed**.
- Peak allocated VRAM: **949.74 MiB**, below **3200 MiB**.
- Paired row-softmax/hard latency ratio: **1.0954**, pass (`<=1.25`).
- Paired SROT/hard latency ratio: **2.4370**, fail (`>1.25`).
- No S5 segmentation metric or treatment checkpoint has been produced.

The authoritative machine-readable decision is `runs/s5_detached_srot_gate/status.json`.
The bounded optimization search and rejected variants are recorded in
`S5_OPTIMIZATION_AUDIT.md`; production remains on the reference solver.

## Mathematical contract

For source measure

\[
\mu=\sum_{i=1}^{N}m_i\delta_{(x_i,h_i)},\qquad m_i>0,
\]

and a bounded sparse support \(E\subset[N]\times[M]\), every arm satisfies the exact hard source marginal

\[
\sum_{j:(i,j)\in E}P_{ij}=m_i.
\]

The arms differ only in the definition of \(P\):

\[
P^{\text{hard}}_{ij}=m_i\mathbf 1[j=a(i)],
\]

\[
P^{\text{soft}}_{ij}=
m_i\frac{\exp(-C_{ij}/\varepsilon)}
{\sum_{r:(i,r)\in E}\exp(-C_{ir}/\varepsilon)},
\]

and

\[
P^{\text{SROT}}=
\arg\min_{P\ge0,\,P\mathbf 1=m}
\langle P,C\rangle+arepsilon\operatorname{KL}(P\|R)
+\tau\operatorname{KL}(P^\top\mathbf1\|\nu).
\]

Proxy mass, feature and position are the pushforward

\[
m'_j=\sum_iP_{ij},\qquad
h'_j=\frac{\sum_iP_{ij}h_i}{m'_j},\qquad
x'_j=\frac{\sum_iP_{ij}x_i}{m'_j}.
\]

The plan is detached; gradients flow through \(h_i\), not through support, cost, anchors or Sinkhorn scaling. The S5 decoder is conventional nearest-barycentric-proxy interpolation. Transport-adjoint decoding remains isolated to S6.

## Architecture held fixed across arms

`4096 fine points -> embedding -> fine LMAO -> 1024 proxies -> proxy LMAO -> nearest decoder -> skip fusion -> class head`

- Width: 64.
- Fine/proxy kNN: 16/16.
- Sparse transport support: 8 targets per source.
- SROT: `epsilon=0.2`, `tau=1`, at most 100 iterations, tolerance `1e-5`.
- Objective reserved for training: static balanced CE + `0.5 * Lovasz` under BF16.
- Anchors, support, trainable blocks, decoder and head are identical across arms.
- Labels do not enter anchor selection, support, cost or plan construction.

## Complexity and 4 GB audit

Let feature width be \(D\), local-neighbor count \(k_l\), transport support \(k_t\), proxy count \(M\), and Sinkhorn iterations \(T\).

- Local blocks: time `O(N*k_l*D + N*D^2)`, sparse activation space `O(N*k_l*D)`.
- Plan construction: time `O(T*N*k_t)`, solver state `O(N*k_t + M)`.
- Pushforward: time `O(N*k_t*D)`, space `O(N*k_t + M*D)`.
- No dense `[N,N]` or `[N,M]` tensor exists.

Mandatory stress calculation:

- Raw `[8,100000,9]` FP32 input: **27.47 MiB**.
- Forbidden gathered `[8,100000,16,64]` FP16 edge tensor: **1562.5 MiB** before gradients and other activations.
- Therefore 100k-point LMAO remains forbidden; the full cloud must stream/assign to the 4096-point working measure.

## Numerical stability audit

- Nonpositive masses are rejected; they are not silently replaced by epsilon.
- Geometry distance uses finite sparse edges only; normal cost is sign-invariant `1-|n_i^T n_j|`.
- SROT scaling is FP32/log-domain with LogSumExp.
- A final FP32 row correction restores `P 1=m`; non-convergence is a hard error.
- Every anchor is forced into its own bounded support row, preventing inactive hard proxies without artificial dust mass.
- Proxy axial normals are aligned to their selected anchor normals after eigendecomposition to eliminate arbitrary eigenvector signs before the signed frozen LMAO descriptor.
- The attempted generic over-relaxation/Anderson optimization was rejected: it did not meet latency and a deep-history variant was not reliably convergent at full batch.

## Implemented files

- `camo_build/coarsening.py`: three mass-conservative coarseners and barycentric pushforward.
- `camo_build/pyramid.py`: controlled S5 architecture.
- `camo_s5_dataset.py`: label-free weighted-FPS/voxel topology adapter.
- `tests/test_s5_coarsening.py`: conservation, control-equivalence, coverage and gradient tests.
- `benchmark_camo_s5.py`: independent numerical/VRAM gate.
- `benchmark_s5_paired_latency.py`: paired/interleaved latency gate.
- `protocol/s0/s5_experiment_preregistration.json`: frozen S5 metrics and decision rules.

## Stable handoff workflow

### 1. Re-establish integrity

```powershell
python -c "from camo_build.protocol import verify_protocol; print(verify_protocol('.', 'protocol/s0'))"
python -m pytest -q --basetemp=C:\Users\NVT\AppData\Local\Temp\camo_s5_verify
```

Required: S0 `passed: true` and all tests pass.

### 2. Profile only the SROT fixed-point loop

Do not alter the objective, support, `epsilon`, `tau`, tolerance or exact row correction. Profile segmented LogSumExp/scatter synchronization and kernel-launch overhead. An acceptable optimization must match the reference plan on small graphs, keep relative mass residual `<=1e-6`, converge at `1e-5`, and remain finite for the S4 adversarial fixtures.

Preferred implementation directions, in order:

1. A fused CUDA/Triton sparse segmented LogSumExp/update kernel.
2. A proven sparse OT backend with the same semi-relaxed objective and exact source marginal.
3. Asynchronous precomputation only if end-to-end latency—including preprocessing and I/O—is reported and remains within the same gate.

PyKeOps alone does not remove the sequential segmented Sinkhorn updates and must not be presented as an automatic solution.

### 3. Repeat gates before training

```powershell
python benchmark_camo_s5.py --mode hard --output benchmark_s5_hard.json
python benchmark_camo_s5.py --mode srot --hard-baseline benchmark_s5_hard.json --output benchmark_s5_srot.json
python benchmark_s5_paired_latency.py --candidate srot --output benchmark_s5_paired_srot.json
```

Training remains forbidden unless:

- allocated VRAM `<=3200 MiB`;
- paired SROT/hard median latency ratio `<=1.25`;
- all numerical gates and the complete test suite pass.

### 4. Only after a passing hardware gate

Implement the evaluation-only representation meter exactly as pre-registered, run one seed as `PILOT`, then freeze the runner and run paired seeds 42/123/2026 for hard, row-softmax and SROT. Do not enable differentiable transport unless SROT beats both controls on a qualifying representation metric and passes learned mIoU/accuracy non-inferiority.

### 5. Artifact discipline

Each training run must contain `config.yaml`, source/environment/split/crop manifests, `metrics.csv`, `hardware.json`, `checkpoint_rule.json`, `status.json`, best checkpoint and representation metrics. Failed runs remain on disk with error logs. Test-split predictions remain forbidden until S12.
