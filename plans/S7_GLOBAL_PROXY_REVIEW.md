# S7 — Coarse Global Proxy Block: Frozen Mathematical Review

## Scope

S7 tests global context only.  It does not reinterpret the frozen S5 result.
All arms use the same 4,096-to-256 row-softmax coarsener, local operators,
decoder, objective, crops, and checkpoint rule.  They differ only in the
coarse global block: identity, ordinary MHA, or measure-aware MHA.

The test split remains unopened.  S7 uses validation data only.

## Mathematical formulation

The input cloud is the discrete measure

\[
\mu_N=\sum_{i=1}^{N}m_i\,\delta_{(x_i,a_i)},\qquad m_i>0.
\]

The frozen row-softmax coarsener produces 256 proxy masses and features while
preserving the exact source marginal:

\[
P_{ij}=m_i\frac{\exp(-C_{ij}/\varepsilon)}
 {\sum_{r\in\mathcal N(i)}\exp(-C_{ir}/\varepsilon)},\quad
m'_j=\sum_iP_{ij},\quad
h'_j=\frac{\sum_iP_{ij}h_i}{m'_j}.
\]

For attention head `r`, define

\[
s^{(r)}_{ij}=\frac{q_i^{(r)\top}k_j^{(r)}}{\sqrt{d_r}}
+b_r(g_{ij})+\eta\log(m'_j+10^{-6}),
\]

\[
\alpha^{(r)}_{ij}=\frac{\exp(s^{(r)}_{ij})}
 {\sum_t\exp(s^{(r)}_{it})},\qquad
z_i^{(r)}=\sum_j\alpha^{(r)}_{ij}v_j^{(r)}.
\]

`eta=0` is the ordinary-MHA control and `eta=1` is the measure-aware arm.
The geometry bias receives only gravity-preserving, normal-sign-invariant
descriptors

\[
g_{ij}=[\|\Delta x_{xy}\|,\Delta z,\|\Delta x\|,
|n_i^Tn_j|,|n_{iz}|,|n_{jz}|].
\]

The residual block is pre-normalized:

\[
u=h+W_O[ z^{(1)}\|\cdots\|z^{(H)}],\qquad
h^+=u+\operatorname{MLP}(\operatorname{LN}(u)).
\]

All logits, `log(m+eps)`, softmax reductions, and finite checks are FP32 even
under BF16 autocast.  No mass is changed by attention; the coarsener remains
the only mass-moving operator.

## Complexity and 4 GB audit

For `B=8`, `M=256`, `D=64`, and `H=4`:

- attention time: `O(B*H*M^2*(D/H))`;
- attention storage: `O(B*H*M^2)`;
- FP32 logits: `8*4*256*256*4 = 8 MiB`;
- Q/K/V in BF16: approximately `0.75 MiB` total;
- six-channel pair geometry in FP32: `12 MiB` before its bias projection.

The implementation must avoid materializing `[B,M,M,6]`; it computes the six
geometry terms one at a time and accumulates a per-head bias `[B,H,M,M]`.
For the required `[B=8,N=100k,D=9]` hardware thought experiment, the raw FP32
input is about 27.5 MiB, but training still chunks/samples to 4,096 points and
the quadratic block is applied only after coarsening to 256 proxies.  A dense
`100k x 100k` tensor is forbidden and never constructed.

## Numerical stability

- masses must be finite and strictly positive; no silent epsilon repair;
- `log(m+1e-6)` is evaluated in FP32 and centered per cloud to remove an
  irrelevant common offset;
- attention logits are clamped to `[-80,80]` before FP32 softmax;
- normals are normalized with `eps=1e-6` and used sign-invariantly;
- every output and every training gradient must be finite;
- VRAM allocated must remain at or below 3,200 MiB.

## Falsification workflow

1. Dry-run all three arms for one train and one validation batch.
2. Pilot all three arms with seed 42 for five epochs.
3. Select the pilot candidate by validation Part-IoU; ties within `1e-4` favor
   the simpler arm in order: identity, ordinary MHA, measure-aware MHA.
4. Only if an attention arm wins the pilot, run paired 50-epoch confirmatory
   experiments against the identity arm on seeds 42, 123, and 2026.
5. Promote S7 only if mean paired Part-IoU gain is at least
   `0.0044353341422272395`, mean paired accuracy delta is at least
   `-0.0043557722327992885`, latency ratio is at most 1.25, allocated VRAM is
   at most 3,200 MiB, and the improvement is not head-only.
6. Otherwise retain frozen S5 and do not add the global block to S8.

