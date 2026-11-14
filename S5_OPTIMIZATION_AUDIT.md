# S5 SROT optimization audit

The production solver remains the frozen reference implementation in
`camo_build/transport.py`. No optimization variant is promoted unless it is
numerically equivalent and passes the paired `<=1.25x` latency gate.

| Variant | Correctness | Paired SROT/hard ratio | Decision |
|---|---:|---:|---|
| Reference per-iteration host check | pass | 2.4370x | baseline; gate failure |
| Over-relaxed fixed point (`omega=1.8`) | small-fixture pass; full path finite | 1.8399x | reject |
| Over-relaxed fixed point (`omega=1.95`) | small-fixture pass; full path finite | 1.8326x | reject |
| Interval-check scalar reduction | small-fixture pass | 2.5572x | reject |
| `torch.segment_reduce` grouped edges | small-fixture pass | 2.8190x | reject |
| Warm-start dual, repeated identical batch | plan/objective pass | 1.1756x | reject as non-representative |
| Warm-start dual, alternating batches | plan/objective pass | 2.3971x | reject |
| Cold dual, alternating batches (paired recheck) | plan/objective pass | 2.4189x | reference recheck |
| Triton FP32 fused source/target SROT reductions + prepared gather-reduce pushforward | plan max error 1.2e-7; pushforward adjoint error 8.9e-8 | 3.2829x full-model gate | reject; solver/pushforward kernels do not amortize the decoder path |
| Triton gather-reduce pushforward + custom adjoint | forward error 1.8e-7; adjoint error 7.5e-8 | included above | reject; numerically valid but end-to-end gate still fails |

The reductions were measured with the same paired interleaved forward,
backward and optimizer protocol. The rejected variants are not the default
production path; warm-start remains an explicitly opt-in experimental branch
solely so its negative result is reproducible. The frozen S5 runner never
enables it.

Warm-start demonstrates why the workload definition matters: it passes only
when the exact same geometry/support is repeated. Alternating two independent
clouds removes that benefit, so carrying dual state across real training
batches is not an accepted optimization.

The remaining viable acceleration paths require a fused sparse CUDA/Triton
kernel or a backend with a demonstrably equivalent semi-relaxed objective.
PyKeOps by itself does not fuse the sequential dual updates.

## 2026-09-24 toolchain result

The Windows toolchain was made reproducible without replacing the display
driver: Triton-Windows 3.2.0 on PyTorch 2.6.0+cu124, MSVC 19.44 from Visual
Studio Build Tools 2022, and the official CUDA 12.4.131 `cuda_nvcc` redist
were installed. Triton's bundled ptxas/TinyCC and the redist nvcc both target
the RTX 3050 Ti sm86. The fused kernels are opt-in only; the reference solver
and frozen runner remain unchanged after the negative gate.
