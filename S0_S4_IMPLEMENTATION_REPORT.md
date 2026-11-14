# CAMO-Pyramid S0–S4 implementation report

**Date:** 2026-09-22  
**Scope:** implementation and dry/hardware gates; long confirmatory training is not included.

## Status matrix

| Stage | Code | Dry/unit gate | Long experiment status |
|---|---|---|---|
| S0 protocol/evaluator | complete | passed | protocol frozen in `protocol/s0` |
| S1 corrected bridge | runner complete | one-batch GPU dry-run passed | 3 seeds and modern budget-matched control pending |
| S2 Lovász ablation | complete | gradient/NaN GPU gate passed | paired 3-seed ablation pending |
| S3 sparse backend | complete | full audit passed | native PyTorch edge-list backend selected |
| S4 assignment + SROT | complete | 100k/4096/1024/256 gate passed | real-data representation ablation begins at S5 |

No S1/S2 paper claim is permitted from the dry runs.

## Mathematical implementation

The full-cloud assignment stores the mass plan

\[
Q_{is}=m_iA_{is},\qquad \sum_s A_{is}=1,
\]

as coalesced edge vectors. SROT optimizes on a fixed sparse support

\[
\min_{P\ge0,\,P\mathbf 1=m}
\langle P,C\rangle+arepsilon\,\mathrm{KL}(P\|R)
+\tau\,\mathrm{KL}(P^T\mathbf1\|\nu).
\]

The hard source marginal is restored by a final FP32 log-domain row
normalization. `SparseTransportPlan.pushforward` implements
`D_target^-1 P^T`; `pullback` implements its mass-weighted adjoint
`D_source^-1 P`.

Differentiable SROT is deliberately rejected with `NotImplementedError` until
the detached S5 experiment demonstrates a forward benefit.

## Measured gates

### S3 backend

- Environment: PyTorch 2.6.0 + CUDA 12.4, RTX 3050 Ti Laptop GPU.
- cKDTree graph `N=4096, k=16`: about 0.028 s.
- Ten segmented sum/max/LogSumExp repeats over 65,536 edges: about 0.284 s.
- Peak allocated CUDA memory: 11.63 MiB.
- Selected backend: native `scatter_add_`/`scatter_reduce_`; dense fallback disabled.

### S4 full gate

- Assignment `100000→4096`: 200,000 sparse edges.
- Anchor selection: about 0.98 s; assignment construction: about 0.056 s.
- Assignment relative mass residual: `7.81e-8`.
- Assignment constant reconstruction error: `1.19e-7`.
- Batched SROT `8×4096→8×1024`: converged in 66 iterations, about 0.47 s.
- Batched SROT `8×1024→8×256`: converged in 66 iterations, about 0.32 s.
- Maximum hard-row relative residual: `2.38e-7`.
- Peak transport-only allocation/reservation: 15.44/34.0 MiB.
- The forbidden `[8,100000,16,64]` FP16 edge tensor would be 1,562.5 MiB before additional activations.

### S4 real-building gate

`COMMERCIALcastle_mesh0365` was evaluated through the complete measure pyramid:

- full assignment mass residual: exactly `0.0` in the recorded FP32 run;
- initial assignment adjoint error: `8.20e-8`;
- real `4096→1024` SROT: converged in 65 iterations, adjoint error `1.55e-6` in the final recorded GPU run;
- real `1024→256` SROT: converged in 57 iterations, adjoint error `1.68e-7`;
- transport-path peak allocation: 5.17 MiB.

Normals were pushed through each plan as second moments `nn^T`; principal axes
were used only with sign-invariant dot products.

The provisional ten-iteration budget was falsified and replaced by a maximum
of 100 iterations with early stopping.

## Normal-sign decision

A five-building pilot found negative local normal-dot fractions of 6.6–16.4%;
four buildings exceeded the 10% signed-safety threshold. S4 therefore defaults
to the coherent sign-invariant cost `1-|n_i^T n_j|`. Signed mode remains behind
an explicit audit-dependent flag.

## Verification commands

```powershell
python -m pytest -q
python freeze_camo_protocol.py --overwrite
python audit_sparse_backend.py --output sparse_backend_audit.json
python audit_normals.py --split train --limit 5 --output normal_orientation_audit_dry.json
python benchmark_transport.py --iterations 100 --output transport_benchmark_full.json
python validate_transport_real.py --split train --index 0 --iterations 100 --output transport_real_validation.json
python train_bridge_baseline.py --points 128 --batch-size 2 --epochs 1 --dry-run --skip-full-cloud --run-dir runs\bridge_dry_s1
python benchmark_camo.py --points 256 --batch-size 2 --width 32 --k 8 --steps 2 --warmup 1 --mass-mode density --lovasz-weight 0.5 --output benchmark_p15_dry.json
python verify_camo_protocol.py
```

## Next executable gate

Do not start S5 until:

1. corrected bridge controls finish for seeds 42, 123 and 2026;
2. `calibrate_s1_threshold.py` locks `delta_arch`;
3. the S2 objective decision is frozen as either B-CE or B-CE + Lovász;
4. a modern budget-matched control is reproduced or its incompatibility is documented.
