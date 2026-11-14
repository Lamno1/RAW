# CAMO-Pyramid Construction Blueprint

**Mode:** direct workspace mode; `D:\raw` is not currently a Git repository.  
**Architecture authority:** `CAMO_PYRAMID_ARCHITECTURE.md`  
**Frozen predecessor:** `PROTOTYPE1_MATHEMATICAL_REVIEW.md`  
**Target:** an auditable end-to-end CAMO-Pyramid implementation and paper-grade experiment package under 4 GB VRAM.

## 1. Dependency graph

```text
S0 Protocol freeze
 ├── S1 Corrected bridge baseline
 ├── S2 Optional Lovász objective ablation
 └── S3 Sparse reduction backend audit
          |
          v
S4 Full-cloud assignment + SROT mathematical units
          |
          v
S5 Detached SROT coarsening ablation
  requires S1, S3, S4 and a frozen objective decision from S2/default
          |
          v
S6 Transport-adjoint decoder
          |
          v
S7 Coarse global proxy block
          |
          v
S8 End-to-end CAMO-Pyramid
 ├── S9 Long-tail head ablations
 ├── S10 Pretraining/distillation control
 └── S11 Resolution/robustness studies
          | all retained branches
          v
S12 Frozen test evaluation and paper package
```

Safe parallelism:

- `S1`, `S2`, and the environment-only part of `S3` can proceed independently after `S0`.
- `S5` cannot be interpreted until `S1`, `S3`, and `S4` are complete and the base objective is frozen. `S2` may resolve to “retain B-CE.”
- `S9`, `S10`, and `S11` can proceed independently only after `S8` is frozen.
- Steps that modify the same training entry point must not run concurrently in direct workspace mode.

## 2. Global invariants

Every step must preserve:

1. no dense \(O(N^2)\) tensor at `N>4000`;
2. peak allocated CUDA memory `≤ 3200 MiB`;
3. full-building normalization before sampling;
4. train-only priors and class statistics;
5. test split remains untouched until `S12`;
6. FP32 for mass, OT duals, segmented reductions, marginal residuals, and loss sorting;
7. every result is tied to an exact configuration and seed;
8. no paper claim from a single seed;
9. equation-to-code traceability in module docstrings and tests.
10. every full-cloud point contributes through the initial assignment or is explicitly masked by the frozen protocol;
11. immutable run directories contain source snapshot hash, environment lock, config, split/crop manifest, seed, checkpoint rule, logs, and peak allocated/reserved VRAM.

## 3. Step S0 — Freeze the evaluation protocol

**Context brief:** Current sampled-cloud macro mIoU is useful internally but is not the complete official BuildingNet protocol. Architecture comparisons are invalid if metrics or sampling differ.

**Tasks:**

- Define sampled-cloud mIoU, full-cloud voting mIoU, BuildingNet pIoU/sIoU/class accuracy, head/mid/tail IoU, boundary IoU, latency, throughput, allocated/reserved VRAM.
- Define repeated-crop validation with fixed crop seeds.
- Freeze train/val/test split hashes and class grouping.
- Record whether ID 0 is context-only or excluded from candidate sampling.
- Reproduce the official evaluator on a small fixture or reference prediction, not only the historical project metric.
- Freeze the primary development metric, checkpoint-selection metric, and full-cloud voting schedule.
- Use at most eight single-seed exploratory configurations per component. Freeze one configuration before confirmatory paired runs.
- Pre-register `delta_floor=0.002` absolute mIoU and the rule `delta_arch=max(delta_floor, 2*sigma_pair)`. Estimate `sigma_pair` from S1 bridge controls, then append and lock the numerical `delta_arch` before S5 or any treatment result is inspected. This is a calibration edge from S1 to S5, not a reason to rerun S0. Freeze non-inferiority, tail-gain, latency, and memory rules in S0.
- Create a SHA-256 source snapshot and environment manifest because the workspace has no Git history.

**Verification:**

```powershell
python cache_and_audit.py
python -m pytest -q
```

**Exit criteria:** one metric module reproduces historical metrics and the official fixture; threshold rules, tuning budget, crop manifests, checkpoint rule, and artifact schema are frozen; unit tests cover absent classes and ignore labels. The numerical `delta_arch` is populated once after S1 and before S5 without inspecting treatment results.

**Rollback:** none; protocol errors must be corrected before further training.

## 4. Step S1 — Corrected bridge baseline

**Context brief:** The 14.18% GeometryAwarePCT pilot and CAMO P1 differ in point count, normalization, sampling, and architecture. A bridge baseline is needed for attribution.

**Tasks:**

- Run GeometryAwarePCT with full-building normalization and epoch-varying sampling.
- Evaluate at 1,024 and at most 4,000 points for the unchanged dense-attention model. A 4,096-point global PCT is prohibited by the `N>4000` dense-operation rule even if a one-off run appears to fit.
- Use the same objective and seeds as CAMO controls.
- Record official/full-cloud metrics separately from sampled metrics.
- Estimate `sigma_pair` from the frozen bridge-control design and write the resulting `delta_arch` to the append-only S0 calibration record before S5 starts.
- Add at least one modern point/serialization control and one sparse voxel or superpoint control when a reproducible 4 GB setup exists. Keep public official results separate from locally reproduced budget-matched results.

**Exit criteria:** at least three seeds; no changes other than pipeline corrections; checkpoints and logs complete.

**Rollback:** retain the historical pilot as a labeled historical reference if the at-most-4,000-point dense control fails hardware.

## 5. Step S2 — Prototype 1.5 Lovász ablation

**Context brief:** Tail IoU remains near zero and the current P1 mass effect is small. Lovász directly targets IoU but cannot restore points lost by sampling.

**Tasks:**

- Implement multiclass Lovász-Softmax with absent-class skipping and ignore-index handling.
- Compute sorting and reductions in FP32.
- Add gradient finite checks and all-ignore failure.
- Compare B-CE with B-CE + Lovász under identical density-mode runs.
- Freeze one loss weight or a documented calibration rule before multi-seed runs.

**Verification:**

```powershell
python -m pytest tests\test_lovasz.py -q
python benchmark_camo.py --mass-mode density --output benchmark_p15.json
```

**Exit criteria:** three paired seeds, gradient norms recorded, no NaN/Inf, ≤3.2 GB.

**Stop rule:** if tail and macro IoU do not improve consistently, retain B-CE for architecture work.

## 6. Step S3 — Sparse reduction backend audit

**Context brief:** SciPy cKDTree is valid but CPU graph construction may dominate. SROT needs segmented LogSumExp and scatter reductions; no backend should be installed without a measured need.

**Tasks:**

- Profile cKDTree time separately from GPU forward/backward.
- Audit compatible `pyg_lib`/scatter wheels for PyTorch 2.6 + CUDA 12.4.
- Build a minimal segmented `sum`, `max`, and `logsumexp` backend interface.
- Keep a CPU reference implementation for correctness tests.
- Benchmark the complete forward/backward/optimizer/validation path, CPU graph construction, host-to-device transfer, allocated and reserved CUDA memory; kernel-only timing is insufficient.

**Exit criteria:** GPU and reference reductions agree within FP32 tolerance; backend installation and version are recorded.

**Rollback:** use sorted-segment PyTorch operations on bounded edge lists if extension installation is unstable. Never fall back to dense matrices.

## 7. Step S4 — Full-cloud assignment and SROT mathematical units

**Context brief:** This step implements the two resolution-changing operators only. They must not be integrated into the segmentation network until their invariants pass.

**Tasks:**

- Implement the streamed `100k→4096` conditional assignment `A`, pushforward mass, and exact sparse pullback reference.
- Audit normal orientation. Use signed normal equations only if orientation is consistent; otherwise freeze the coherent sign-invariant variant before S5.
- Verify that every full-cloud point contributes and every working atom has positive mass; do not repair discarded spatial mass by scalar renormalization.
- Implement sparse SROT edge construction with forward and bounded reciprocal coverage; coalesce duplicate edges.
- Implement semi-relaxed entropic transport in log domain.
- Enforce hard source row mass and relaxed target capacity.
- Define positive target reference `nu` with the same total mass and generalized KL explicitly.
- Do not use a production dustbin. If an audit dustbin is enabled, retain and report its mass.
- Return `P`, output mass, absolute and relative row/mass residuals, target KL, entropy, objective change, iterations, and convergence flag.
- Keep transport plan detached from learnable features.

**Verification tests:**

- relative total-mass residual below the S0 threshold and scale-aware absolute residual reported;
- each source has at least one edge;
- no proxy has zero in-degree after bounded reciprocal coverage construction;
- initial assignment and each transport operator satisfy the weighted-adjoint identity on random test vectors;
- pullbacks reconstruct constant features within the S0 tolerance;
- finite behavior for duplicate points, isolated points, tiny masses, and extreme costs;
- permutation consistency;
- CPU reference agreement on small graphs;
- complexity test confirms no dense source-proxy matrix.

**Hardware gate:** benchmark streamed `100k→4096`, `4096→1024`, and `1024→256`; the GPU training path at batch 8 must allocate ≤3.2 GB. Include the mandatory `[8,100k,16,64]` edge-tensor stress calculation to demonstrate why no 100k LMAO is allowed.

**Exit criteria:** all mathematical and hardware gates pass before any segmentation training.

**Rollback:** weighted FPS/voxel partition becomes the coarsening control if SROT is unstable.

## 8. Step S5 — Detached SROT coarsening ablation

**Context brief:** The first scientific question is whether the transport partition is useful, not whether gradients can flow through Sinkhorn.

**Tasks:**

- Compare weighted FPS/voxel against geometry-only detached SROT.
- Add a same-support, same-cost row-softmax pooling control (`tau=0`). Without this control, any gain could come from ordinary soft pooling rather than the target-capacity penalty.
- Hold local blocks, proxy counts, decoder, loss, and seeds fixed.
- Measure Chamfer coverage, normal consistency, boundary retention, per-class point survival, proxy mass distribution, latency, and mIoU.
- Label-based survival metrics are evaluation-only and must not affect sampling.

**Exit criteria:** SROT must beat or clearly differentiate itself from both hard coarsening and same-support soft pooling: improve at least one pre-registered representation metric and remain within the frozen mIoU non-inferiority margin. A headline architecture path requires mean paired mIoU gain at least `delta_arch` in confirmatory runs.

**Go/no-go:** enable differentiable transport only if detached SROT shows a forward benefit.

## 9. Step S6 — Transport-adjoint decoder

**Context brief:** Reusing `P` for decoding is a central novelty and must be tested separately from the encoder partition.

**Tasks:**

- Implement the mathematical operator `U_ij=P_ij/m_i` on active positive masses.
- Use positive-mass masks and exact reciprocals on active entries; `clamp(min=eps)` is a guard whose activation fails the plan, not a perturbed definition of `U`.
- Test row sums, constant-feature reconstruction, and the weighted-adjoint inner-product identity.
- Compare against nearest-neighbor and inverse-distance interpolation.
- Add skip fusion without a new attention mechanism.
- Run a 2×2 factorial control: hard versus SROT coarsener crossed with conventional versus plan-reuse decoder.

**Exit criteria:** constant and adjoint errors remain below the S0 tolerances; boundary/tail IoU stays within the frozen non-inferiority margin; VRAM gate passes.

**Rollback:** retain nearest interpolation if pullback offers no benefit; do not hide failure with another decoder block.

## 10. Step S7 — Coarse global proxy block

**Context brief:** Global context is needed for architecturally related parts, but point-level global attention is unnecessary.

**Tasks:**

- Add one or two MHA blocks only at 256 proxies.
- Use gravity-preserving relative bias.
- Compare no-global, unweighted MHA, mass-aware MHA with `log(m_j)` key measure, and optional serialized SSM under equal proxy features.
- Record Pareto accuracy/VRAM/latency.

**Exit criteria:** the block meets the S0 effect criterion or improves a pre-registered structural-confusion metric while remaining within frozen mIoU non-inferiority, latency, and VRAM margins.

**Rollback:** remove global block if it only improves head accuracy.

## 11. Step S8 — End-to-end CAMO-Pyramid freeze

**Context brief:** This combines only components that independently passed gates.

**Tasks:**

- Assemble `4096→1024→256` encoder and adjoint decoder.
- Run gradient-conflict audit before adding auxiliary objectives.
- Benchmark forward/backward and full epoch.
- Run three paired seeds for the internal gate and at least five paired seeds for any headline architecture claim, unless S0 freezes a more efficient uncertainty design before results are inspected.

**Exit criteria:** complete immutable checkpoints/logs; ≤3.2 GB allocated with reserved memory reported; no NaN; repeated-crop and full-cloud evaluation; paired mean effect meets `delta_arch` with seed/crop uncertainty reported.

**Rollback:** remove the last-added component, not multiple components at once.

## 12. Step S9 — Long-tail head ablations

**Context brief:** Representation loss and decision bias are distinct problems.

**Tasks:**

- Test Lovász, logit adjustment, and cosine classifier one at a time.
- Freeze whether prior correction is used during training or inference and its sign convention before running it.
- Report calibration, confusion matrices, class recall, and head/mid/tail IoU.
- Only combine treatments whose gradients are compatible and whose isolated effects survive multiple seeds.

**Exit criteria:** tail improvement exceeds the absolute threshold frozen in S0 and neither mid nor head IoU crosses its frozen degradation margin.

## 13. Step S10 — Pretraining/distillation control

**Context brief:** BuildingNet is small, but external 3D pretraining may suffer domain mismatch.

**Tasks:**

- Evaluate a Sonata/PTv3 teacher as a control.
- Prefer offline cached teacher features to simultaneous teacher inference under 4 GB.
- Compare scratch, frozen teacher feature distillation, and fine-tuning where compatible.

**Exit criteria:** benefit survives at least three seeds and is separated from architecture contribution.

**Rollback:** remove external pretraining from the main claim if domain transfer is weak.

## 14. Step S11 — Resolution and robustness studies

**Tasks:**

- Evaluate 1k/4k and gated 8k settings.
- Run point-drop, density-shift, normal-noise, RGB-dropout, azimuth rotation, and repeated-crop tests.
- Plot accuracy-VRAM-latency Pareto curves.
- Measure transport stability across two discretizations of the same building.

**Exit criteria:** the claimed stability property is empirically demonstrated, not inferred from equations alone.

## 15. Step S12 — Frozen test and paper package

**Tasks:**

- Freeze architecture, objective, seeds, and checkpoint-selection rule.
- Confirm that every retained S9–S11 branch and every required modern baseline has completed or is explicitly abandoned with evidence.
- Evaluate test exactly once under the official protocol.
- Produce main table, ablations, per-class results, failure cases, complexity analysis, and reproducibility manifest.
- Distinguish observed results, inference, and theory assumptions.

**Exit criteria:** every paper claim maps to a table/figure/test and every table maps to immutable artifacts.

## 16. Stable execution state machine

Each experiment moves through `DRAFT → DRY_RUN → PILOT → CONFIRMATORY → FROZEN`.

- `DRAFT`: equations, one-axis hypothesis, control, and stop rule are written.
- `DRY_RUN`: unit tests, NaN gate, full-path VRAM gate, and one mini-epoch pass.
- `PILOT`: one seed within the fixed tuning budget; may select a configuration but cannot support a paper claim.
- `CONFIRMATORY`: frozen config and paired seeds/crop manifests; no hyperparameter changes.
- `FROZEN`: checksums, checkpoints, logs, summaries, and failure artifacts are immutable.

Every run directory must contain at least `config.yaml`, `source_manifest.sha256`, `environment.txt`, `split_manifest.json`, `crop_manifest.json`, `metrics.csv`, `hardware.json`, `checkpoint_rule.json`, and `status.json`. A failed run is retained with its terminal status and error log. Resumption restores model, optimizer, scheduler, scaler, RNG states, sampler epoch, and transport solver state.

## 17. Plan mutation protocol

Any mutation must be appended to this file with:

- date;
- affected step;
- evidence motivating the change;
- mathematical or implementation delta;
- invalidated artifacts;
- new verification command;
- approval status.

Steps may be skipped only when their hypothesis is explicitly abandoned. Failed experiments remain part of the audit trail.

### Mutation 2026-09-22 — S4 normal-sign policy

- **Evidence:** the first five train buildings had local negative normal-dot fractions from 6.6% to 16.4%; four of five exceeded the preregistered 10% signed-safety threshold.
- **Decision:** freeze sign-invariant normal geometry for S4/S5 (`|n_i^T n_j|`, `|n_z|`) unless a later complete audit establishes a consistently oriented subset.
- **Implementation delta:** `geometry_transport_cost(..., sign_invariant_normals=True)` is the default; signed mode requires an explicit flag and audit artifact.
- **Invalidated artifacts:** none; no S5 treatment has run.
- **Verification:** `python audit_normals.py --split train` and `python -m pytest tests\test_transport.py -q`.
- **Approval status:** automatic safety fallback already authorized by the frozen architecture contract.

### Mutation 2026-09-22 — S4 iteration budget

- **Evidence:** the full `[B=8]` gates at `4096→1024` and `1024→256` both required 66 iterations to reach a `1e-5` log-scaling tolerance with the frozen geometry cost; a 10-iteration budget was not converged.
- **Decision:** use `max_iterations=100` with early stopping for S4/S5. Iteration count and convergence remain mandatory diagnostics.

- **Hardware impact:** measured SROT times were approximately 0.47 s and 0.32 s for the two synthetic batched levels; peak allocated memory was 15.44 MiB for the transport-only gate.
- **Invalidated artifacts:** the earlier 10-iteration candidate and `transport_benchmark_dry.json` cannot be used as a mathematical pass.
- **Verification:** `python benchmark_transport.py --iterations 100 --output transport_benchmark_full.json`.
- **Approval status:** accepted because numerical convergence overrides the provisional iteration budget.

### Mutation 2026-09-23 — Explicit assignment coverage guard

- **Evidence:** independent S0–S4 review found that `build_full_cloud_assignment` relied on reciprocal coverage implicitly, without a final assertion.
- **Decision:** assert positive source and anchor coverage after edge construction, and reject non-finite/non-positive conditional weights before returning the assignment.
- **Scope:** defensive validation only; the assignment equations, sparse support, and mass semantics are unchanged.
- **Verification:** 25 protocol/unit tests pass; real 100k-point assignment has zero relative mass residual and the SROT levels converge with row residual below `2.1e-7`.
- **Status:** accepted; no S5 run is authorized by this mutation.

### Mutation 2026-09-23 — S2 artifact isolation and gradient audit

- **Evidence:** the initial S2 runner reused density-run filenames for B-CE and B-CE+Lovász, so a paired treatment could overwrite its control; epoch logs also omitted gradient norms required by the S2 exit criteria.
- **Decision:** encode the objective and Lovász weight in every S2 log/checkpoint name; unscale, finite-check, clip, and record the mean pre-clipping gradient norm for every training epoch.
- **Scope:** both paired arms use the same revised optimization path; no model, data split, seed, loss coefficient, or metric definition changes after the confirmatory runs begin.
- **Invalidated artifacts:** the earlier Lovász dry-run remains a hardware smoke test only and is not confirmatory evidence.
- **Status:** accepted before launching S1/S2 confirmatory training.

### Mutation 2026-09-23 — AMP scale stabilization after strict-gradient replay

- **Evidence:** both the frozen `StaticBalancedCrossEntropy` and an independent weighted `F.cross_entropy` reference overflowed only `head.4.weight` at seed 42, epoch 1, batch 72 when the initial AMP scale was 65,536. The legacy runner silently skipped that step and reduced the scale to 32,768, then completed the epoch.
- **Decision:** set and record `amp_init_scale=32768` for both paired S2 arms and their hardware benchmarks; retain the hard non-finite-gradient failure gate.
- **Attribution:** the loss implementations agreed within `1.6e-5` and failed at the identical location, so this mutation addresses AMP scaling rather than changing the loss equation.
- **Invalidated artifacts:** the failed S2 seed-42 attempt and all earlier Lovász smoke benchmarks with an unrecorded/default scale are diagnostic only.
- **Status:** authorized after both objectives completed an independent full-epoch smoke test with finite gradients at scale 32,768.

### Mutation 2026-09-23 — Replace FP16 loss scaling with BF16 autocast for S2

- **Evidence:** FP16 scale 32,768 passed one epoch but overflowed near the end of epoch 6. Mean raw gradient norm increased from 1.20 to 2.62 over the five completed epochs; an individual gradient above approximately 1.999 is sufficient to overflow after scaling even though the unscaled norm is not pathological.
- **Decision:** use BF16 autocast without gradient scaling for both S2 arms. The RTX 3050 Ti reports native BF16 support; BF16 removes the FP16 exponent-range bottleneck while retaining mixed-precision activation memory savings.
- **Controls:** retain `error_if_nonfinite=True`, identical optimizer/clipping/schedule for both arms, exact benchmark matching, and one full-epoch smoke per objective before confirmatory execution.
- **Invalidated artifacts:** all FP16 S2 runs and smoke tests are numerical diagnostics only; none are paired treatment evidence.
- **Status:** authorized after both BF16 benchmarks and objective-specific full-epoch smoke gates passed with finite raw gradients.

### Mutation 2026-09-24 — SROT v2 absolute hardware budget

- **Affected step:** S5 SROT only; the row-softmax confirmatory control is independent and unchanged.
- **Evidence:** v1 reference SROT failed the frozen relative latency gate at 2.4370× and the opt-in Triton backend failed at 3.2829× after eager, warm-start, reduction, and fused-kernel studies. The v1 1.25× result and all negative artifacts remain immutable.
- **Mathematical/implementation delta:** none to the transport equation. V2 replaces the *prospective* eligibility question with an independently grounded absolute systems budget: at least 50,000 source points/s for the complete model step, at most 3,200 MiB allocated VRAM, and at most 216 projected GPU-hours for three 50-epoch seeds. The 1.25× ratio remains a reported v1 result and a v2 diagnostic, not a v2 hard gate.
- **Invalidated artifacts:** none. V1 results cannot be reused as a v2 pass and cannot be rewritten.
- **Verification:** `python benchmark_srot_v2_gate.py --output runs/srot_v2_gate/benchmark.json`, followed by `python -m pytest -q --basetemp=<external-temp>` and direct `verify_protocol()`; no SROT segmentation training is authorized unless all v2 numerical and hardware gates pass.
- **Approval status:** user authorized the parallel v2 branch before any v2 training metric; full preregistration is `protocol/v2/srot_latency_mutation.json`.

## 18. Anti-pattern catalog

- Claiming PyKeOps changes dense Sinkhorn time from quadratic to linear.
- Installing Mamba before a coarse global-context baseline exists.
- Comparing 1,024-point PCT with 4,096-point CAMO as an architecture-only gain.
- Using ground-truth class labels to allocate inference proxies.
- Reporting relative tail gains without absolute IoU.
- Selecting a loss weight separately for every seed.
- Evaluating the test set while architecture decisions remain open.
- Calling a row-normalized kernel feature-mass-conservative.
- Adding losses to rescue an unstable transport implementation.
