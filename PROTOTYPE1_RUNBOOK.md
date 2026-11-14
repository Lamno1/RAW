# CAMO-Build Prototype 1 — Agent Handoff Runbook

**Purpose:** Execute the frozen Prototype 1 without silently changing its mathematics.  
**Authoritative specification:** `PROTOTYPE1_MATHEMATICAL_REVIEW.md`  
**Hardware:** NVIDIA RTX 3050 Ti Laptop GPU, 4 GB VRAM  
**Implemented default:** \(k=16\), width 64, kernel range \([0.25,4.0]\), 4,096 sampled points, batch 8.

## 1. Files and responsibilities

- `camo_build/measure.py`: frozen 2-D adaptive KDE mass estimator.
- `camo_build/neighbors.py`: non-dense SciPy cKDTree graph construction.
- `camo_build/geometry.py`: gravity-preserving edge invariants.
- `camo_build/operator.py`: Local Measure-Aware Operator.
- `camo_build/model.py`: two-block Prototype 1 segmentation model.
- `camo_build/losses.py`: static weighted multiclass CE with explicit all-ignore failure.
- `prepare_camo_cache.py`: full-building normalization and point-aligned mass metadata.
- `camo_dataset.py`: sampling after full-building normalization; sparse crop kNN.
- `benchmark_camo.py`: independent debug/memory/throughput gates.
- `train_camo_p1.py`: gated training entry point.
- `tests/test_camo_prototype1.py`: mathematical and numerical behavior tests.

Do not modify `model.py`, `dataset.py`, or `train.py`; they remain the original baseline.

## 2. Immutable experiment rules

1. Never construct a dense \([N,N]\) distance or attention tensor for \(N>4000\).
2. Mass is cached from the full building and detached from autograd.
3. Full-building center and scale are computed before point sampling.
4. The local mass/kernel reductions run in FP32.
5. The three mass controls use identical data, model, loss, optimizer and seeds.
6. Validation selects checkpoints. Do not inspect the test set until the design is frozen after validation.
7. Any equation or kernel-bound change requires a versioned amendment.

## 3. Environment verification

From `D:\raw`:

```powershell
python -c "import torch, scipy; print(torch.__version__, torch.cuda.is_available(), scipy.__version__)"
python -m pytest tests\test_camo_prototype1.py -q
```

Expected: all seven tests pass. Stop immediately on any failure.

## 4. Build CAMO metadata

First run a small smoke test:

```powershell
python prepare_camo_cache.py --limit 2
```

Inspect the generated files under `camo_cache/train`, `camo_cache/val`, and `camo_cache/test`. Each must contain `center`, `scale`, `mass`, `k`, and `num_points`.

Then build all metadata:

```powershell
python prepare_camo_cache.py
```

This step uses CPU `scipy.spatial.cKDTree`, may be long, and is resumable because existing outputs are skipped. Do not use `--overwrite` unless the mathematical specification or source data changed.

After completion, verify counts:

```powershell
python -c "from pathlib import Path; print({s: len(list((Path('camo_cache')/s).glob('*.npz'))) for s in ('train','val','test')})"
```

Counts must match the corresponding split files.

## 5. Dataset smoke test

```powershell
python -c "from camo_dataset import CAMOBuildingNetDataset; d=CAMOBuildingNetDataset('.', 'train', 4096, 16); x=d[0]; print({k: (tuple(v.shape) if hasattr(v,'shape') else v) for k,v in x.items()})"
```

Required shapes:

- `features`: `[4096, 9]`
- `mass`: `[4096]`
- `neighbors`: `[4096, 16]`
- `label`: `[4096]`

Confirm `mass.mean()` is approximately one and no neighbor equals its own row index.

## 6. Hardware gates

Run one benchmark for every exact training configuration and mass mode:

```powershell
python benchmark_camo.py --mass-mode none    --output benchmark_camo_none.json
python benchmark_camo.py --mass-mode uniform --output benchmark_camo_uniform.json
python benchmark_camo.py --mass-mode density --output benchmark_camo_density.json
```

Acceptance requirements:

- debug gate completes with finite gradients;
- `max_memory_allocated_mib <= 3200`;
- no NaN or Inf;
- record allocated memory, reserved memory, points/s and seconds/step.

The initial density benchmark on the target machine produced approximately 1,370 MiB allocated, 1,474 MiB reserved and 423k points/s for the synthetic module-only workload. These are reference observations, not guaranteed thresholds for the data pipeline.

## 7. One-epoch smoke runs

Only after metadata and benchmarks pass:

```powershell
python train_camo_p1.py --benchmark-report benchmark_camo_none.json    --mass-mode none    --epochs 1 --seed 42
python train_camo_p1.py --benchmark-report benchmark_camo_uniform.json --mass-mode uniform --epochs 1 --seed 42
python train_camo_p1.py --benchmark-report benchmark_camo_density.json --mass-mode density --epochs 1 --seed 42
```

`none` and `uniform` are mathematically equivalent under the frozen normalized operator and act as an implementation sanity control. Their logits are unit-tested for equivalence with identical weights. Large training differences indicate nondeterminism or a bug, not a scientific effect.

## 8. Full attribution experiment

Recommended seed set: 42, 123, 2026. Run all modes with the same seed set:

```powershell
python train_camo_p1.py --benchmark-report benchmark_camo_none.json --mass-mode none --epochs 50 --seed 42
python train_camo_p1.py --benchmark-report benchmark_camo_none.json --mass-mode none --epochs 50 --seed 123
python train_camo_p1.py --benchmark-report benchmark_camo_none.json --mass-mode none --epochs 50 --seed 2026

python train_camo_p1.py --benchmark-report benchmark_camo_density.json --mass-mode density --epochs 50 --seed 42
python train_camo_p1.py --benchmark-report benchmark_camo_density.json --mass-mode density --epochs 50 --seed 123
python train_camo_p1.py --benchmark-report benchmark_camo_density.json --mass-mode density --epochs 50 --seed 2026
```

The `uniform` 50-epoch run is optional after its one-epoch equivalence check; it contains no distinct mathematical treatment from `none`.

## 9. Required analysis

For each run collect:

- best validation mIoU and epoch;
- validation accuracy;
- 31 per-class IoUs from the checkpoint;
- mean and standard deviation across seeds;
- head/mid/tail class aggregates using train-only counts;
- wall-clock epoch time;
- peak allocated/reserved VRAM;
- failure logs and NaN/Inf status.

Primary falsification question:

> Under an identical local architecture and training protocol, does cached density-derived mass improve validation mIoU or tail-class IoU over the unweighted/uniform operator across seeds?

Do not attribute a gain to CAMO mass weighting if it occurs in only one seed or only in overall accuracy.

## 10. Stop conditions

Stop and report rather than modify code if any condition occurs:

- metadata point counts do not match cached features;
- normals or vertical axes appear inconsistent;
- peak allocated VRAM exceeds 3.2 GB;
- all-ignore batches occur;
- mass has non-finite/non-positive values;
- density mode fails to beat controls consistently;
- CPU cKDTree construction dominates wall-clock time.

If CPU kNN is the bottleneck, the next authorized engineering action is to benchmark `torch-cluster`, PyTorch3D, or FAISS CUDA. Do not introduce dense `torch.cdist` as a shortcut.

## 11. Handoff report template

```markdown
# Prototype 1 execution report

- Commit/worktree state:
- GPU/driver/PyTorch:
- Metadata counts:
- Unit tests:
- Hardware benchmark per mode:
- Smoke-run result:
- Full runs completed:
- Mean ± std validation mIoU:
- Head/mid/tail IoU:
- Peak VRAM:
- Epoch wall time:
- Falsification decision: supported / rejected / inconclusive
- Blocking issue, if any:
- Exact artifact paths:
```

