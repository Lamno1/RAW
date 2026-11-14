# BuildingNet: legacy corrected research plan

> **Status (2026-09-21): historical bridge plan, not the current execution authority.**
> The current architecture is defined in [`CAMO_PYRAMID_ARCHITECTURE.md`](CAMO_PYRAMID_ARCHITECTURE.md), and the only authoritative execution order is [`plans/CAMO_PYRAMID_BLUEPRINT.md`](plans/CAMO_PYRAMID_BLUEPRINT.md). The 1,024-point commands below remain only for reproducing the historical baseline and must not be used to attribute CAMO-Pyramid gains.

## Hypothesis

Geometry-aware point features plus efficient attention can improve rare-part IoU
for **point-wise architectural-part segmentation** under a 4 GB VRAM budget.
The earlier filename-derived building classification is only a pilot and cannot
support claims about component segmentation.

## Architecture

```text
100k aligned points + point labels
              |
      sample 1,024 initially
              |
       XYZ | normal | RGB
              |
       shared embedding -------- geometry branch (phase 2)
              |                  height/curvature/kNN offsets
       offset attention ------------------|
              |                           |
              +------- gated fusion ------+
                          |
                  global context
                          |
             point-wise 31-class head
                          |
                 per-class IoU / mIoU
```

## Experimental gates

1. Audit point-label alignment, label range, missing files, split overlap and
   train-only class frequencies.
2. Reproduce XYZ-only PointNet-like and corrected PCT baselines using three seeds.
   Validation selects checkpoints; test is evaluated only after design freezes.
3. Input ablation: XYZ; XYZ+normal; XYZ+normal+RGB.
4. Loss ablation: CE, weighted CE, focal, class-balanced focal. Focal by itself
   is not class balancing.
5. Proposed architecture: local kNN attention instead of global O(N^2), explicit
   geometric descriptors and gated multi-scale fusion.
6. Report parameters, peak VRAM, latency, throughput and mIoU at 1,024/2,048/4,096
   points, plus mean/std, per-class IoU and failure cases.

## Commands

```powershell
python dataset.py
python model.py
python train.py --points 1024 --batch-size 8 --epochs 50
# Weighted-loss ablation after the complete audit:
python train.py --points 1024 --batch-size 8 --epochs 50 --loss weighted_ce
```

The old classifier checkpoints remain untouched. New training writes
`checkpoints/best_segmentation_model.pth`.

Official raw labels use IDs 1--31; ID 0 means `undetermined` and is ignored by
loss and metrics. The model output indices 0--30 therefore map to raw IDs 1--31.
Run `python cache_and_audit.py` once to create the resumable `.npz` cache and
train-only `class_counts.npy`, `class_weights.npy`, and `class_audit.json`.
