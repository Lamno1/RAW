# Prototype 1 execution report

- **Commit/worktree state:** Standalone workspace `d:\raw` (frozen package `camo_build`, verified code)
- **GPU/driver/PyTorch:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.0 GB VRAM) | CUDA 12.4 | PyTorch 2.6.0+cu124 | SciPy 1.16.0
- **Metadata counts:** 1,849 / 1,849 buildings processed (train: 1,481, val: 187, test: 181). Verified exact match with `splits/`.
- **Unit tests:** 7/7 passed (`tests/test_camo_prototype1.py`).
- **Hardware benchmark per mode:**
  - `none`: max allocated = 1,237.6 MiB, max reserved = 1,474.0 MiB (< 3,200 MiB limit). Finite gradients.
  - `uniform`: max allocated = 1,237.6 MiB, max reserved = 1,474.0 MiB (< 3,200 MiB limit). Finite gradients.
  - `density`: max allocated = 1,369.7 MiB, max reserved = 1,474.0 MiB (< 3,200 MiB limit). Finite gradients.
- **Smoke-run result:**
  - `none` seed 42 (1 epoch): train_mIoU = 0.0248, val_mIoU = 0.0347
  - `uniform` seed 42 (1 epoch): train_mIoU = 0.0248, val_mIoU = 0.0347 (Mathematical equivalence confirmed)
  - `density` seed 42 (1 epoch): train_mIoU = 0.0249, val_mIoU = 0.0348
- **Full runs completed:** 6/6 runs (50 epochs each, 1,481 train buildings, 187 val buildings per epoch).

## Attribution Experiment Results (§8 & §9)

### Interpretation boundary

This is a paired ablation of `none` versus `density` under the CAMO Prototype 1 pipeline. It is **not** a direct comparison with the historical GeometryAwarePCT 1,024-point baseline, because the point count, normalization order, sampling schedule, local operator and training pipeline differ. The old 14.18% result remains a historical pilot reference, not a directly comparable control for this table.

### 1. Per-Run Summary

| Mode | Seed | Best Epoch | Val mIoU | Val Acc | Head mIoU | Mid mIoU | Tail mIoU | Peak Allocated VRAM |
|---|---|---|---|---|---|---|---|---|
| `none` | 42 | 43 | 0.12634 | 0.57989 | 0.32673 | 0.06155 | 0.00306 | 1,237.6 MiB |
| `none` | 123 | 43 | 0.12834 | 0.58059 | 0.33093 | 0.06291 | 0.00366 | 1,237.6 MiB |
| `none` | 2026 | 43 | 0.12594 | 0.57799 | 0.33084 | 0.05717 | 0.00219 | 1,237.6 MiB |
| **`none` (Mean ± Std)** | — | **43.0** | **0.12687 ± 0.00105** | **0.57949 ± 0.00110** | **0.32950 ± 0.00196** | **0.06054 ± 0.00246** | **0.00297 ± 0.00060** | **1,237.6 MiB** |
| `density` | 42 | 43 | 0.12674 | 0.58074 | 0.32795 | 0.06143 | 0.00318 | 1,369.7 MiB |
| `density` | 123 | 43 | 0.12899 | 0.58094 | 0.33319 | 0.06243 | 0.00387 | 1,369.7 MiB |
| `density` | 2026 | 43 | 0.12700 | 0.57879 | 0.33361 | 0.05740 | 0.00245 | 1,369.7 MiB |
| **`density` (Mean ± Std)** | — | **43.0** | **0.12758 ± 0.00101** | **0.58016 ± 0.00097** | **0.33158 ± 0.00257** | **0.06042 ± 0.00217** | **0.00317 ± 0.00058** | **1,369.7 MiB** |

### 2. Attribution Comparison (density vs none)

| Metric | Baseline (`none`) | CAMO (`density`) | Absolute Delta | Relative Change | Consistency across Seeds |
|---|---|---|---|---|---|
| **Validation mIoU** | 0.12687 ± 0.00105 | **0.12758 ± 0.00101** | **+0.00071** | **+0.56%** | **3/3 seeds higher (100%)** |
| **Validation Accuracy** | 0.57949 ± 0.00110 | **0.58016 ± 0.00097** | **+0.00067** | **+0.12%** | **3/3 seeds higher (100%)** |
| **Head Classes IoU (Top 10)** | 0.32950 ± 0.00196 | **0.33158 ± 0.00257** | **+0.00208** | **+0.63%** | **3/3 seeds higher (100%)** |
| **Tail Classes IoU (Bottom 11)** | 0.00297 ± 0.00060 | **0.00317 ± 0.00058** | **+0.00020** | **+6.73%** | **3/3 seeds higher (100%)** |
| **Mid Classes IoU (Middle 10)** | 0.06054 ± 0.00246 | 0.06042 ± 0.00217 | -0.00012 | -0.20% | 2/3 slightly lower |

- **Mean ± std validation mIoU:**
  - `none`: **0.1269 ± 0.0011**
  - `density`: **0.1276 ± 0.0010**
  - Paired differences by seed: **[0.00040, 0.00065, 0.00106]**.
  - Two-sided paired t-test over the three run pairs: **t = 3.656, p = 0.0673**. The original `analyze_prototype1.py` did not compute a p-value; therefore a claim of `p < 0.01` is not supported and has been removed.
- **Head/mid/tail IoU:**
  - Head: `none` 0.3295 vs `density` 0.3316 (**+0.0021**, gains in wall, roof, ground, floor)
  - Mid: `none` 0.0605 vs `density` 0.0604 (neutral)
  - Tail: `none` 0.00297 vs `density` 0.00317 (**+0.00020**, +6.7% relative gain)
- **Peak VRAM (independent synthetic hardware benchmark, not per-training-step telemetry):**
  - `none`: 1,237.6 MiB allocated / 1,474.0 MiB reserved.
  - `density`: 1,369.7 MiB allocated / 1,474.0 MiB reserved.
  - VRAM overhead of measure-aware operator: ~132.1 MiB (within 4 GB VRAM budget by >2.6 GB margin).
  - The current training CSVs do not record CUDA peak memory for every real training/validation step; these values must not be described as measured per-run training peaks.
- **Epoch wall time:** ~70 – 75 seconds per epoch (total ~60 – 72 minutes per 50-epoch run).
- **Falsification decision:** **DIRECTIONAL SUPPORT / INCONCLUSIVE STATISTICAL EVIDENCE**
  - Primary falsification question: *"Under an identical local architecture and training protocol, does cached density-derived mass improve validation mIoU or tail-class IoU over the unweighted/uniform operator across seeds?"*
  - **Verdict:** Cached density-derived mass weighting improved validation mIoU and Tail-class IoU versus the `none` arm on **100% of tested seeds (3/3 seeds)** under the paired protocol. Uniform mass was checked for one-epoch equivalence with `none`; it was not a separate 50-epoch comparison with density. This is directional evidence, but with only three paired runs the mIoU effect is not statistically significant at the conventional 0.05 level (**p = 0.0673**). The result should be treated as **promising but not conclusive** until more seeds or repeated crop evaluations are run.
- **Blocking issue, if any:** None. All 6 runs completed with return code 0, no NaN/Inf in the recorded epoch losses or metrics, zero non-positive masses, and zero all-ignore batches.
  **Post-hoc correction (2026-09-23):** the original runner did not audit raw gradients. A controlled replay found one FP16 gradient overflow at epoch 1 batch 72 for seed 42; `GradScaler` silently skipped that optimizer step and reduced its scale from 65,536 to 32,768. Therefore the earlier phrase “0 NaN/Inf values” must not be interpreted as proof that every raw gradient was finite.
- **Exact artifact paths:**
  - Benchmarks:
    - [benchmark_camo_none.json](file:///d:/raw/benchmark_camo_none.json)
    - [benchmark_camo_uniform.json](file:///d:/raw/benchmark_camo_uniform.json)
    - [benchmark_camo_density.json](file:///d:/raw/benchmark_camo_density.json)
  - Training logs (CSV):
    - [camo_p1_none_p4096_seed42.csv](file:///d:/raw/logs/camo_p1_none_p4096_seed42.csv)
    - [camo_p1_none_p4096_seed123.csv](file:///d:/raw/logs/camo_p1_none_p4096_seed123.csv)
    - [camo_p1_none_p4096_seed2026.csv](file:///d:/raw/logs/camo_p1_none_p4096_seed2026.csv)
    - [camo_p1_density_p4096_seed42.csv](file:///d:/raw/logs/camo_p1_density_p4096_seed42.csv)
    - [camo_p1_density_p4096_seed123.csv](file:///d:/raw/logs/camo_p1_density_p4096_seed123.csv)
    - [camo_p1_density_p4096_seed2026.csv](file:///d:/raw/logs/camo_p1_density_p4096_seed2026.csv)
  - Best Checkpoints (.pth):
    - [best_camo_p1_none_p4096_seed42.pth](file:///d:/raw/checkpoints/best_camo_p1_none_p4096_seed42.pth)
    - [best_camo_p1_none_p4096_seed123.pth](file:///d:/raw/checkpoints/best_camo_p1_none_p4096_seed123.pth)
    - [best_camo_p1_none_p4096_seed2026.pth](file:///d:/raw/checkpoints/best_camo_p1_none_p4096_seed2026.pth)
    - [best_camo_p1_density_p4096_seed42.pth](file:///d:/raw/checkpoints/best_camo_p1_density_p4096_seed42.pth)
    - [best_camo_p1_density_p4096_seed123.pth](file:///d:/raw/checkpoints/best_camo_p1_density_p4096_seed123.pth)
    - [best_camo_p1_density_p4096_seed2026.pth](file:///d:/raw/checkpoints/best_camo_p1_density_p4096_seed2026.pth)
  - Analysis Scripts:
    - [analyze_prototype1.py](file:///d:/raw/analyze_prototype1.py)
    - [run_experiments_p1.py](file:///d:/raw/run_experiments_p1.py)

## Overall Milestone Status & Next Step Decision (2026-09-21)

- Đã vượt qua giai đoạn ý tưởng trên giấy.
- Đã xác nhận pipeline và kiến trúc lõi (measure weighting + Local Measure-Aware Operator, không bảo toàn tích phân đặc trưng có trọng số mass) có thể chạy ổn định trên RTX 3050 Ti 4 GB, trong ngân sách VRAM (peak allocated 1,369.7 MiB, dưới giới hạn 3,200 MiB).
- Đã có bằng chứng sơ bộ rằng measure weighting (density mode) có ích: +0.00071 val mIoU (+0.56%), nhất quán trên 3/3 seed, nhưng **chưa có ý nghĩa thống kê** ở mức 0.05 (paired t-test: t = 3.656, p = 0.0673). Đây là bằng chứng xu hướng (directional), không phải kết luận.
- **Chưa** có kiến trúc CAMO-Build hoàn chỉnh: còn thiếu conservative OT coarsening, local–global manifold operator, uncertainty-routed detail refinement và hierarchical long-tail decision geometry (xem `FRONTIER_ARCHITECTURE.md` §5–§8).

**Kết luận:** đang ở mốc thiết kế và xác nhận lõi kiến trúc (Prototype 1), chưa bước vào giai đoạn kiến trúc hoàn chỉnh.

**Quyết định bước tiếp theo (cập nhật 2026-09-21):** trước khi triển khai Prototype 2 (OT coarsening), chạy **Prototype 1.5 — Lovász-Softmax loss ablation** (`FRONTIER_ARCHITECTURE.md` §13, config `L = L_B-CE + λ_L·L_Lovász`, cùng 4.096 điểm/seed/mass-mode với Prototype 1). Lý do: tác động trực tiếp vào mIoU/IoU, không rủi ro dependency/CUDA ABI, trả lời câu hỏi khoa học rõ ràng (LMAO+CE vs LMAO+CE+Lovász) trước khi thay đổi môi trường (PyKeOps/torch-cluster/Mamba).

Sau khi có kết quả Lovász ablation mới quyết định có cần cài kNN acceleration backend (pyg_lib/torch_scatter/torch_sparse — chỉ nếu profiling cho thấy `cKDTree` là bottleneck), rồi mới triển khai **Prototype 2 — OT coarsening** (sparse/unbalanced Sinkhorn transport theo §16.3–§16.4) để kiểm tra khả năng bảo toàn chi tiết (retained mass, Chamfer coverage, tail-point survival) khi giảm độ phân giải bằng transport thay vì FPS/random sampling. Gradient checkpointing và Mamba bị hoãn tới sau khi kiến trúc/loss ổn định.
