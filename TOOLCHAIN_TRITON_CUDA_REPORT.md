# Triton/CUDA toolchain report — 2026-09-24

The requested compiler path is installed and verified without changing the
existing NVIDIA display driver.

| Component | Verified value |
|---|---|
| PyTorch | `2.6.0+cu124` |
| CUDA runtime | `12.4` |
| GPU | RTX 3050 Ti Laptop GPU, compute capability `8.6` |
| Triton | `triton-windows 3.2.0.post21` (imports as Triton `3.2.0`) |
| MSVC | Visual Studio Build Tools 2022, compiler `19.44.35229` |
| CUDA compiler | official CUDA `cuda_nvcc` redist `12.4.131`, `nvcc --version` verified |
| Triton compiler support | bundled `ptxas.exe` and TinyCC verified |

`benchmark_triton_smoke.py` passes a real GPU Triton kernel. The SROT smoke
test has maximum plan error `1.2e-7` and zero mass residual; the pushforward
test has forward error `2.4e-7` and adjoint error `8.9e-8`.

The fused backend is intentionally opt-in (`transport_backend="triton"`). The
final full paired S5 run remains negative at `3.28287x` versus the frozen `1.25x`
limit, so no production/default or protocol gate was changed.
