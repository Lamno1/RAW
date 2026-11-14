"""Minimal Triton-Windows compile/runtime smoke test for S5 toolchain setup."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton smoke test")
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(100):
        _add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / 100.0
    report = {
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "elements": n,
        "mean_seconds": elapsed,
        "passed": True,
    }
    Path("triton_smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
