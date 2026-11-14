"""Direct paired transport timing at the frozen S5 topology."""
from __future__ import annotations

import json
import statistics
import time

import torch

from camo_build.transport import semi_relaxed_sparse_transport
from camo_build.triton_transport import semi_relaxed_sparse_transport_triton


def time_call(fn, repeats: int = 3) -> list[float]:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    torch.manual_seed(20260924)
    device = torch.device("cuda")
    n, m, k = 8 * 4096, 8 * 1024, 8
    source = torch.arange(n, device=device, dtype=torch.long)[:, None].expand(n, k).reshape(-1)
    target = torch.randint(0, m, (n * k,), device=device, dtype=torch.long)
    target[:m] = torch.arange(m, device=device)
    mass = torch.rand(n, device=device, dtype=torch.float32) + 0.05
    cost = torch.rand(n * k, device=device, dtype=torch.float32)
    kwargs = dict(num_targets=m, epsilon=0.2, tau=1.0, max_iterations=100, tolerance=1e-5)
    ref = time_call(lambda: semi_relaxed_sparse_transport(mass, source, target, cost, **kwargs))
    tri = time_call(lambda: semi_relaxed_sparse_transport_triton(mass, source, target, cost, **kwargs))
    hybrid = time_call(lambda: semi_relaxed_sparse_transport_triton(mass, source, target, cost, target_backend="native", **kwargs))
    ref_plan, _ = semi_relaxed_sparse_transport(mass, source, target, cost, **kwargs)
    tri_plan, tri_target, tri_iter, tri_conv, tri_change = semi_relaxed_sparse_transport_triton(mass, source, target, cost, **kwargs)
    report = {
        "n": n, "m": m, "k": k,
        "reference_seconds": ref,
        "triton_seconds": tri,
        "hybrid_seconds": hybrid,
        "median_ratio": statistics.median(tri) / statistics.median(ref),
        "hybrid_ratio": statistics.median(hybrid) / statistics.median(ref),
        "max_abs_plan_error": float((tri_plan - ref_plan.value).abs().max()),
        "mass_residual": float((tri_target.sum() - mass.sum()).abs()),
        "iterations": tri_iter, "converged": tri_conv, "final_change": tri_change,
    }
    print(json.dumps(report, indent=2))
    with open("triton_srot_realistic_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
