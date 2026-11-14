"""Numerical/adjoint smoke test for Triton sparse pushforward."""
from __future__ import annotations

import json

import torch

from camo_build.sparse_ops import segment_sum
from camo_build.triton_transport import (
    pushforward_triton,
    pushforward_triton_autograd,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260924)
    device = torch.device("cuda")
    n, m, k, d = 512, 128, 8, 64
    source = torch.arange(n, device=device, dtype=torch.long)[:, None].expand(n, k).reshape(-1)
    target = torch.randint(0, m, (n * k,), device=device, dtype=torch.long)
    target[:m] = torch.arange(m, device=device)
    values = torch.rand(n * k, device=device) + 0.01
    target_mass = segment_sum(values, target, m)
    feature = torch.randn(n, d, device=device)
    native = segment_sum(feature[source] * values[:, None], target, m) / target_mass[:, None]
    fast = pushforward_triton(feature, source, target, values, target_mass, num_targets=m)
    max_error = float((native - fast).abs().max())
    probe = torch.randn(m, d, device=device)
    source_req = feature.detach().clone().requires_grad_(True)
    output = pushforward_triton_autograd(
        source_req, source, target, values, target_mass, num_targets=m
    )
    (output * probe).sum().backward()
    native_req = feature.detach().clone().requires_grad_(True)
    native_output = segment_sum(
        native_req[source] * values[:, None], target, m
    ) / target_mass[:, None]
    (native_output * probe).sum().backward()
    grad_error = float((source_req.grad - native_req.grad).abs().max())
    report = {"max_forward_error": max_error, "max_adjoint_error": grad_error,
              "passed": max_error < 2e-5 and grad_error < 2e-5}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError(report)
    with open("triton_pushforward_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
