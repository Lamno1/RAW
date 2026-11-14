"""S3 audit for native sparse reductions and CPU graph construction."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from camo_build.neighbors import scipy_knn_indices
from camo_build.sparse_ops import segment_logsumexp, segment_max, segment_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--segments", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("sparse_backend_audit.json"))
    args = parser.parse_args()
    if not 0 < args.k < args.points:
        parser.error("k must satisfy 0 < k < points")
    if min(args.edges, args.segments, args.channels, args.repeats) <= 0:
        parser.error("edge, segment, channel and repeat counts must be positive")

    generator = np.random.default_rng(2026)
    xyz = generator.normal(size=(args.points, 3)).astype(np.float32)
    started = time.perf_counter()
    graph = scipy_knn_indices(xyz, args.k)
    graph_seconds = time.perf_counter() - started
    if graph.shape != (args.points, args.k):
        raise RuntimeError("cKDTree returned an invalid graph")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_generator = torch.Generator(device=device).manual_seed(2026)
    values = torch.randn(
        args.edges, args.channels, generator=torch_generator, device=device,
        dtype=torch.float32,
    )
    segments = torch.arange(args.edges, device=device) % args.segments
    permutation = torch.randperm(args.edges, generator=torch_generator, device=device)
    segments = segments[permutation]

    # Small explicit CPU reference verifies the native backend independently.
    reference_values = values[: min(1024, args.edges)].cpu()
    reference_segments = segments[: reference_values.shape[0]].cpu()
    reference_sum = torch.zeros(args.segments, args.channels)
    reference_max = torch.full_like(reference_sum, -torch.inf)
    for edge in range(reference_values.shape[0]):
        segment = int(reference_segments[edge])
        reference_sum[segment] += reference_values[edge]
        reference_max[segment] = torch.maximum(
            reference_max[segment], reference_values[edge]
        )
    tested_sum = segment_sum(
        reference_values, reference_segments, args.segments
    )
    tested_max = segment_max(
        reference_values, reference_segments, args.segments
    )
    torch.testing.assert_close(tested_sum, reference_sum)
    torch.testing.assert_close(tested_max, reference_max)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(args.repeats):
        checksum += float(segment_sum(values, segments, args.segments).sum())
        checksum += float(segment_max(values, segments, args.segments).nan_to_num().sum())
        checksum += float(segment_logsumexp(values, segments, args.segments).nan_to_num().sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    reduction_seconds = time.perf_counter() - started
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "native_scatter_reduce": hasattr(torch.Tensor, "scatter_reduce_"),
        "optional_backends": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch_cluster", "torch_scatter", "pyg_lib", "pykeops")
        },
        "ckdtree": {
            "points": args.points, "k": args.k, "seconds": graph_seconds,
        },
        "segmented_reductions": {
            "edges": args.edges, "segments": args.segments,
            "channels": args.channels, "repeats": args.repeats,
            "seconds": reduction_seconds, "checksum": checksum,
        },
        "max_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "decision": "native_torch_edge_list_backend",
        "dense_fallback_allowed": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

