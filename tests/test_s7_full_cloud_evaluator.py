"""Regression tests for strict full-cloud evaluation artifacts."""
from __future__ import annotations

import json

import numpy as np

from evaluate_camo_s7_full_cloud import atomic_json


def test_atomic_json_serializes_undefined_metrics_as_null(tmp_path) -> None:
    output = tmp_path / "metrics.json"
    atomic_json(
        output,
        {
            "part_iou": 0.25,
            "per_class_iou": [0.5, float("nan"), np.float64("inf")],
            "group_iou": {"head": 0.4, "tail": float("nan")},
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "part_iou": 0.25,
        "per_class_iou": [0.5, None, None],
        "group_iou": {"head": 0.4, "tail": None},
    }
