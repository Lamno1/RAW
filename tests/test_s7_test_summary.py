"""Pure summary tests; never instantiate or read the test dataset."""
from __future__ import annotations

import pytest

from evaluate_camo_s7_test_once import summarize_reports


def test_summarize_reports_is_descriptive_and_seed_aligned() -> None:
    reports = []
    for seed, value in zip((42, 123, 2026), (0.1, 0.2, 0.3), strict=True):
        reports.append({
            "seed": seed,
            "metrics": {
                "part_iou": value,
                "classification_accuracy": value + 0.4,
                "group_iou": {"head": value + 0.1, "mid": value + 0.2, "tail": value + 0.3},
                "per_class_iou": [value, None] + [value] * 29,
            },
        })

    result = summarize_reports(reports)
    assert result["part_iou"]["values"] == [0.1, 0.2, 0.3]
    assert result["part_iou"]["mean"] == pytest.approx(0.2)
    assert result["part_iou"]["sample_sd"] == pytest.approx(0.1)
    assert result["per_class_iou"][1]["defined_seed_count"] == 0
    assert result["per_class_iou"][1]["mean_over_defined_seed_metrics"] is None


def test_summarize_reports_rejects_missing_seed_and_nonfinite_metric() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        summarize_reports([])
    reports = [{"metrics": {"part_iou": float("nan")}}] * 3
    with pytest.raises(FloatingPointError, match="non-finite"):
        summarize_reports(reports)
