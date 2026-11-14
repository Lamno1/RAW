"""S0 protocol and metric tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from camo_build.metrics import (
    SegmentationMeter, boundary_mask, boundary_part_iou, full_cloud_chunks,
)
from camo_build.protocol import assert_external_temp, freeze_protocol, verify_protocol


def test_official_point_metrics_perfect_with_ignore() -> None:
    meter = SegmentationMeter(num_classes=3, ignore_index=-100)
    target = torch.tensor([[0, 1, 2, -100], [0, 0, 1, 2]])
    meter.update(target.clone(), target, model_names=["a", "b"])
    result = meter.compute()
    assert result["classification_accuracy"] == pytest.approx(1.0)
    assert result["part_iou"] == pytest.approx(1.0)
    assert result["shape_iou"] == pytest.approx(1.0)
    assert result["per_class_iou"] == pytest.approx([1.0, 1.0, 1.0])


def test_absent_class_is_nan_and_excluded_from_part_iou() -> None:
    meter = SegmentationMeter(num_classes=3)
    meter.update(torch.tensor([[0, 0, 1, 1]]), torch.tensor([[0, 1, 1, 1]]))
    result = meter.compute()
    assert np.isnan(result["per_class_iou"][2])
    expected_class0 = 1.0 / 2.0
    expected_class1 = 2.0 / 3.0
    assert result["part_iou"] == pytest.approx((expected_class0 + expected_class1) / 2)


def test_full_cloud_chunks_cover_every_point_once_per_vote() -> None:
    chunks = full_cloud_chunks(23, chunk_size=8, seed=17)
    observed = np.concatenate([chunk[:valid] for chunk, valid in chunks])
    assert sorted(observed.tolist()) == list(range(23))
    assert all(chunk.shape == (8,) for chunk, _ in chunks)


def test_boundary_mask_uses_valid_graph_neighbors() -> None:
    labels = torch.tensor([[0, 0, 1, 1, -100]])
    graph = torch.tensor([[[1], [2], [1], [2], [3]]])
    result = boundary_mask(labels, graph, ignore_index=-100)
    assert result.tolist() == [[False, True, True, False, False]]
    metric = boundary_part_iou(
        labels, labels, graph, num_classes=2, ignore_index=-100
    )
    assert metric["boundary_part_iou"] == pytest.approx(1.0)


def test_protocol_freeze_writes_reproducibility_artifacts(tmp_path) -> None:
    (tmp_path / "splits").mkdir()
    for split, names in {
        "train": ["a", "b"], "val": ["c"], "test": ["d"]
    }.items():
        (tmp_path / "splits" / f"{split}_split.txt").write_text(
            "\n".join(names), encoding="utf-8"
        )
    np.save(tmp_path / "class_counts.npy", np.arange(1, 32), allow_pickle=False)
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "protocol"
    protocol = freeze_protocol(tmp_path, output)
    assert protocol["label_zero_policy"] == "geometry_context_ignore_loss_metric"
    assert set(protocol["class_groups"]) == {"head", "mid", "tail"}
    for name in (
        "camo_pyramid_protocol.json", "source_manifest.sha256",
        "environment.txt", "split_manifest.json", "crop_manifest.json",
        "checkpoint_rule.json", "status.json",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_protocol(tmp_path, output)
    assert verify_protocol(tmp_path, output)["passed"] is True
    (tmp_path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    verification = verify_protocol(tmp_path, output)
    assert verification["passed"] is False
    assert verification["changed_sources"] == ["module.py"]


def test_protocol_manifest_excludes_pytest_scratch(tmp_path) -> None:
    (tmp_path / "splits").mkdir()
    for split, name in (("train", "a"), ("val", "b"), ("test", "c")):
        (tmp_path / "splits" / f"{split}_split.txt").write_text(name + "\n", encoding="utf-8")
    np.save(tmp_path / "class_counts.npy", np.arange(1, 32), allow_pickle=False)
    (tmp_path / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    scratch = tmp_path / "pytest_basetemp_s0" / ".pytest_cache"
    scratch.mkdir(parents=True)
    (scratch / "README.md").write_text("scratch\n", encoding="utf-8")
    output = tmp_path / "protocol"
    freeze_protocol(tmp_path, output)
    manifest = (output / "source_manifest.sha256").read_text(encoding="utf-8")
    assert "pytest_basetemp_s0" not in manifest
    assert ".pytest_cache" not in manifest


def test_external_basetemp_guard(tmp_path) -> None:
    with pytest.raises(ValueError, match="outside protocol root"):
        assert_external_temp(tmp_path, tmp_path / "pytest_basetemp_s0")
