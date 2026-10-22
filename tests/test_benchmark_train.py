"""Tests for timing helpers and the YOLO data.yaml writer (no GPU needed)."""

import time

import pytest
import yaml

from rasher_inspection import benchmark as bm
from rasher_inspection.train import write_data_yaml


def test_time_per_item_measures_sleep():
    calls = []
    stats = bm.time_per_item(lambda x: (calls.append(x), time.sleep(0.002)),
                             [1, 2, 3], warmup=2, repeats=2)
    assert len(calls) == 2 + 3 * 2            # warm-up calls are not timed
    assert stats["n_items"] == 3 and stats["repeats"] == 2
    assert 1.5 < stats["median_ms"] < 50
    assert stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"]


def test_time_per_item_rejects_empty():
    with pytest.raises(ValueError):
        bm.time_per_item(lambda x: x, [])


def test_environment_has_keys():
    env = bm.environment()
    assert "python" in env and "gpu" in env


def test_write_data_yaml_uses_absolute_paths(fake_root):
    out = write_data_yaml(fake_root)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["path"] == str(fake_root.resolve())
    assert cfg["val"] == "valid/images" and cfg["test"] == "test/images"
    assert cfg["nc"] == 5 and cfg["names"][3] == "unsealed"


def test_write_data_yaml_missing_split(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [a]\n")
    with pytest.raises(FileNotFoundError):
        write_data_yaml(tmp_path)
