"""Shared pytest fixtures: a tiny fake Roboflow-style dataset on disk."""

from pathlib import Path

import pytest
from PIL import Image

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def _add(root: Path, split: str, name: str, label: str | None) -> Path:
    """Create a 10x10 image and (optionally) its label file."""
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    img = root / split / "images" / name
    Image.new("RGB", (10, 10)).save(img)
    if label is not None:
        (root / split / "labels" / f"{img.stem}.txt").write_text(label)
    return img


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Dataset with one overlap (A in train+test), one empty and one missing label."""
    (tmp_path / "data.yaml").write_text("names: " + str(NAMES) + "\n")
    _add(tmp_path, "train", "A_jpg.rf.111.jpg", "3 0.5 0.5 0.2 0.2\n")
    _add(tmp_path, "train", "A_jpg.rf.222.jpg", "3 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    _add(tmp_path, "train", "B_jpg.rf.333.jpg", "")          # annotated as clean
    _add(tmp_path, "valid", "C_jpg.rf.444.jpg", None)        # label file missing
    _add(tmp_path, "test", "A_jpg.rf.555.jpg", "0 0.5 0.5 0.4 0.4\n")
    _add(tmp_path, "test", "D_jpg.rf.666.jpg", "2 0.5 0.5 0.4 0.4\n")
    return tmp_path
