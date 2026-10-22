"""Tests for rasher_inspection.dataset, using a tiny fake dataset on disk."""

from pathlib import Path

import pytest

from rasher_inspection import dataset as ds

# --- helpers ---------------------------------------------------------------

def test_source_id_strips_roboflow_suffix():
    assert ds.source_id("x/IMG_01_jpg.rf.abc123.jpg") == "IMG_01_jpg"


def test_source_id_without_roboflow_pattern_uses_stem():
    assert ds.source_id("plain_name.png") == "plain_name"


def test_label_path_for_maps_images_to_labels():
    assert ds.label_path_for("r/train/images/a.jpg") == Path("r/train/labels/a.txt")


def test_load_class_names_accepts_dict_form(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n  1: b\n  0: a\n")
    assert ds.load_class_names(tmp_path) == ["a", "b"]


def test_load_class_names_missing_names_raises(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\n")
    with pytest.raises(ValueError):
        ds.load_class_names(tmp_path)


# --- read_labels -------------------------------------------------------------

def test_read_labels_parses_boxes(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("3 0.5 0.4 0.2 0.1\n\n1 0.1 0.1 0.05 0.05\n")
    boxes = ds.read_labels(f)
    assert [b.class_id for b in boxes] == [3, 1]
    assert boxes[0].y == pytest.approx(0.4)


def test_read_labels_empty_file_is_empty_list(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("")
    assert ds.read_labels(f) == []


@pytest.mark.parametrize("line", ["3 0.5 0.5 0.2", "3 0.5 0.5 0.2 1.5"])
def test_read_labels_rejects_malformed_lines(tmp_path, line):
    f = tmp_path / "l.txt"
    f.write_text(line)
    with pytest.raises(ValueError, match="l.txt:1"):
        ds.read_labels(f)


# --- audits ------------------------------------------------------------------

def test_split_summary_counts(fake_root):
    s = ds.split_summary(fake_root)
    assert s["train"]["images"] == 3
    assert s["train"]["sources"] == 2            # A's two copies count once
    assert s["train"]["empty_label"] == 1
    assert s["train"]["boxes"]["unsealed"] == 2
    assert s["train"]["boxes"]["packaging-error"] == 1
    assert s["valid"]["missing_label"] == 1
    assert s["test"]["boxes"]["wrinkle"] == 0    # zero classes still reported


def test_find_unlabelled_separates_empty_and_missing(fake_root):
    assert [p.name for p in ds.find_unlabelled(fake_root, "train")["empty_label"]] \
        == ["B_jpg.rf.333.jpg"]
    assert len(ds.find_unlabelled(fake_root, "valid")["missing_label"]) == 1


def test_find_cross_split_overlaps(fake_root):
    overlaps = ds.find_cross_split_overlaps(fake_root)
    assert len(overlaps) == 1
    assert overlaps[0]["source"] == "A_jpg"
    assert overlaps[0]["splits"] == ["train", "test"]
    assert len(overlaps[0]["images"]["train"]) == 2


# --- fixes -------------------------------------------------------------------

def test_quarantine_dry_run_moves_nothing(fake_root, tmp_path):
    overlaps = ds.find_cross_split_overlaps(fake_root)
    moved = ds.quarantine_overlaps(fake_root, overlaps, tmp_path / "q")
    assert len(moved) == 2
    assert len(ds.list_images(fake_root, "train")) == 3


def test_quarantine_moves_train_copies_and_logs(fake_root, tmp_path):
    overlaps = ds.find_cross_split_overlaps(fake_root)
    log = tmp_path / "log.csv"
    ds.quarantine_overlaps(fake_root, overlaps, tmp_path / "q",
                           log_csv=log, dry_run=False)
    assert [p.name for p in ds.list_images(fake_root, "train")] == ["B_jpg.rf.333.jpg"]
    assert len(ds.list_images(fake_root, "test")) == 2           # test untouched
    assert len(list((tmp_path / "q" / "train" / "labels").iterdir())) == 2
    assert ds.find_cross_split_overlaps(fake_root) == []
    assert len(log.read_text().strip().splitlines()) == 3       # header + 2 rows


def test_sample_for_review_is_reproducible():
    imgs = [Path(f"{i}.jpg") for i in range(50)]
    assert ds.sample_for_review(imgs, 10, seed=3) == ds.sample_for_review(imgs, 10, seed=3)
    assert len(ds.sample_for_review(imgs, 100)) == 50


def test_write_review_sheet(tmp_path):
    out = ds.write_review_sheet([Path("a.jpg"), Path("b.jpg")], tmp_path / "r.csv")
    lines = out.read_text().splitlines()
    assert lines[0] == "index,image,verdict,notes"
    assert lines[2].startswith("1,b.jpg")
