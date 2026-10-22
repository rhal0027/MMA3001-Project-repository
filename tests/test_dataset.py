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


# --- content-based duplicate check -----------------------------------------

def _noise_image(path: Path, seed: int, brightness: float = 1.0) -> None:
    """Save a random-texture image; same seed = same tray, brightness = augmentation."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 200, size=(54, 72, 3)).astype(float) * brightness
    Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(path)


def test_dhash_survives_brightness_change_but_not_new_image(tmp_path):
    _noise_image(tmp_path / "a.png", seed=1)
    _noise_image(tmp_path / "a_bright.png", seed=1, brightness=1.25)
    _noise_image(tmp_path / "b.png", seed=2)
    a, a2, b = (ds.dhash(tmp_path / n) for n in ("a.png", "a_bright.png", "b.png"))
    assert ds.hamming(a, a2) <= 2
    assert ds.hamming(a, b) > 15


def test_find_near_duplicates_ignores_names_and_matches_content(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
    # Same tray, different names, brightened copy in test -> should be flagged.
    _noise_image(tmp_path / "train" / "images" / "X_jpg.rf.1.png", seed=10)
    _noise_image(tmp_path / "test" / "images" / "Y_jpg.rf.2.png", seed=10, brightness=1.2)
    # Different trays sharing a name -> should NOT be flagged.
    _noise_image(tmp_path / "train" / "images" / "Z_jpg.rf.3.png", seed=20)
    _noise_image(tmp_path / "valid" / "images" / "Z_jpg.rf.4.png", seed=30)

    matches = ds.find_near_duplicates(tmp_path, max_distance=6)
    assert len(matches) == 1
    assert matches[0]["query"].name == "Y_jpg.rf.2.png"
    assert matches[0]["match"].name == "X_jpg.rf.1.png"
    assert matches[0]["query_split"] == "test" and matches[0]["match_split"] == "train"


def test_hash_calibration_separates_known_copies(tmp_path):
    imgs = tmp_path / "train" / "images"
    imgs.mkdir(parents=True)
    for seed in (1, 2, 3):                      # three trays, two copies each
        _noise_image(imgs / f"T{seed}_jpg.rf.a.png", seed=seed)
        _noise_image(imgs / f"T{seed}_jpg.rf.b.png", seed=seed, brightness=1.2)
    cal = ds.hash_calibration(tmp_path, size=16)
    assert len(cal["same"]) == 3                # one pair per tray
    assert len(cal["different"]) == 6           # one nearest-other per image
    assert max(cal["same"]) < min(cal["different"])


def test_find_near_duplicates_accepts_larger_hash(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
    _noise_image(tmp_path / "train" / "images" / "X.png", seed=10)
    _noise_image(tmp_path / "test" / "images" / "Y.png", seed=10, brightness=1.2)
    m = ds.find_near_duplicates(tmp_path, max_distance=20, size=16)
    assert len(m) == 1


# --- video-frame neighbours and correlation ---------------------------------

def test_frame_number_parses_video_and_frame():
    assert ds.frame_number("a/frame_with_red_62_jpg.rf.x.jpg") == ("frame_with_red", 62)
    assert ds.frame_number("IMG_0123_jpg.rf.y.jpg") == ("IMG", 123)
    assert ds.frame_number("no_number.jpg") is None


def test_frame_gaps_finds_nearest_training_frame(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
    for f in (100, 110, 300):
        _noise_image(tmp_path / "train" / "images" / f"frame_with_red_{f}_jpg.rf.a.png", seed=f)
    _noise_image(tmp_path / "test" / "images" / "frame_with_red_108_jpg.rf.b.png", seed=1)
    _noise_image(tmp_path / "valid" / "images" / "frame_with_red_200_jpg.rf.c.png", seed=2)
    _noise_image(tmp_path / "valid" / "images" / "other_5_jpg.rf.d.png", seed=3)

    gaps = ds.frame_gaps(tmp_path)
    assert [(g["frame"], g["nearest_frame"], g["gap"]) for g in gaps] == \
        [(108, 110, 2), (200, 110, 90), (5, None, None)]   # sorted, unknown video last


def test_nearest_by_correlation_ignores_brightness(tmp_path):
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
    _noise_image(tmp_path / "train" / "images" / "A.png", seed=1)
    _noise_image(tmp_path / "train" / "images" / "B.png", seed=2)
    _noise_image(tmp_path / "test" / "images" / "A_copy.png", seed=1, brightness=1.3)
    _noise_image(tmp_path / "valid" / "images" / "C.png", seed=3)

    res = ds.nearest_by_correlation(tmp_path, size=(36, 27))
    assert res[0]["query"].name == "A_copy.png" and res[0]["match"].name == "A.png"
    assert res[0]["correlation"] > 0.95
    assert res[-1]["correlation"] < 0.5          # unrelated image
