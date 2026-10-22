"""Tests for tray-level pass/fail evaluation."""

import math
from pathlib import Path

import pytest

from rasher_inspection import evaluate as ev
from rasher_inspection.dataset import list_images


def test_max_confidence_with_and_without_boxes():
    assert ev.ImagePrediction(Path("a"), [1, 3], [0.2, 0.7]).max_confidence == 0.7
    assert ev.ImagePrediction(Path("b"), [], []).max_confidence == 0.0


def test_load_missed_defects(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("index,image,verdict,notes\n"
                 "0,frame_with_red_4088_jpg.rf.aa.jpg,missed_defect,unsealed\n"
                 "1,frame_with_red_62_jpg.rf.bb.jpg,clean,\n")
    assert ev.load_missed_defects(f) == {"frame_with_red_4088_jpg"}


def test_tray_truth_uses_labels_and_corrections(fake_root):
    # fake_root test split: A (has a box) and D (has a box); train has B (empty)
    truth = ev.tray_truth(fake_root, "train", corrections={"B_jpg"})
    assert all(truth.values())                       # B corrected to defective
    truth = ev.tray_truth(fake_root, "train")
    assert sorted(v for v in truth.values()) == [False, True, True]


@pytest.fixture
def toy():
    """4 defective trays, 2 clean trays, with known max confidences."""
    truth = {Path(f"d{i}"): True for i in range(4)} | {Path(f"c{i}"): False for i in range(2)}
    conf = {Path("d0"): 0.9, Path("d1"): 0.6, Path("d2"): 0.3, Path("d3"): 0.0,
            Path("c0"): 0.4, Path("c1"): 0.1}
    return truth, conf


def test_tray_errors_counts(toy):
    truth, conf = toy
    e = ev.tray_errors(truth, conf, threshold=0.5)
    assert (e["tp"], e["fn"], e["fp"], e["tn"]) == (2, 2, 0, 2)
    assert e["false_pass_rate"] == 0.5 and e["false_fail_rate"] == 0.0
    lo, hi = e["false_pass_ci"]
    assert lo < 0.5 < hi


def test_tray_errors_no_clean_trays_gives_nan():
    e = ev.tray_errors({Path("d"): True}, {Path("d"): 0.9}, 0.5)
    assert math.isnan(e["false_fail_rate"])


def test_threshold_trade_off_is_monotonic(toy):
    truth, conf = toy
    sweep = ev.threshold_sweep(truth, conf)
    fp_rates = [s["false_pass_rate"] for s in sweep]
    ff_rates = [s["false_fail_rate"] for s in sweep]
    assert fp_rates == sorted(fp_rates)                  # higher threshold, more false passes
    assert ff_rates == sorted(ff_rates, reverse=True)    # and fewer false fails


def test_choose_threshold_prefers_no_false_passes(toy):
    truth, conf = toy
    sweep = ev.threshold_sweep(truth, conf, [0.0, 0.05, 0.2, 0.5])
    # d3 has confidence 0.0, so only threshold 0.0 catches every defect.
    best = ev.choose_threshold(sweep, max_false_pass=0.0)
    assert best["threshold"] == 0.0
    best = ev.choose_threshold(sweep, max_false_pass=0.25)
    assert best["threshold"] == 0.2 and best["false_fail_rate"] == 0.5
    assert ev.choose_threshold(sweep[-1:], max_false_pass=0.0) is None
