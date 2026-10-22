"""Tray-level pass/fail evaluation of a defect detector.

The detector outputs boxes with confidences. On a production line the
decision that matters is per tray: **fail** (reject or re-pack) if any defect
is detected, **pass** otherwise. This module turns per-image detections into
that decision and measures the two kinds of error:

* **false pass** - a defective tray is passed. This is the costly error: a
  faulty pack reaches a customer (food safety, shelf life, complaints).
* **false fail** - a good tray is rejected. This costs re-handling or waste.

Detection metrics (mAP, per-class precision/recall) come from Ultralytics'
own validator; this module covers what it does not.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import label_path_for, list_images, read_labels, source_id
from .stats import wilson_interval


@dataclass
class ImagePrediction:
    """Detections for one image.

    Attributes:
        image: Image path.
        classes: Predicted class id for each box.
        confidences: Confidence in [0, 1] for each box.
    """

    image: Path
    classes: list[int]
    confidences: list[float]

    @property
    def max_confidence(self) -> float:
        """Highest box confidence, or 0.0 if there are no boxes."""
        return max(self.confidences, default=0.0)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def load_missed_defects(review_csv: str | Path) -> set[str]:
    """Read the source ids marked ``missed_defect`` in a manual review sheet.

    Args:
        review_csv: CSV written by
            :func:`rasher_inspection.dataset.write_review_sheet` with the
            ``verdict`` column filled in.

    Returns:
        Source ids (see :func:`rasher_inspection.dataset.source_id`) of images
        that show a defect but have no box.
    """
    with open(review_csv, newline="") as f:
        return {source_id(row["image"]) for row in csv.DictReader(f)
                if row.get("verdict") == "missed_defect"}


def tray_truth(root: str | Path, split: str = "test",
               corrections: set[str] | None = None) -> dict[Path, bool]:
    """Ground-truth tray decision for every image in a split.

    An image is defective (``True``) if its label file has at least one box,
    or if its source id is in ``corrections`` (defects found by manual review
    that the original labels missed).

    Args:
        root: Dataset root folder.
        split: Split to read.
        corrections: Optional source ids to treat as defective.

    Returns:
        ``{image_path: is_defective}``.
    """
    corrections = corrections or set()
    truth = {}
    for img in list_images(root, split):
        lp = label_path_for(img)
        has_box = lp.exists() and bool(read_labels(lp))
        truth[img] = has_box or source_id(img) in corrections
    return truth


# --------------------------------------------------------------------------
# Decisions and error rates
# --------------------------------------------------------------------------


def tray_errors(truth: dict[Path, bool], max_conf: dict[Path, float],
                threshold: float) -> dict:
    """Pass/fail error rates at one confidence threshold.

    A tray fails if any detection has confidence >= ``threshold``.

    Args:
        truth: ``{image: is_defective}`` from :func:`tray_truth`.
        max_conf: ``{image: highest detection confidence}`` (0.0 if none).
            Must contain every image in ``truth``.
        threshold: Confidence threshold in [0, 1].

    Returns:
        Dict with counts (``tp`` defective & failed, ``fn`` defective &
        passed, ``fp`` clean & failed, ``tn`` clean & passed), the
        ``false_pass_rate`` = fn / defective and ``false_fail_rate`` =
        fp / clean, each with a 95% Wilson interval (``*_ci``). A rate is
        ``nan`` if its denominator is zero.

    Raises:
        KeyError: If an image in ``truth`` has no entry in ``max_conf``.
    """
    tp = fn = fp = tn = 0
    for img, defective in truth.items():
        fail = max_conf[img] >= threshold
        if defective:
            tp += fail
            fn += not fail
        else:
            fp += fail
            tn += not fail

    def rate(k, n):
        return (k / n, wilson_interval(k, n)) if n else (float("nan"), (float("nan"),) * 2)

    fpr, fpr_ci = rate(fn, tp + fn)
    ffr, ffr_ci = rate(fp, fp + tn)
    return {"threshold": threshold, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "false_pass_rate": fpr, "false_pass_ci": fpr_ci,
            "false_fail_rate": ffr, "false_fail_ci": ffr_ci}


def threshold_sweep(truth: dict[Path, bool], max_conf: dict[Path, float],
                    thresholds: np.ndarray | list[float] | None = None) -> list[dict]:
    """Evaluate :func:`tray_errors` over a range of thresholds.

    Lowering the threshold fails more trays: fewer false passes, more false
    fails. The sweep shows that trade-off so the operating point can be
    chosen deliberately.

    Args:
        truth: Ground truth from :func:`tray_truth`.
        max_conf: Highest confidence per image.
        thresholds: Thresholds to try. Defaults to 0.05 to 0.95 in steps of
            0.05.

    Returns:
        One :func:`tray_errors` dict per threshold, in the given order.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return [tray_errors(truth, max_conf, float(t)) for t in thresholds]


def choose_threshold(sweep: list[dict], max_false_pass: float = 0.0) -> dict | None:
    """Pick the operating point from a sweep.

    Among thresholds whose false-pass rate is at most ``max_false_pass``,
    choose the one with the lowest false-fail rate (ties go to the higher
    threshold). This encodes "do not let defective trays through, then waste
    as few good trays as possible".

    Choose the threshold on the **validation** split and report its errors
    on the **test** split, so the test result is not tuned to the test data.

    Args:
        sweep: Output of :func:`threshold_sweep`.
        max_false_pass: Highest acceptable false-pass rate.

    Returns:
        The chosen sweep entry, or ``None`` if no threshold meets the limit.
    """
    ok = [s for s in sweep if s["false_pass_rate"] <= max_false_pass]
    if not ok:
        return None
    return min(ok, key=lambda s: (s["false_fail_rate"], -s["threshold"]))
