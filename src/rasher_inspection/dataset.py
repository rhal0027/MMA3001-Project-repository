"""Dataset loading and auditing for the pork-rasher packaging dataset.

The dataset is exported from Roboflow in YOLO format::

    <root>/
        data.yaml               # class names
        train/images/*.jpg      train/labels/*.txt
        valid/images/*.jpg      valid/labels/*.txt
        test/images/*.jpg       test/labels/*.txt

Each label file has one line per object::

    <class_id> <x_centre> <y_centre> <width> <height>

with coordinates normalised to [0, 1] relative to the image size. An empty
label file means the image was annotated as containing no defects.

Roboflow names every exported image ``<original>_<ext>.rf.<hash>.<ext>``, and
augmented copies of one photo share the same ``<original>`` prefix. This module
uses that prefix (the *source id*) to find photos that appear in more than one
split, which would leak test information into training.
"""

from __future__ import annotations

import collections
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS: tuple[str, ...] = ("train", "valid", "test")
"""Split folder names used by the Roboflow YOLO export."""

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
"""File extensions treated as images."""


@dataclass(frozen=True)
class Box:
    """One labelled object from a YOLO label file.

    Attributes:
        class_id: Integer index into the class-name list in ``data.yaml``.
        x: Box centre x, normalised to [0, 1] of image width.
        y: Box centre y, normalised to [0, 1] of image height.
        w: Box width, normalised to [0, 1] of image width.
        h: Box height, normalised to [0, 1] of image height.
    """

    class_id: int
    x: float
    y: float
    w: float
    h: float


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_dataset(api_key: str, location: str | Path = "data/raw/pork-rasher",
                     version: int = 4) -> Path:
    """Download the Roboflow pork-rasher dataset in YOLO format.

    Args:
        api_key: Roboflow *private* API key. Read it from Colab Secrets or an
            environment variable. Never hard-code it.
        location: Folder to download into. Keep it under ``data/raw/`` so it is
            git-ignored.
        version: Roboflow dataset version. Version 4 is the augmented release
            described in the project brief.

    Returns:
        Path to the downloaded dataset root (the folder containing ``data.yaml``).

    Raises:
        ImportError: If the ``roboflow`` package is not installed.
    """
    from roboflow import Roboflow  # imported here so the rest of the module works without it

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("hello-2aqe0").project("pork-rasher-error-packaging")
    dataset = project.version(version).download("yolov8", location=str(location))
    return Path(dataset.location)


# --------------------------------------------------------------------------
# Basic file helpers
# --------------------------------------------------------------------------


def load_class_names(root: str | Path) -> list[str]:
    """Read the class names from ``<root>/data.yaml``.

    Args:
        root: Dataset root folder.

    Returns:
        Class names, indexed by class id.

    Raises:
        FileNotFoundError: If ``data.yaml`` does not exist.
        ValueError: If ``data.yaml`` has no ``names`` entry.
    """
    cfg = yaml.safe_load((Path(root) / "data.yaml").read_text())
    names = cfg.get("names") if cfg else None
    if names is None:
        raise ValueError(f"No 'names' entry in {Path(root) / 'data.yaml'}")
    # YOLO allows names as a list or as a {id: name} dict.
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def list_images(root: str | Path, split: str) -> list[Path]:
    """List the image files in one split, sorted by name.

    Args:
        root: Dataset root folder.
        split: One of ``"train"``, ``"valid"`` or ``"test"``.

    Returns:
        Sorted image paths. Empty if the split folder does not exist.
    """
    folder = Path(root) / split / "images"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for(image_path: str | Path) -> Path:
    """Return the label file that belongs to an image.

    ``<split>/images/abc.jpg`` maps to ``<split>/labels/abc.txt``.

    Args:
        image_path: Path to an image inside a ``<split>/images`` folder.

    Returns:
        Expected path of the matching label file (it may not exist).
    """
    p = Path(image_path)
    return p.parent.parent / "labels" / f"{p.stem}.txt"


def source_id(image_path: str | Path) -> str:
    """Return the original-photo id for a Roboflow export file name.

    Roboflow renames ``IMG_0123.jpg`` to ``IMG_0123_jpg.rf.<hash>.jpg``. All
    augmented copies of the same photo share the part before ``.rf.``.

    Args:
        image_path: Image path or file name.

    Returns:
        The part of the file name before ``.rf.``, or the file stem if the name
        does not follow the Roboflow pattern.

    Example:
        >>> source_id("train/images/IMG_0123_jpg.rf.9f8e7d.jpg")
        'IMG_0123_jpg'
    """
    name = Path(image_path).name
    return name.split(".rf.")[0] if ".rf." in name else Path(name).stem


def read_labels(label_path: str | Path) -> list[Box]:
    """Parse a YOLO label file.

    Args:
        label_path: Path to a ``.txt`` label file.

    Returns:
        The boxes in the file. An empty file gives an empty list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line does not have five fields, or a coordinate is
            outside [0, 1]. The message gives the file and line number.
    """
    boxes = []
    for n, line in enumerate(Path(label_path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{n}: expected 5 fields, got {len(parts)}")
        cls, *coords = parts
        x, y, w, h = (float(c) for c in coords)
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            raise ValueError(f"{label_path}:{n}: coordinates must be in [0, 1]")
        boxes.append(Box(int(cls), x, y, w, h))
    return boxes


# --------------------------------------------------------------------------
# Audits
# --------------------------------------------------------------------------


def split_summary(root: str | Path) -> dict[str, dict]:
    """Count images, unlabelled images and boxes per class for every split.

    Args:
        root: Dataset root folder.

    Returns:
        ``{split: {"images": int, "sources": int, "empty_label": int,
        "missing_label": int, "boxes": {class_name: int}}}``. ``sources`` is
        the number of distinct original photos (augmented copies counted once).
    """
    names = load_class_names(root)
    summary = {}
    for split in SPLITS:
        images = list_images(root, split)
        boxes: collections.Counter[str] = collections.Counter()
        empty = missing = 0
        for img in images:
            lp = label_path_for(img)
            if not lp.exists():
                missing += 1
                continue
            labels = read_labels(lp)
            if not labels:
                empty += 1
            for b in labels:
                boxes[names[b.class_id]] += 1
        summary[split] = {
            "images": len(images),
            "sources": len({source_id(p) for p in images}),
            "empty_label": empty,
            "missing_label": missing,
            "boxes": {name: boxes.get(name, 0) for name in names},
        }
    return summary


def find_unlabelled(root: str | Path, split: str) -> dict[str, list[Path]]:
    """Find images in a split that have no defect boxes.

    Args:
        root: Dataset root folder.
        split: Split to search.

    Returns:
        ``{"empty_label": [...], "missing_label": [...]}``. Empty label files
        are images annotated as defect-free; missing label files usually point
        to an export problem.
    """
    out: dict[str, list[Path]] = {"empty_label": [], "missing_label": []}
    for img in list_images(root, split):
        lp = label_path_for(img)
        if not lp.exists():
            out["missing_label"].append(img)
        elif not read_labels(lp):
            out["empty_label"].append(img)
    return out


def find_cross_split_overlaps(root: str | Path) -> list[dict]:
    """Find original photos that appear in more than one split.

    Overlaps are matched by :func:`source_id`, so two *different* photos that
    happen to share a file name are also reported. Check each pair visually
    (see :func:`rasher_inspection.visualise.show_pairs`) before removing it.

    In the version 4 export, all 13 file-name overlaps were confirmed by eye to
    be different trays, so file names alone are not a reliable leakage check.
    Use :func:`find_near_duplicates` to compare image content instead.

    Args:
        root: Dataset root folder.

    Returns:
        One dict per overlapping source: ``{"source": str, "splits": [str],
        "images": {split: [Path]}}``, sorted by source id.
    """
    by_split: dict[str, dict[str, list[Path]]] = {}
    for split in SPLITS:
        groups: dict[str, list[Path]] = collections.defaultdict(list)
        for img in list_images(root, split):
            groups[source_id(img)].append(img)
        by_split[split] = groups

    all_sources = set().union(*(g.keys() for g in by_split.values()))
    overlaps = []
    for src in sorted(all_sources):
        present = [s for s in SPLITS if src in by_split[s]]
        if len(present) > 1:
            overlaps.append({
                "source": src,
                "splits": present,
                "images": {s: by_split[s][src] for s in present},
            })
    return overlaps


def dhash(image_path: str | Path, size: int = 8) -> int:
    """Compute a difference hash (dHash) fingerprint of an image.

    The image is converted to greyscale and shrunk to ``(size + 1) x size``
    pixels. Each bit records whether a pixel is brighter than its right-hand
    neighbour, giving ``size * size`` bits. Because only the *direction* of
    brightness change is kept, the hash barely changes under the exposure and
    saturation augmentations used in this dataset, but differs for different
    trays.

    Args:
        image_path: Image file.
        size: Hash grid size; 8 gives a 64-bit hash.

    Returns:
        The hash as a Python integer.
    """
    grey = Image.open(image_path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(grey, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes.

    Example:
        >>> hamming(0b1011, 0b0010)
        2
    """
    return (a ^ b).bit_count()


def hash_calibration(root: str | Path, split: str = "train", size: int = 8,
                     max_images: int | None = None, seed: int = 0) -> dict[str, list[int]]:
    """Measure dHash distances for known-same and known-different photos.

    Augmented copies in the training split share a :func:`source_id`, so they
    are known to be the *same* photo. This gives labelled examples to check
    whether :func:`dhash` can separate duplicates from different trays, and to
    choose a threshold for :func:`find_near_duplicates` from evidence.

    Args:
        root: Dataset root folder.
        split: Split containing augmented copies (normally ``"train"``).
        size: Hash grid size passed to :func:`dhash`.
        max_images: Optional random subset size, to keep the run short.
        seed: Seed for the subset.

    Returns:
        ``{"same": [...], "different": [...]}``. ``same`` holds the distance for
        every pair of copies with the same source id. ``different`` holds, for
        each image, the distance to its closest image with a *different*
        source id (the hardest negative). The two should not overlap if the
        hash is usable.
    """
    images = list_images(root, split)
    if max_images is not None and max_images < len(images):
        images = sample_for_review(images, max_images, seed=seed)
    hashed = [(source_id(p), dhash(p, size)) for p in images]

    same, different = [], []
    for i, (si, hi) in enumerate(hashed):
        closest_other = None
        for j, (sj, hj) in enumerate(hashed):
            if i == j:
                continue
            d = hamming(hi, hj)
            if si == sj:
                if j > i:
                    same.append(d)
            elif closest_other is None or d < closest_other:
                closest_other = d
        if closest_other is not None:
            different.append(closest_other)
    return {"same": same, "different": different}


def find_near_duplicates(root: str | Path, query_splits: tuple[str, ...] = ("valid", "test"),
                         reference_split: str = "train",
                         max_distance: int = 6, size: int = 8) -> list[dict]:
    """Find evaluation images that look like a training image, by content.

    Unlike :func:`find_cross_split_overlaps`, this ignores file names, so it
    catches the same tray saved under a different name and ignores different
    trays that share a name. Each query image is compared with every reference
    image (and with the other query splits) using :func:`dhash`.

    Args:
        root: Dataset root folder.
        query_splits: Splits to check (normally the evaluation splits).
        reference_split: Split to compare against (normally ``"train"``).
        max_distance: Largest Hamming distance counted as a match (out of
            ``size * size`` bits). Choose it with :func:`hash_calibration`
            rather than guessing: with ``size=8`` the trays in this dataset are
            so alike that different trays are often only 2-5 bits apart.
        size: Hash grid size passed to :func:`dhash`.

    Returns:
        One dict per flagged pair, sorted by distance: ``{"query": Path,
        "match": Path, "query_split": str, "match_split": str,
        "distance": int}``. Each query image is reported once, with its
        closest match.

    Note:
        Kept as a baseline. On this dataset it does **not** separate duplicates
        from different trays: every tray is photographed with the same camera,
        layout and background, so the 8 x 8 greyscale gradient pattern is
        almost the same for all images. 179 of 200 valid/test images had a
        "match" within 6 bits, and none of the closest pairs inspected were
        the same tray. Use :func:`nearest_by_correlation` instead.
    """
    hashes = {s: [(p, dhash(p, size)) for p in list_images(root, s)]
              for s in (reference_split, *query_splits)}

    flagged = []
    for i, qs in enumerate(query_splits):
        # Compare against train, plus later query splits (valid vs test once).
        targets = [reference_split, *query_splits[i + 1:]]
        for qp, qh in hashes[qs]:
            best = None
            for ts in targets:
                for tp, th in hashes[ts]:
                    d = hamming(qh, th)
                    if d <= max_distance and (best is None or d < best["distance"]):
                        best = {"query": qp, "match": tp, "query_split": qs,
                                "match_split": ts, "distance": d}
            if best:
                flagged.append(best)
    return sorted(flagged, key=lambda r: r["distance"])


def image_vectors(paths: list[Path], size: tuple[int, int] = (96, 72)) -> np.ndarray:
    """Turn images into normalised greyscale vectors for correlation.

    Each image is converted to greyscale, resized to ``size`` (width, height),
    flattened, shifted to zero mean and scaled to unit length. The dot product
    of two such vectors is then their Pearson correlation coefficient, which is
    unchanged by any brightness offset or contrast scaling. That makes it
    insensitive to the exposure augmentation, while it still compares the
    actual pixel layout (unlike an 8 x 8 hash).

    Args:
        paths: Image files.
        size: Resize target as (width, height). The default keeps the 4:3
            aspect ratio of the 720 x 540 images at 6,912 pixels each.

    Returns:
        Array of shape ``(len(paths), width * height)``, dtype float32.
        A completely flat image gives an all-zero row.
    """
    vecs = np.empty((len(paths), size[0] * size[1]), dtype=np.float32)
    for i, p in enumerate(paths):
        v = np.asarray(Image.open(p).convert("L").resize(size, Image.BILINEAR),
                       dtype=np.float32).ravel()
        v -= v.mean()
        norm = np.linalg.norm(v)
        vecs[i] = v / norm if norm > 0 else 0.0
    return vecs


def nearest_by_correlation(root: str | Path,
                           query_splits: tuple[str, ...] = ("valid", "test"),
                           reference_split: str = "train",
                           size: tuple[int, int] = (96, 72)) -> list[dict]:
    """Find, for every evaluation image, the most similar other image.

    Similarity is the Pearson correlation of the normalised greyscale images
    (see :func:`image_vectors`): 1.0 means identical apart from brightness and
    contrast. All pairs are computed at once as a matrix product, so this is
    fast even for thousands of images.

    Each query image is compared with the reference split and with any later
    query split (so valid is compared with test once, not twice).

    Args:
        root: Dataset root folder.
        query_splits: Splits to check (normally the evaluation splits).
        reference_split: Split to compare against (normally ``"train"``).
        size: Resize target passed to :func:`image_vectors`.

    Returns:
        One dict per query image, sorted from most to least similar:
        ``{"query": Path, "match": Path, "query_split": str,
        "match_split": str, "correlation": float}``. Choose a threshold from
        the distribution of ``correlation`` and inspect the pairs above it.
    """
    paths = {s: list_images(root, s) for s in (reference_split, *query_splits)}
    vecs = {s: image_vectors(p, size) for s, p in paths.items()}

    results = []
    for i, qs in enumerate(query_splits):
        targets = [reference_split, *query_splits[i + 1:]]
        tpaths = [(ts, p) for ts in targets for p in paths[ts]]
        if not tpaths or not paths[qs]:
            continue
        tvecs = np.vstack([vecs[ts] for ts in targets])
        sim = vecs[qs] @ tvecs.T                      # (n_query, n_target) correlations
        best = sim.argmax(axis=1)
        for qi, ti in enumerate(best):
            ts, tp = tpaths[ti]
            results.append({"query": paths[qs][qi], "match": tp, "query_split": qs,
                            "match_split": ts, "correlation": float(sim[qi, ti])})
    return sorted(results, key=lambda r: -r["correlation"])


# --------------------------------------------------------------------------
# Fixes and manual review
# --------------------------------------------------------------------------


def quarantine_overlaps(root: str | Path, overlaps: list[dict],
                        quarantine_dir: str | Path, from_split: str = "train",
                        log_csv: str | Path | None = None,
                        dry_run: bool = True) -> list[Path]:
    """Move overlapping images (and their labels) out of one split.

    Files are moved, not deleted, so the change can be undone and audited.
    The evaluation splits are left untouched by default: removing the copies
    from ``train`` keeps ``valid`` and ``test`` the same size.

    Args:
        root: Dataset root folder.
        overlaps: Output of :func:`find_cross_split_overlaps`, optionally
            filtered to the pairs confirmed as the same photo.
        quarantine_dir: Folder to move files into. Keep it under ``data/`` so
            it is git-ignored.
        from_split: Split to remove the overlapping copies from.
        log_csv: Optional CSV file recording each moved image (commit this to
            ``results/`` as evidence).
        dry_run: If ``True`` (the default), only report what would be moved.

    Returns:
        Image paths that were moved (or would be moved, in a dry run).
    """
    quarantine = Path(quarantine_dir) / from_split
    to_move = [img for o in overlaps if from_split in o["splits"]
               for img in o["images"][from_split]]

    if not dry_run:
        (quarantine / "images").mkdir(parents=True, exist_ok=True)
        (quarantine / "labels").mkdir(parents=True, exist_ok=True)
        for img in to_move:
            lp = label_path_for(img)
            shutil.move(str(img), quarantine / "images" / img.name)
            if lp.exists():
                shutil.move(str(lp), quarantine / "labels" / lp.name)

    if log_csv is not None:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "splits", "moved_image", "dry_run"])
            for o in overlaps:
                for img in o["images"].get(from_split, []):
                    writer.writerow([o["source"], "+".join(o["splits"]), img.name, dry_run])
    return to_move


def sample_for_review(images: list[Path], n: int, seed: int = 0) -> list[Path]:
    """Pick a reproducible random sample of images for manual inspection.

    Args:
        images: Images to sample from.
        n: Sample size. If larger than ``len(images)``, every image is returned.
        seed: Random seed. Report it so the sample can be reproduced.

    Returns:
        The sampled image paths, in sampled order.
    """
    rng = random.Random(seed)
    return rng.sample(list(images), min(n, len(images)))


def write_review_sheet(images: list[Path], out_csv: str | Path) -> Path:
    """Write a CSV template for recording manual review verdicts.

    Columns are ``index, image, verdict, notes``. Fill ``verdict`` with
    ``clean``, ``missed_defect`` or ``unclear``, using criteria written down
    before reviewing.

    Args:
        images: Images being reviewed, in display order.
        out_csv: Where to write the CSV.

    Returns:
        Path to the CSV file.
    """
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "verdict", "notes"])
        for i, img in enumerate(images):
            writer.writerow([i, Path(img).name, "", ""])
    return out
