"""Plotting helpers for inspecting dataset images in a notebook."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .dataset import label_path_for, read_labels


def _draw_boxes(ax, image_path: Path, class_names: list[str] | None) -> None:
    """Overlay the YOLO boxes for ``image_path`` on an axis, if it has any."""
    lp = label_path_for(image_path)
    if not lp.exists():
        return
    width, height = Image.open(image_path).size
    for b in read_labels(lp):
        x0 = (b.x - b.w / 2) * width
        y0 = (b.y - b.h / 2) * height
        ax.add_patch(Rectangle((x0, y0), b.w * width, b.h * height,
                               fill=False, edgecolor="yellow", linewidth=1.5))
        if class_names:
            ax.text(x0, y0, class_names[b.class_id], color="black", fontsize=7,
                    backgroundcolor="yellow")


def show_grid(images: list[Path], cols: int = 6,
              class_names: list[str] | None = None, title: str | None = None):
    """Show images in a numbered grid, with label boxes drawn on top.

    The number above each image matches the ``index`` column written by
    :func:`rasher_inspection.dataset.write_review_sheet`.

    Args:
        images: Image paths to show.
        cols: Number of columns.
        class_names: If given, each box is labelled with its class name.
        title: Optional figure title.

    Returns:
        The matplotlib figure.
    """
    rows = max(1, math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(images):
            continue
        ax.imshow(Image.open(images[i]))
        _draw_boxes(ax, Path(images[i]), class_names)
        ax.set_title(f"{i}: {Path(images[i]).name[:28]}", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_pairs(overlaps: list[dict], class_names: list[str] | None = None):
    """Show each cross-split overlap side by side, one row per source photo.

    Args:
        overlaps: Output of
            :func:`rasher_inspection.dataset.find_cross_split_overlaps`.
        class_names: If given, boxes are labelled with class names.

    Returns:
        The matplotlib figure, or ``None`` if there are no overlaps.
    """
    if not overlaps:
        print("No cross-split overlaps.")
        return None
    cols = max(len(o["splits"]) for o in overlaps)
    fig, axes = plt.subplots(len(overlaps), cols,
                             figsize=(5 * cols, 3.6 * len(overlaps)), squeeze=False)
    for row, o in zip(axes, overlaps):
        for ax in row:
            ax.axis("off")
        for ax, split in zip(row, o["splits"]):
            img = o["images"][split][0]
            ax.imshow(Image.open(img))
            _draw_boxes(ax, Path(img), class_names)
            ax.set_title(f"{split}: {o['source'][:30]}", fontsize=8)
    fig.tight_layout()
    return fig


def show_matches(matches: list[dict], class_names: list[str] | None = None,
                 max_rows: int = 20):
    """Show near-duplicate pairs side by side, closest first.

    Args:
        matches: Output of
            :func:`rasher_inspection.dataset.find_near_duplicates`.
        class_names: If given, boxes are labelled with class names.
        max_rows: Show at most this many pairs.

    Returns:
        The matplotlib figure, or ``None`` if there are no matches.
    """
    if not matches:
        print("No near-duplicates found.")
        return None
    rows = matches[:max_rows]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.6 * len(rows)), squeeze=False)
    for i, (row, m) in enumerate(zip(axes, rows)):
        for ax, key, split in ((row[0], "query", m["query_split"]),
                               (row[1], "match", m["match_split"])):
            ax.axis("off")
            ax.imshow(Image.open(m[key]))
            _draw_boxes(ax, Path(m[key]), class_names)
            ax.set_title(f"{i}: {split}: {Path(m[key]).name[:28]}  (d={m['distance']})",
                         fontsize=8)
    fig.tight_layout()
    return fig
