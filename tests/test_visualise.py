"""Smoke tests for the plotting helpers (they must run without errors)."""

import matplotlib

matplotlib.use("Agg")  # no display needed

from rasher_inspection import dataset as ds
from rasher_inspection import visualise as vis
from conftest import NAMES


def test_show_grid_numbers_every_image(fake_root):
    imgs = ds.list_images(fake_root, "train")
    fig = vis.show_grid(imgs, cols=2, class_names=NAMES)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 3 and titles[0].startswith("0:")


def test_show_pairs_one_row_per_overlap(fake_root):
    overlaps = ds.find_cross_split_overlaps(fake_root)
    fig = vis.show_pairs(overlaps, class_names=NAMES)
    assert len(fig.axes) == 2


def test_show_pairs_handles_no_overlaps():
    assert vis.show_pairs([]) is None


def test_show_matches_handles_empty_and_pairs(fake_root):
    assert vis.show_matches([]) is None
    imgs = ds.list_images(fake_root, "test")
    m = [{"query": imgs[0], "match": imgs[1], "query_split": "test",
          "match_split": "train", "distance": 0}]
    assert len(vis.show_matches(m).axes) == 2
