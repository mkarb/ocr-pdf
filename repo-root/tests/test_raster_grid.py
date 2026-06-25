"""Tests for raster_grid box merging."""
from pdf_compare.raster_grid import _merge_adjacent_boxes


def test_empty():
    assert _merge_adjacent_boxes([]) == []


def test_non_touching_boxes_kept_separate():
    boxes = [(0, 0, 5, 5), (100, 100, 105, 105)]
    assert len(_merge_adjacent_boxes(boxes)) == 2


def test_chain_merge_across_sort_order():
    # Regression for the single-pass merge: the third box overlaps the first
    # but, after sorting, a non-touching box sits between them. A merge that
    # only compares against the previous box leaves them un-merged.
    boxes = [(0, 0, 5, 5), (1, 100, 6, 105), (2, 0, 7, 5)]
    merged = _merge_adjacent_boxes(boxes)
    assert len(merged) == 2
    # The two overlapping boxes collapse into one spanning x:0..7, y:0..5.
    assert (0, 0, 7, 5) in merged
