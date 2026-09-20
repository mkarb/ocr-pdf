"""Regression coverage for raster detection and revised-page coordinates."""

import cv2
import fitz
import numpy as np
import pytest

from pdf_compare.raster_grid import (
    _align_ecc,
    _capped_dpi,
    _mask_diff_adaptive,
    raster_grid_changed_boxes,
    raster_grid_changed_boxes_aligned,
)


def _pdf(path, image=None, *, size=(320, 240), rotation=0):
    with fitz.open() as doc:
        page = doc.new_page(width=size[0], height=size[1])
        if image is not None:
            ok, png = cv2.imencode(".png", image)
            assert ok
            page.insert_image(page.rect, stream=png.tobytes())
        page.set_rotation(rotation)
        doc.save(path)
    return str(path)


def _drawing(sparse=False):
    image = np.full((240, 320), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (100, 85), 0, 1 if sparse else -1)
    cv2.rectangle(image, (185, 145), (270, 195), 0, 1 if sparse else 3)
    return image


@pytest.mark.parametrize("sparse", [False, True])
def test_ecc_removes_scan_translation(sparse):
    base = _drawing(sparse)
    moving = cv2.warpAffine(
        base, np.float32([[1, 0, 7], [0, 1, -5]]), (320, 240), borderValue=255
    )
    aligned, metrics = _align_ecc(base, moving)
    before = cv2.absdiff(base, moving).mean()
    assert metrics["success"] is True
    assert cv2.absdiff(base, aligned).mean() < before * 0.05


def test_ecc_translation_on_large_renders_uses_full_resolution_coordinates():
    base = cv2.resize(_drawing(), (2400, 1800), interpolation=cv2.INTER_NEAREST)
    moving = cv2.warpAffine(
        base, np.float32([[1, 0, 21], [0, 1, -15]]), (2400, 1800), borderValue=255
    )
    aligned, metrics = _align_ecc(base, moving)
    assert metrics["translation"] == pytest.approx(np.hypot(21, 15), abs=0.5)
    assert cv2.absdiff(base, aligned).mean() < cv2.absdiff(base, moving).mean() * 0.05


@pytest.mark.parametrize("gray", [0, 100])
def test_adaptive_detects_uniform_whole_page_change(gray):
    old = np.full((40, 40), 255, np.uint8)
    new = np.full_like(old, gray)
    mask, _ = _mask_diff_adaptive(old, new)
    assert np.all(mask == 255)


@pytest.mark.parametrize("method", ["abs", "hybrid"])
def test_manual_zero_threshold_detects_small_difference(method):
    old = np.full((20, 20), 255, np.uint8)
    new = np.full_like(old, 254)
    mask, metrics = _mask_diff_adaptive(old, new, method=method, threshold=0)
    assert metrics["threshold"] == 0
    assert np.all(mask == 255)


@pytest.mark.parametrize("new_size", [(400, 240), (200, 240), (320, 300)])
def test_blank_page_size_changes_are_reported_in_new_page_bounds(tmp_path, new_size):
    old = _pdf(tmp_path / "old.pdf")
    new = _pdf(tmp_path / "new.pdf", size=new_size)
    boxes, metrics = raster_grid_changed_boxes(old, new, 0, dpi=72, return_metrics=True)
    assert boxes == [(0, 0, *new_size)]
    assert metrics["page_size_changed"] is True
    assert metrics["identical"] is False
    assert metrics["coordinate_space"] == "display"
    assert metrics["base_page"] == "new"


def test_highlights_follow_revised_page_after_registration(tmp_path):
    old_image = _drawing()
    new_image = cv2.warpAffine(
        old_image, np.float32([[1, 0, 10], [0, 1, 6]]), (320, 240), borderValue=255
    )
    cv2.rectangle(new_image, (115, 170), (145, 200), 0, -1)
    old = _pdf(tmp_path / "old.pdf", old_image)
    new = _pdf(tmp_path / "new.pdf", new_image)
    boxes = raster_grid_changed_boxes(
        old, new, 0, dpi=72, rows=24, cols=32, method="abs", cell_change_ratio=0.1
    )
    assert boxes
    assert any(x0 <= 140 <= x1 and y0 <= 195 <= y1 for x0, y0, x1, y1 in boxes)
    assert not any(x0 <= 50 <= x1 and y0 <= 50 <= y1 for x0, y0, x1, y1 in boxes)
    # The new added rectangle reaches x=145, unlike its old-frame position x=135.
    assert max(box[2] for box in boxes) >= 145


def test_sparse_cell_optimization_preserves_detected_changes(tmp_path):
    new_image = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(new_image, (90, 90), (94, 94), 0, -1)
    old = _pdf(tmp_path / "old.pdf", size=(200, 200))
    new = _pdf(tmp_path / "new.pdf", new_image, size=(200, 200))
    boxes = raster_grid_changed_boxes(
        old, new, 0, dpi=72, rows=1, cols=1, cell_change_ratio=0.0001
    )
    assert boxes == [(0, 0, 200, 200)]


@pytest.mark.parametrize("method", ["adaptive", "ssim"])
def test_tiny_render_and_grid_larger_than_image(tmp_path, method):
    old = _pdf(tmp_path / "old.pdf", size=(2, 2))
    new = _pdf(tmp_path / "new.pdf", np.zeros((2, 2), np.uint8), size=(2, 2))
    boxes, metrics = raster_grid_changed_boxes(
        old, new, 0, dpi=72, method=method, return_metrics=True
    )
    assert boxes == [(0, 0, 2, 2)]
    assert metrics["total_cells"] == 4


def test_pixel_rounding_cannot_place_boxes_outside_pdf_page(tmp_path):
    size = (200.25, 150.5)
    old = _pdf(tmp_path / "old.pdf", size=size)
    new = _pdf(tmp_path / "new.pdf", np.zeros((150, 200), np.uint8), size=size)
    boxes = raster_grid_changed_boxes(old, new, 0, dpi=85)
    assert boxes == [(0, 0, *size)]


def test_rotated_pdf_boxes_use_display_coordinates(tmp_path):
    old = _pdf(tmp_path / "old.pdf", rotation=90)
    new = _pdf(tmp_path / "new.pdf", np.zeros((240, 320), np.uint8), rotation=90)
    boxes = raster_grid_changed_boxes(old, new, 0, dpi=72)
    assert boxes == [(0, 0, 240, 320)]


def test_zero_change_ratio_does_not_flag_unchanged_grid_cells(tmp_path):
    new_image = np.full((240, 320), 255, np.uint8)
    cv2.rectangle(new_image, (10, 10), (30, 30), 0, -1)
    old = _pdf(tmp_path / "old.pdf")
    new = _pdf(tmp_path / "new.pdf", new_image)
    boxes = raster_grid_changed_boxes(
        old, new, 0, dpi=72, rows=2, cols=2,
        cell_change_ratio=0, skip_empty_cells=False,
    )
    assert boxes == [(0, 0, 160, 120)]


def test_alternate_page_index_compares_the_selected_revision_page(tmp_path):
    old = _pdf(tmp_path / "old.pdf", _drawing())
    new_path = tmp_path / "new.pdf"
    with fitz.open(old) as source, fitz.open() as doc:
        doc.new_page(width=320, height=240)
        doc.insert_pdf(source)
        doc.save(new_path)
    boxes = raster_grid_changed_boxes_aligned(old, str(new_path), 0, 1, dpi=72)
    assert boxes == []


@pytest.mark.parametrize(
    "kwargs",
    [{"rows": 0}, {"cols": -1}, {"dpi": 0}, {"max_render_pixels": 0},
     {"cell_change_ratio": -0.1}, {"min_content_ratio": 1.1},
     {"threshold": -1}, {"method": "typo"}],
)
def test_invalid_settings_raise_clear_errors_before_rendering(kwargs):
    with pytest.raises(ValueError):
        raster_grid_changed_boxes("unused.pdf", "unused.pdf", 0, **kwargs)


def test_negative_page_index_is_not_silently_treated_as_last_page(tmp_path):
    pdf = _pdf(tmp_path / "one.pdf")
    with pytest.raises(ValueError, match="Page index"):
        raster_grid_changed_boxes(pdf, pdf, -1)


def test_render_cap_still_applies_below_one_dpi():
    dpi = _capped_dpi(20000, 10000, 400, 100)
    assert dpi * 20000 / 72 <= 100
