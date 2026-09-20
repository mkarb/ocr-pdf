"""Tiles holding sparse-but-legible text must not be discarded as blank.

Ink coverage on an engineering sheet is roughly scale-invariant (~0.3-1% of
area), so the old `content_ratio >= 0.01` test classified tiles full of legible
callouts as empty and returned no OCR for them at all.
"""
import importlib

import numpy as np

h = importlib.import_module("pdf_compare.analyzers.highres_ocr")


def _blank(w=4000, h=3000):
    return np.full((h, w), 255, np.uint8)


def _sparse_text_tile():
    """A tile with several glyph-sized ink blobs - far under 1% of area."""
    img = _blank()
    for i in range(12):
        x, y = 200 + (i % 4) * 700, 300 + (i // 4) * 800
        img[y:y + 40, x:x + 180] = 0  # ~7.2k inked px each
    return img


def test_sparse_text_tile_is_content():
    img = _sparse_text_tile()
    ink_ratio = float((img < 250).mean())
    assert ink_ratio < 0.01, "fixture must sit under the old 1% threshold"
    assert h.detect_tile_content(img) is True


def test_blank_tile_is_not_content():
    assert h.detect_tile_content(_blank()) is False


def test_threshold_is_an_absolute_pixel_floor():
    img = _blank()
    img[100:110, 100:140] = 0  # 400 inked px
    assert h.detect_tile_content(img, min_content_pixels=100) is True
    assert h.detect_tile_content(img, min_content_pixels=10_000) is False


def test_pixel_floor_is_independent_of_tile_size():
    """The same ink must read as content whether the tile is small or large."""
    for w, h_ in ((1500, 1200), (6000, 4800)):
        img = np.full((h_, w), 255, np.uint8)
        img[50:90, 50:230] = 0
        assert h.detect_tile_content(img) is True, f"tile {w}x{h_} lost its text"
