"""Tests for ocr_page auto-tiling decision (S1: large pages must tile)."""
import importlib

import fitz

# import_module to get the real submodule (the package __init__ re-exports a
# function named `highres_ocr` that otherwise shadows the submodule attribute).
h = importlib.import_module("pdf_compare.analyzers.highres_ocr")


def _make_pdf(path, w=200, h=100):
    doc = fitz.open()
    doc.new_page(width=w, height=h)
    doc.save(str(path))
    doc.close()


def test_small_page_ocrs_whole(tmp_path, monkeypatch):
    pdf = tmp_path / "small.pdf"
    _make_pdf(pdf)
    monkeypatch.setattr(h, "resolve_ocr_engine", lambda e, g: ("tesseract", False))
    monkeypatch.setattr(h, "highres_ocr", lambda p, i, cfg: [{"text": "WHOLE", "bbox": (0, 0, 1, 1)}])

    def _no_tile(*a, **k):
        raise AssertionError("small page should not tile")
    monkeypatch.setattr(h, "tiled_ocr", _no_tile)

    out = h.ocr_page(str(pdf), 0, dpi=72)  # 200x100pt @ zoom 1 = 200x100px << 29000
    assert out == [{"text": "WHOLE", "bbox": (0, 0, 1, 1)}]


def test_large_page_tiles(tmp_path, monkeypatch):
    pdf = tmp_path / "large.pdf"
    _make_pdf(pdf)
    monkeypatch.setattr(h, "resolve_ocr_engine", lambda e, g: ("tesseract", False))
    monkeypatch.setattr(h, "tiled_ocr", lambda *a, **k: [{"text": "TILE", "bbox": (0, 0, 1, 1), "conf": 90}])

    def _no_whole(*a, **k):
        raise AssertionError("large page should tile, not whole-page OCR")
    monkeypatch.setattr(h, "highres_ocr", _no_whole)

    # 200pt * (11000/72) ~= 30555 px > 29000 -> tiling
    out = h.ocr_page(str(pdf), 0, dpi=11000)
    assert out == [{"text": "TILE", "bbox": (0, 0, 1, 1)}]  # conf normalized away
