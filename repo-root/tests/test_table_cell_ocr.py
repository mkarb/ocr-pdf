"""Tests for table cell OCR engine routing (5.3 — GPU-capable cell OCR)."""
import numpy as np

import pdf_compare.analyzers.table_extractor as te


def test_routes_to_easyocr_when_selected(monkeypatch):
    monkeypatch.setattr(te, "_easyocr_cell", lambda gray, gpu: ("EASY", 90.0))
    monkeypatch.setattr(te, "_tesseract_cell", lambda gray, cfg: ("TESS", 80.0))
    img = np.zeros((20, 40), dtype=np.uint8)
    cfg = te.TableExtractionConfig()
    assert te.extract_cell_text(img, (0, 0, 40, 20), cfg, engine="easyocr", use_gpu=True) == ("EASY", 90.0)


def test_routes_to_tesseract_by_default(monkeypatch):
    monkeypatch.setattr(te, "_easyocr_cell", lambda gray, gpu: ("EASY", 90.0))
    monkeypatch.setattr(te, "_tesseract_cell", lambda gray, cfg: ("TESS", 80.0))
    img = np.zeros((20, 40), dtype=np.uint8)
    cfg = te.TableExtractionConfig()
    assert te.extract_cell_text(img, (0, 0, 40, 20), cfg, engine="tesseract") == ("TESS", 80.0)


def test_empty_cell_returns_blank():
    img = np.zeros((10, 10), dtype=np.uint8)
    cfg = te.TableExtractionConfig()
    assert te.extract_cell_text(img, (0, 0, 0, 0), cfg, engine="tesseract") == ("", 0.0)


def test_config_exposes_ocr_engine_fields():
    cfg = te.TableExtractionConfig()
    assert hasattr(cfg, "ocr_engine") and hasattr(cfg, "ocr_use_gpu")
