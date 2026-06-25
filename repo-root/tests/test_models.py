"""Tests for pdf_compare.models — text provenance (`source`) wiring."""
from pdf_compare.models import TextRun, text_run_from_dict


def test_text_run_default_source_is_native():
    tr = TextRun(text="x", bbox=(0, 0, 1, 1), font=None, size=None)
    assert tr.source == "native"


def test_from_dict_preserves_ocr_source():
    # Regression: OCR-extracted text used to lose its provenance at conversion.
    d = {"text": "Valve", "bbox": [1, 2, 3, 4], "font": None, "size": None, "source": "ocr"}
    tr = text_run_from_dict(d)
    assert tr.source == "ocr"
    assert tr.bbox == (1, 2, 3, 4)


def test_from_dict_defaults_missing_source_to_native():
    d = {"text": "Pump", "bbox": [0, 0, 5, 5]}
    assert text_run_from_dict(d).source == "native"
