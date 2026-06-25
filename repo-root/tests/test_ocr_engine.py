"""Tests for OCR engine/GPU resolution (graceful fallback for large-diagram OCR)."""
import importlib

# Import the actual module object. Note: `pdf_compare.analyzers.highres_ocr`
# attribute access returns the re-exported *function* (the package __init__ binds
# a name that shadows the submodule), so use import_module to get the module.
h = importlib.import_module("pdf_compare.analyzers.highres_ocr")


def test_explicit_engine_and_no_gpu():
    assert h.resolve_ocr_engine(engine="tesseract", use_gpu=False) == ("tesseract", False)


def test_explicit_gpu_flag_is_respected():
    assert h.resolve_ocr_engine(engine="tesseract", use_gpu=True) == ("tesseract", True)


def test_env_engine_override(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.delenv("OCR_USE_GPU", raising=False)
    engine, _ = h.resolve_ocr_engine()
    assert engine == "tesseract"


def test_env_gpu_override(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setenv("OCR_USE_GPU", "0")
    assert h.resolve_ocr_engine() == ("tesseract", False)


def test_env_gpu_force_on(monkeypatch):
    # A 5090 box can force GPU even if autodetect is bypassed.
    monkeypatch.setenv("OCR_ENGINE", "easyocr")
    monkeypatch.setenv("OCR_USE_GPU", "1")
    monkeypatch.setattr(h, "HAVE_EASYOCR", True)
    assert h.resolve_ocr_engine() == ("easyocr", True)


def test_easyocr_request_falls_back_when_not_installed(monkeypatch):
    # Regression: EasyOCR-missing must not leave OCR broken; fall back to Tesseract.
    monkeypatch.setattr(h, "HAVE_EASYOCR", False)
    engine, gpu = h.resolve_ocr_engine(engine="easyocr")
    assert engine == "tesseract"
    assert gpu is False
