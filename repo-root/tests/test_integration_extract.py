"""End-to-end extraction validation on a real PyMuPDF-generated PDF.

Unlike test_extract_core (fake page objects), this drives the full
``pdf_to_vectormap`` pipeline on an actual PDF to confirm geometry and native
text (with provenance) survive end to end. No database or OCR engine required.
"""
import fitz

from pdf_compare.pdf_extract import pdf_to_vectormap
from pdf_compare.models import GeoKind


def test_extraction_pipeline(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(fitz.Rect(50, 50, 250, 150), color=(0, 0, 0), width=1)  # stroked box
    page.insert_text(fitz.Point(60, 100), "VALVE V-101", fontsize=12)       # native text
    doc.save(str(pdf))
    doc.close()

    vm = pdf_to_vectormap(str(pdf), workers=1)

    assert vm.meta.page_count == 1
    page = vm.pages[0]

    # Geometry: the drawn rectangle must produce stroke geometry.
    assert page.geoms, "no geometry extracted from a page with a drawn rectangle"
    assert any(g.kind == GeoKind.STROKE for g in page.geoms)

    # Union of geom bboxes must cover the rectangle extent (whether PyMuPDF
    # emits a single 're' ring or decomposes it into line segments).
    x0 = min(g.bbox[0] for g in page.geoms)
    y0 = min(g.bbox[1] for g in page.geoms)
    x1 = max(g.bbox[2] for g in page.geoms)
    y1 = max(g.bbox[3] for g in page.geoms)
    assert x0 <= 55 and y0 <= 55 and x1 >= 245 and y1 >= 145, (x0, y0, x1, y1)

    # Native text with correct provenance.
    assert any("VALVE" in t.text for t in page.texts), [t.text for t in page.texts]
    assert page.texts and all(t.source == "native" for t in page.texts)


def test_resolve_engine_real_fallback():
    # EasyOCR is not installed in this environment, so requesting it must fall
    # back to Tesseract rather than failing — the real (non-mocked) fallback.
    from pdf_compare.analyzers.highres_ocr import resolve_ocr_engine, HAVE_EASYOCR

    engine, gpu = resolve_ocr_engine(engine="easyocr")
    if HAVE_EASYOCR:
        assert engine == "easyocr"
    else:
        assert engine == "tesseract"
        assert gpu is False
