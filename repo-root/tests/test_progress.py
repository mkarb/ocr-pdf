"""Test that pdf_to_vectormap reports per-page progress (7.4)."""
import fitz

from pdf_compare.pdf_extract import pdf_to_vectormap


def test_progress_callback_invoked_per_page(tmp_path):
    pdf = tmp_path / "two.pdf"
    doc = fitz.open()
    doc.new_page(width=100, height=80)
    doc.new_page(width=100, height=80)
    doc.save(str(pdf))
    doc.close()

    calls = []
    pdf_to_vectormap(str(pdf), workers=1, progress_callback=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 2), (2, 2)]
