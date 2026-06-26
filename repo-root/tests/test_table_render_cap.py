"""Tests for table_extractor render bounding (S6: don't render large sheets whole)."""
import fitz

from pdf_compare.analyzers.table_extractor import (
    _capped_dpi,
    TableExtractor,
    TableExtractionConfig,
)


def test_capped_dpi_no_cap_when_small():
    assert _capped_dpi(200, 150, 72, 8000) == 72.0


def test_capped_dpi_reduces_large():
    eff = _capped_dpi(1000, 500, 400, 2000)
    assert eff < 400
    assert abs(1000 * (eff / 72.0) - 2000) < 1.0


def test_detect_regions_runs_under_cap(tmp_path):
    pdf = tmp_path / "grid.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    for x in (50, 150, 250):
        page.draw_line(fitz.Point(x, 40), fitz.Point(x, 160), width=1)
    for y in (40, 100, 160):
        page.draw_line(fitz.Point(50, y), fitz.Point(250, y), width=1)
    doc.save(str(pdf))
    doc.close()

    # Tiny cap forces the capped-DPI render path; should run without error.
    ext = TableExtractor(TableExtractionConfig(max_render_pixels=200))
    regions = ext.detect_table_regions(str(pdf), 0)
    assert isinstance(regions, list)


def test_extract_table_clip_path_runs(tmp_path):
    # A blank region yields no cells, so extract_table returns before any OCR —
    # but it still exercises the new clip-based render path (get_pixmap(clip=...)).
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=200)
    doc.save(str(pdf))
    doc.close()

    ext = TableExtractor(TableExtractionConfig())
    result = ext.extract_table(str(pdf), 0, table_bbox=(50, 50, 250, 150))
    assert result is None
