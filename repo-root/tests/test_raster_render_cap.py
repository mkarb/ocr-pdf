"""Tests for raster-grid render-resolution capping (S3: large sheets must not
render whole at full DPI)."""
import fitz

from pdf_compare.raster_grid import _capped_dpi, raster_grid_changed_boxes


def test_capped_dpi_no_cap_when_within_budget():
    # 200x150 pt at 72 DPI -> 200 px longest, well under 8000.
    assert _capped_dpi(200, 150, 72, 8000) == 72.0


def test_capped_dpi_reduces_for_large_sheet():
    # 1000 pt at 400 DPI -> longest ~5555 px; cap to 2000.
    eff = _capped_dpi(1000, 500, 400, 2000)
    assert eff < 400
    # The capped render's longest side lands ~ at the budget.
    assert abs(1000 * (eff / 72.0) - 2000) < 1.0


def _make_pdf(path, boxes):
    doc = fitz.open()
    page = doc.new_page(width=200, height=150)
    for rect in boxes:
        page.draw_rect(fitz.Rect(*rect), fill=(0, 0, 0), width=1)
    doc.save(str(path))
    doc.close()


def test_identical_pages_return_empty(tmp_path):
    p = tmp_path / "a.pdf"
    _make_pdf(p, [(40, 40, 160, 70)])
    boxes, m = raster_grid_changed_boxes(str(p), str(p), 0, return_metrics=True)
    assert boxes == []
    assert m["identical"] is True


def test_changed_pages_run_and_report_render_dpi(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    _make_pdf(a, [(40, 40, 160, 70)])
    _make_pdf(b, [(40, 40, 160, 70), (40, 100, 160, 130)])  # an added box
    boxes, m = raster_grid_changed_boxes(
        str(a), str(b), 0, method="abs", cell_change_ratio=0.01,
        skip_empty_cells=False, max_render_pixels=400, return_metrics=True,
    )
    assert m["identical"] is False
    assert "render_dpi" in m            # capping path executed
    assert isinstance(boxes, list)
