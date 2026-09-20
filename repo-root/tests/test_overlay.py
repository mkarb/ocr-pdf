"""Exercise saved overlay PDFs, including rendering and recovery paths."""
import os

import fitz
import pytest

from pdf_compare import overlay


def _make_pdf(path, pages=1, rotation=0):
    with fitz.open() as doc:
        for number in range(pages):
            page = doc.new_page(width=400, height=300)
            page.insert_text((150, 150), f"Source page {number + 1}")
            page.set_rotation(rotation)
        doc.save(path)


def _diff(page=1, **metadata):
    return {
        "page": page,
        "geometry": {"added": [], "removed": [], "changed": []},
        "text": {"added": [], "removed": [], "moved": []},
        **metadata,
    }


def test_highlights_are_translucent_rgb_and_legend_is_readable(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "nested" / "out.pdf"
    _make_pdf(source)
    diff = _diff()
    for index, kind in enumerate(("added", "removed", "changed")):
        diff["geometry"][kind] = [(40 + index * 100, 200, 90 + index * 100, 250)]
    diff["text"]["moved"] = [{"from": (260, 100, 280, 110), "to": (290, 100, 310, 110)}]

    overlay.write_overlay(source, [diff], output)

    with fitz.open(output) as doc:
        page = doc[0]
        drawing = page.get_drawings()
        for shape, color in zip(drawing[:3], ((0, 1, 0), (1, 0, 0), (0.5, 0, 0.8))):
            assert shape["fill"] == pytest.approx(color)
            assert shape["fill_opacity"] == pytest.approx(0.15)
        assert drawing[4]["dashes"] == "[ 2 2 ] 0"
        pix = page.get_pixmap()
        red, green, blue = pix.pixel(60, 220)
        assert green == 255 and 210 <= red <= 220 and red == blue
        assert pix.pixel(20, 20) == (255, 255, 255)
        assert all(label in page.get_text() for label in ("Legend", "Added", "Removed", "Changed", "Moved"))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("coordinate_space", ["unrotated", "display"])
@pytest.mark.parametrize("fallback", [False, True])
def test_rotated_boxes_stay_on_the_displayed_change(tmp_path, monkeypatch, rotation, coordinate_space, fallback):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    _make_pdf(source, pages=2, rotation=rotation)
    with fitz.open(source) as doc:
        page = doc[1]
        display_box = fitz.Rect(page.rect.width - 70, page.rect.height - 70,
                                page.rect.width - 20, page.rect.height - 20)
        box = display_box if coordinate_space == "display" else display_box * page.derotation_matrix
    diff = _diff(2, coordinate_space=coordinate_space)
    diff["geometry"]["changed"] = [tuple(box)]

    if fallback:
        draw = overlay._draw_overlay_rects
        calls = 0

        def fail_first(page, diff):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Force raster recovery")
            return draw(page, diff)

        monkeypatch.setattr(overlay, "_draw_overlay_rects", fail_first)
        monkeypatch.setattr(overlay, "_try_strip_layers_with_pikepdf", lambda path: None)

    overlay.write_overlay(source, [diff], output)

    with fitz.open(source) as original, fitz.open(output) as doc:
        assert len(doc) == 2  # Unselected pages survive recovery.
        page = doc[1]
        assert page.rotation == rotation
        assert page.rect == original[1].rect
        pix = page.get_pixmap()
        x, y = int(display_box.x0 + 20), int(display_box.y0 + 20)
        red, green, blue = pix.pixel(x, y)
        assert 210 <= green <= 220 and blue > red > green
        legend = page.search_for("Legend")
        assert len(legend) == 1
        assert (legend[0] * page.rotation_matrix).y0 < 40


@pytest.mark.parametrize("status", ["added", "removed"])
def test_missing_pages_are_copied_from_other_revision(tmp_path, status):
    base, alternate, output = (tmp_path / name for name in ("base.pdf", "alternate.pdf", "out.pdf"))
    _make_pdf(base)
    _make_pdf(alternate, pages=3, rotation=90)
    diff = _diff(3, page_status=status)

    overlay.write_overlay(base, [diff], output, alternate_pdf_path=alternate)

    with fitz.open(output) as doc:
        assert len(doc) == 3
        assert "Source page 1" in doc[0].get_text()
        assert "Source page 2" in doc[1].get_text()
        assert "Source page 3" in doc[2].get_text()
        assert f"Page {status}" in doc[2].get_text()
        assert doc[2].rotation == 90


def test_missing_page_requires_source_instead_of_silent_omission(tmp_path):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    with pytest.raises(ValueError, match="alternate_pdf_path"):
        overlay.write_overlay(source, [_diff(2)], tmp_path / "out.pdf")


@pytest.mark.parametrize("page", [0, -1, 1.5])
def test_invalid_page_number_is_rejected(tmp_path, page):
    source = tmp_path / "source.pdf"
    _make_pdf(source)
    with pytest.raises(ValueError, match="positive integers"):
        overlay.write_overlay(source, [_diff(page)], tmp_path / "out.pdf")


def test_empty_diffs_preserve_document_without_legend(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    _make_pdf(source, pages=3)
    overlay.write_overlay(source, [], output)
    with fitz.open(output) as doc:
        assert len(doc) == 3
        assert all("Legend" not in page.get_text() for page in doc)


def test_output_cannot_overwrite_either_source(tmp_path):
    source, alternate = tmp_path / "source.pdf", tmp_path / "alternate.pdf"
    _make_pdf(source)
    _make_pdf(alternate)
    for output in (source, alternate):
        with pytest.raises(ValueError, match="different from the source"):
            overlay.write_overlay(source, [], output, alternate_pdf_path=alternate)


@pytest.mark.parametrize("failures", [1, 2])
def test_layer_retry_uses_temporary_file_and_cleans_it(tmp_path, monkeypatch, failures):
    pytest.importorskip("pikepdf")
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    _make_pdf(source)
    draw = overlay._draw_overlay_rects
    strip = overlay._try_strip_layers_with_pikepdf
    calls, temporary_paths = 0, []

    def fail_first(page, diff):
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise RuntimeError("Force layer recovery")
        draw(page, diff)

    def track_strip(path):
        result = strip(path)
        temporary_paths.append(result)
        return result

    monkeypatch.setattr(overlay, "_draw_overlay_rects", fail_first)
    monkeypatch.setattr(overlay, "_try_strip_layers_with_pikepdf", track_strip)
    overlay.write_overlay(source, [_diff()], output)
    assert temporary_paths[0] is not None
    assert not os.path.exists(temporary_paths[0])
    assert not (tmp_path / "source.nolayers.pdf").exists()
    with fitz.open(output) as doc:
        if failures == 1:
            assert "Source page 1" in doc[0].get_text()
        assert "Legend" in doc[0].get_text()


def test_legend_fits_small_rotated_page(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=80, height=100)
        page.set_rotation(90)
        doc.save(source)
    overlay.write_overlay(source, [_diff()], output)
    with fitz.open(output) as doc:
        page = doc[0]
        for label in ("Legend", "Added", "Removed", "Changed", "Moved"):
            rects = [fitz.Rect(word[:4]) for word in page.get_text("words") if word[4] == label]
            assert len(rects) == 1
            assert page.rect.contains(rects[0] * page.rotation_matrix)


def test_cropped_rotated_page_preserves_coordinates_in_fallback(tmp_path, monkeypatch):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=500)
        page.set_cropbox(fitz.Rect(100, 80, 500, 380))
        page.draw_rect(fitz.Rect(250, 200, 290, 250), fill=(0.8, 0.8, 0.8))
        page.set_rotation(270)
        display_box = fitz.Rect(250, 200, 290, 250) * page.rotation_matrix
        doc.save(source)
    diff = _diff(coordinate_space="display")
    diff["geometry"]["added"] = [tuple(display_box)]
    draw, calls = overlay._draw_overlay_rects, 0

    def fail_first(page, diff):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Force raster recovery")
        draw(page, diff)

    monkeypatch.setattr(overlay, "_draw_overlay_rects", fail_first)
    monkeypatch.setattr(overlay, "_try_strip_layers_with_pikepdf", lambda path: None)
    overlay.write_overlay(source, [diff], output)
    with fitz.open(output) as doc:
        page = doc[0]
        assert page.rect == fitz.Rect(0, 0, 300, 400)
        x, y = int((display_box.x0 + display_box.x1) / 2), int((display_box.y0 + display_box.y1) / 2)
        red, green, blue = page.get_pixmap().pixel(x, y)
        assert 165 < red < 180 and red == blue and green > red


def test_legend_does_not_fade_changed_revision_text(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((30, 70), "REV B", fontsize=18)
        text_box = page.search_for("REV B")[0]
        doc.save(source)
    diff = _diff()
    diff["text"]["added"] = [{"bbox": tuple(text_box), "text": "REV B"}]
    with fitz.open(source) as expected:
        overlay._draw_overlay_rects(expected[0], diff)
        pixels_without_legend = expected[0].get_pixmap(clip=text_box).samples

    overlay.write_overlay(source, [diff], output)

    with fitz.open(output) as doc:
        assert len(doc) == 1
        assert doc[0].rect == fitz.Rect(0, 0, 300, 200)
        assert doc[0].get_pixmap(clip=text_box).samples == pixels_without_legend
        assert "Legend" in doc[0].get_text()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_dense_page_uses_legend_comment_without_covering_content(tmp_path, rotation):
    source, output = tmp_path / "source.pdf", tmp_path / "out.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.draw_rect(page.rect, fill=(0.7, 0.7, 0.7))
        page.insert_text((30, 70), "Drawing fills the page", fontsize=18)
        page.set_rotation(rotation)
        before = page.get_pixmap().samples
        bounds = page.rect
        doc.save(source)

    overlay.write_overlay(source, [_diff()], output)

    with fitz.open(output) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert page.rect == bounds
        assert page.get_pixmap(annots=False).samples == before
        notes = list(page.annots())
        assert len(notes) == 1
        assert "Green: Added" in notes[0].info["content"]
        assert "Purple: Changed" in notes[0].info["content"]
        assert page.rect.contains(notes[0].rect * page.rotation_matrix)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_raster_comparison_highlights_actual_rotated_change(tmp_path, rotation):
    from pdf_compare.raster_grid import raster_grid_changed_boxes

    old, new, output = (tmp_path / name for name in ("old.pdf", "new.pdf", "out.pdf"))
    for path in (old, new):
        with fitz.open() as doc:
            page = doc.new_page(width=320, height=240)
            if path == new:
                page.draw_rect(fitz.Rect(180, 150, 210, 180), fill=(0.5, 0.5, 0.5))
            page.set_rotation(rotation)
            center = fitz.Point(195, 165) * page.rotation_matrix
            doc.save(path)
    boxes, metrics = raster_grid_changed_boxes(
        str(old), str(new), 0, dpi=72, rows=24, cols=32,
        method="abs", cell_change_ratio=0.1, return_metrics=True,
    )
    assert any(fitz.Rect(box).contains(center) for box in boxes)
    diff = _diff(coordinate_space=metrics["coordinate_space"])
    diff["geometry"]["changed"] = boxes

    overlay.write_overlay(new, [diff], output)

    with fitz.open(output) as doc:
        red, green, blue = doc[0].get_pixmap().pixel(int(center.x), int(center.y))
        assert blue > red > green
        assert green < 120
