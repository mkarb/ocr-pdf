"""Check layout highlights against the displayed coordinates of saved PDFs."""

import fitz
import pytest

from pdf_compare import overlay


ORANGE = (1, 0.5, 0)
PURPLE = (0.5, 0, 0.8)


def _layout_diff(**metadata):
    return {
        "page": 1,
        "coordinate_space": "display",
        "layout": {kind: [] for kind in ("added", "removed", "moved", "resized", "changed")},
        **metadata,
    }


def _block_pdf(path, kind, box, rotation=0):
    with fitz.open() as doc:
        page = doc.new_page(width=420, height=320)
        if kind == "text":
            page.insert_text((box[0], box[1]), "A layout block", fontsize=12)
            bounds = page.search_for("A layout block")[0]
        else:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
            pix.clear_with(210)
            page.insert_image(fitz.Rect(box), pixmap=pix, keep_proportion=False)
            bounds = page.get_image_info()[0]["bbox"]
        page.set_rotation(rotation)
        displayed = fitz.Rect(bounds) * page.rotation_matrix
        doc.save(path)
    return tuple(displayed)


def _outlines(page, color):
    return [drawing for drawing in page.get_drawings()
            if drawing["fill"] is None and drawing["color"] == pytest.approx(color)]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("kind", ["text", "image"])
def test_moved_block_boxes_follow_actual_rotated_content(tmp_path, rotation, kind):
    old, new, output = (tmp_path / name for name in ("old.pdf", "new.pdf", "overlay.pdf"))
    previous = _block_pdf(old, kind, (70, 140, 140, 185), rotation)
    current = _block_pdf(new, kind, (220, 210, 290, 255), rotation)
    diff = _layout_diff()
    diff["layout"]["moved"] = [{"kind": kind, "from": previous, "to": current}]

    overlay.write_overlay(new, [diff], output)

    with fitz.open(new) as source, fitz.open(output) as result:
        page = result[0]
        assert page.rotation == rotation
        assert page.rect == source[0].rect
        outlines = _outlines(page, ORANGE)
        assert len(outlines) == 2
        for drawing, expected in zip(outlines, (previous, current)):
            assert tuple(drawing["rect"] * page.rotation_matrix) == pytest.approx(expected)
        assert outlines[0]["dashes"] == "[] 0"
        assert outlines[1]["dashes"] == "[ 2 2 ] 0"
        assert "Resized" in page.get_text()
        assert "Changed" not in page.get_text()


@pytest.mark.parametrize("also_moved", [False, True])
def test_resized_image_boxes_show_previous_and_new_extents(tmp_path, also_moved):
    old, new, output = (tmp_path / name for name in ("old.pdf", "new.pdf", "overlay.pdf"))
    previous = _block_pdf(old, "image", (100, 140, 170, 185))
    new_box = (180, 170, 300, 255) if also_moved else (100, 140, 220, 225)
    current = _block_pdf(new, "image", new_box)
    diff = _layout_diff()
    pair = {"kind": "image", "from": previous, "to": current}
    diff["layout"]["resized"] = [pair]
    if also_moved:
        diff["layout"]["moved"] = [pair]

    overlay.write_overlay(new, [diff], output)

    with fitz.open(output) as result:
        purple = _outlines(result[0], PURPLE)
        assert len(purple) == 2
        assert tuple(purple[0]["rect"]) == pytest.approx(previous)
        assert tuple(purple[1]["rect"]) == pytest.approx(current)
        assert purple[1]["dashes"] == "[ 2 2 ] 0"
        if also_moved:
            orange = _outlines(result[0], ORANGE)
            assert len(orange) == 2
            assert orange[0]["width"] > purple[0]["width"]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_page_size_or_rotation_change_has_displayed_border(tmp_path, rotation):
    source, output = tmp_path / "source.pdf", tmp_path / "overlay.pdf"
    _block_pdf(source, "text", (150, 150, 230, 180), rotation)
    diff = _layout_diff(layout_page_changed=True)

    overlay.write_overlay(source, [diff], output)

    with fitz.open(output) as result:
        page = result[0]
        borders = _outlines(page, PURPLE)
        assert len(borders) == 1
        assert tuple(borders[0]["rect"] * page.rotation_matrix) == pytest.approx(
            tuple(page.rect + (2, 2, -2, -2)))
        assert borders[0]["width"] == 2


def test_added_and_removed_layout_blocks_use_transparent_colors(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "overlay.pdf"
    _block_pdf(source, "text", (130, 130, 200, 180))
    diff = _layout_diff()
    diff["layout"]["added"] = [{"kind": "drawing", "bbox": (80, 200, 130, 250)}]
    diff["layout"]["removed"] = [{"kind": "image", "bbox": (240, 200, 290, 250)}]

    overlay.write_overlay(source, [diff], output)

    with fitz.open(output) as result:
        added, removed = result[0].get_drawings()[:2]
        assert added["fill"] == pytest.approx((0, 1, 0))
        assert removed["fill"] == pytest.approx((1, 0, 0))
        assert added["fill_opacity"] == pytest.approx(0.1)
        assert removed["fill_opacity"] == pytest.approx(0.1)


def test_layout_legend_preserves_dense_content_and_explains_page_border(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "overlay.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.draw_rect(page.rect, fill=(0.7, 0.7, 0.7))
        page.insert_text((30, 70), "Layout fills the page", fontsize=18)
        doc.save(source)
    diff = _layout_diff(layout_page_changed=True)
    diff["layout"]["moved"] = [{"kind": "text", "from": (30, 40, 130, 80),
                               "to": (50, 120, 150, 160)}]
    with fitz.open(source) as expected:
        overlay._draw_overlay_rects(expected[0], diff)
        pixels_without_legend = expected[0].get_pixmap(annots=False).samples

    overlay.write_overlay(source, [diff], output)

    with fitz.open(output) as result:
        page = result[0]
        assert page.get_pixmap(annots=False).samples == pixels_without_legend
        notes = list(page.annots())
        assert len(notes) == 1
        description = notes[0].info["content"]
        assert "Purple: Resized" in description
        assert "Dashed: New position/size" in description
        assert "Page size or rotation changed" in description


@pytest.mark.parametrize("status", ["added", "removed"])
def test_extra_layout_page_preserves_page_status(tmp_path, status):
    base, alternate, output = (tmp_path / name for name in ("base.pdf", "alternate.pdf", "overlay.pdf"))
    _block_pdf(base, "text", (130, 130, 200, 180))
    with fitz.open(base) as doc:
        page = doc.new_page(width=420, height=320)
        page.insert_text((130, 130), "Extra layout page")
        doc.save(alternate)
    diff = _layout_diff(page=2, page_status=status)

    overlay.write_overlay(base, [diff], output, alternate_pdf_path=alternate)

    with fitz.open(output) as result:
        assert len(result) == 2
        assert "Extra layout page" in result[1].get_text()
        assert f"Page {status}" in result[1].get_text()
        assert "Resized" in result[1].get_text()
