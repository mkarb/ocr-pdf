"""Layout extraction regressions using real native and scanned PDF fixtures."""

from io import BytesIO

import fitz
from PIL import Image, ImageDraw
import pytest

from pdf_compare.layout_compare import diff_layout_files


def write_pdf(path, draw, *, rotation=0, size=(400, 400)):
    with fitz.open() as document:
        page = document.new_page(width=size[0], height=size[1])
        draw(page)
        page.set_rotation(rotation)
        document.save(path)
    return path


def compare(tmp_path, old_draw, new_draw, **kwargs):
    old = write_pdf(tmp_path / "old.pdf", old_draw)
    new = write_pdf(tmp_path / "new.pdf", new_draw)
    return diff_layout_files(old, new, **kwargs)[0]


def png(rect=(30, 30, 90, 70), *, changed_inside=False, dimensions=(400, 400)):
    image = Image.new("RGB", dimensions, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, fill="black")
    if changed_inside:
        draw.rectangle((rect[0] + 10, rect[1] + 10, rect[2] - 10, rect[3] - 10), fill="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def scan(data, *, add_ocr=False):
    def draw(page):
        page.insert_image(page.rect, stream=data)
        if add_ocr:
            page.insert_text((30, 60), "hidden OCR", render_mode=3)
    return draw


def test_repeated_text_preserves_stationary_occurrence(tmp_path):
    result = compare(
        tmp_path,
        lambda p: [p.insert_text((40, y), "Valve") for y in (40, 140)],
        lambda p: [p.insert_text((40, y), "Valve") for y in (140, 270)],
    )["layout"]
    assert len(result["moved"]) == 1
    assert result["moved"][0]["from"][1] < 40
    assert result["moved"][0]["to"][1] > 250
    assert result["added"] == result["removed"] == result["resized"] == []


def test_removed_duplicate_does_not_steal_jittered_stationary_text(tmp_path):
    result = compare(
        tmp_path,
        lambda p: [p.insert_text((40, y), "Valve") for y in (40, 140)],
        lambda p: p.insert_text((41, 141), "Valve"),
    )["layout"]
    assert len(result["removed"]) == 1
    assert result["removed"][0]["bbox"][1] < 40
    assert result["moved"] == result["added"] == []


def test_unrelated_distant_replacement_is_not_a_move(tmp_path):
    result = compare(
        tmp_path, lambda p: p.insert_text((30, 30), "First label"),
        lambda p: p.insert_text((250, 300), "Second label"),
    )["layout"]
    assert len(result["added"]) == len(result["removed"]) == 1
    assert result["moved"] == result["resized"] == []


def test_same_text_can_move_across_page_without_overlap(tmp_path):
    result = compare(
        tmp_path, lambda p: p.insert_text((30, 30), "Same label"),
        lambda p: p.insert_text((250, 300), "Same label"),
    )["layout"]
    assert len(result["moved"]) == 1
    assert result["added"] == result["removed"] == []


def test_image_resizing_and_movement_are_independent(tmp_path):
    data = png()
    result = compare(
        tmp_path,
        lambda p: p.insert_image((30, 30, 100, 100), stream=data),
        lambda p: p.insert_image((70, 50, 180, 160), stream=data),
    )["layout"]
    assert len(result["moved"]) == len(result["resized"]) == 1
    assert result["moved"][0]["kind"] == "image"
    assert result["moved"][0]["from"] == (30, 30, 100, 100)
    assert result["moved"][0]["to"] == (70, 50, 180, 160)


def test_changed_image_content_at_fixed_bounds_is_ignored(tmp_path):
    result = compare(
        tmp_path,
        lambda p: p.insert_image((30, 30, 100, 100), stream=png()),
        lambda p: p.insert_image((30, 30, 100, 100), stream=png(changed_inside=True)),
    )["layout"]
    assert not any(result.values())


@pytest.mark.parametrize("vertical", [False, True])
def test_shortened_line_has_visible_resize_bounds(tmp_path, vertical):
    a, b, c = ((40, 40), (40, 200), (40, 100)) if vertical else (
        (40, 40), (200, 40), (100, 40))
    result = compare(tmp_path, lambda p: p.draw_line(a, b),
                     lambda p: p.draw_line(a, c))["layout"]
    assert len(result["resized"]) == 1
    old_box = result["resized"][0]["from"]
    assert old_box[2] > old_box[0] and old_box[3] > old_box[1]
    assert result["moved"] == []


@pytest.mark.parametrize("ocr", [False, True])
def test_scan_region_moves_even_with_hidden_ocr_text(tmp_path, ocr):
    result = compare(
        tmp_path, scan(png(), add_ocr=ocr),
        scan(png((200, 230, 260, 270)), add_ocr=ocr),
    )
    assert result["layout_source"] == "raster"
    assert len(result["layout"]["moved"]) == 1
    assert result["layout"]["moved"][0]["kind"] == "region"
    assert result["layout"]["added"] == result["layout"]["removed"] == []


def test_scan_internal_content_edits_at_same_bounds_are_ignored(tmp_path):
    result = compare(tmp_path, scan(png()), scan(png(changed_inside=True)))
    assert result["layout_source"] == "raster"
    assert not any(result["layout"].values())


def test_scan_separate_added_region_is_detected(tmp_path):
    image = Image.open(BytesIO(png()))
    ImageDraw.Draw(image).rectangle((230, 240, 270, 280), fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    result = compare(tmp_path, scan(png()), scan(stream.getvalue()))["layout"]
    assert len(result["added"]) == 1
    assert result["added"][0]["bbox"] == pytest.approx((230, 240, 271, 281), abs=1)
    assert result["moved"] == result["removed"] == []


def test_native_to_scan_keeps_same_layout(tmp_path):
    result = compare(
        tmp_path,
        lambda p: p.draw_rect((30, 30, 90, 70), fill=(0, 0, 0), color=None),
        scan(png()),
    )
    assert result["layout_source"] == "raster"
    assert not any(result["layout"].values())


def test_rotated_page_bounds_are_display_coordinates(tmp_path):
    old = write_pdf(tmp_path / "old.pdf", lambda p: p.draw_rect((30, 40, 100, 80)),
                    rotation=90, size=(400, 300))
    new = write_pdf(tmp_path / "new.pdf", lambda p: p.draw_rect((50, 40, 120, 80)),
                    rotation=90, size=(400, 300))
    result = diff_layout_files(old, new)[0]
    assert result["coordinate_space"] == "display"
    item = result["layout"]["moved"][0]
    assert item["from"] == (220, 30, 260, 100)
    assert item["to"] == (220, 50, 260, 120)


def test_layout_ignores_color_and_stroke_style_edits(tmp_path):
    result = compare(
        tmp_path, lambda p: p.draw_rect((30, 40, 100, 80), color=(1, 0, 0)),
        lambda p: p.draw_rect((30, 40, 100, 80), color=(0, 0, 1), width=2),
    )["layout"]
    assert not any(result.values())


def test_content_outside_cropbox_is_excluded(tmp_path):
    def draw(page, offset):
        page.draw_rect((10 + offset, 10, 40 + offset, 40))
        page.draw_rect((100, 100, 180, 180))
        page.set_cropbox((80, 80, 300, 300))
    result = compare(tmp_path, lambda p: draw(p, 0), lambda p: draw(p, 10))
    assert not any(result["layout"].values())


@pytest.mark.parametrize("render_mode,opacity", [(3, 1), (0, 0)])
def test_invisible_native_text_does_not_change_visible_blocks(tmp_path, render_mode, opacity):
    def draw(page, hidden_x):
        page.insert_text((30, 40), "Visible text")
        page.insert_text((hidden_x, 40), "hidden text", render_mode=render_mode,
                         fill_opacity=opacity)
    result = compare(tmp_path, lambda p: draw(p, 90), lambda p: draw(p, 180))
    assert result["layout_source"] == "native"
    assert not any(result["layout"].values())


def test_visible_text_overlapping_hidden_text_remains_comparable(tmp_path):
    def draw(page, x):
        page.insert_text((x, 40), "Visible text")
        page.insert_text((x, 40), "Visible text", render_mode=3)
    result = compare(tmp_path, lambda p: draw(p, 30), lambda p: draw(p, 60))
    assert len(result["layout"]["moved"]) == 1


def test_hidden_text_at_visible_origin_does_not_enlarge_block(tmp_path):
    def draw(page, hidden):
        page.insert_text((30, 40), "Visible")
        page.insert_text((30, 40), hidden, render_mode=3)
    result = compare(tmp_path, lambda p: draw(p, "Invisible short"),
                     lambda p: draw(p, "Invisible text that takes much more space"))
    assert not any(result["layout"].values())


def test_fully_transparent_paths_do_not_count_as_layout(tmp_path):
    result = compare(
        tmp_path,
        lambda p: p.draw_rect((30, 40, 100, 80), stroke_opacity=0, fill_opacity=0, fill=(0, 0, 0)),
        lambda p: p.draw_rect((60, 40, 130, 80), stroke_opacity=0, fill_opacity=0, fill=(0, 0, 0)),
    )
    assert not any(result["layout"].values())


def test_tiled_scan_uses_visible_regions(tmp_path):
    def draw(page, rectangle):
        page.insert_image((0, 0, 200, 400), stream=png(rectangle, dimensions=(200, 400)))
        page.insert_image((200, 0, 400, 400), stream=png((30, 250, 90, 290), dimensions=(200, 400)))
    result = compare(tmp_path, lambda p: draw(p, (30, 30, 90, 70)),
                     lambda p: draw(p, (100, 30, 160, 70)))
    assert result["layout_source"] == "raster"
    assert len(result["layout"]["moved"]) == 1
    assert result["layout"]["moved"][0]["to"][0] == pytest.approx(100, abs=1)


def test_repeated_overlapping_images_are_not_mistaken_for_scan(tmp_path):
    def draw(page):
        for _ in range(4):
            page.insert_image((0, 0, 200, 400), stream=png(dimensions=(200, 400)))
    result = compare(tmp_path, draw, draw)
    assert result["layout_source"] == "native"
    assert not any(result["layout"].values())


def test_large_sheet_thin_strokes_survive_native_to_scan_conversion(tmp_path):
    def draw(page):
        page.draw_rect((20, 20, 2428, 1564), width=2)
        page.draw_line((260, 406), (294, 406), width=0.7)
        page.insert_text((300, 410), "VALVE-221", fontsize=10)
    old = write_pdf(tmp_path / "old.pdf", draw, size=(2448, 1584))
    with fitz.open(old) as document:
        data = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    new = write_pdf(tmp_path / "new.pdf", scan(data), size=(2448, 1584))
    result = diff_layout_files(old, new)[0]
    assert result["layout_source"] == "raster"
    assert not any(result["layout"].values())
