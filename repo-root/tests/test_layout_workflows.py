"""Exercise layout comparison through PDF extraction and saved CLI overlays."""

import fitz
import pytest
from typer.testing import CliRunner

from pdf_compare.cli import app
from pdf_compare.layout_compare import diff_layout_files


def _make_pdf(path, *, text="ALPHA", text_x=35, rectangle=None,
              pages=1, size=(420, 300), rotation=0):
    with fitz.open() as doc:
        for number in range(pages):
            page = doc.new_page(width=size[0], height=size[1])
            if number == 0:
                if text:
                    page.insert_text((text_x, 145), text, fontname="cour", fontsize=12)
                if rectangle:
                    page.draw_rect(fitz.Rect(rectangle))
            page.set_rotation(rotation)
        doc.save(path)
    return str(path)


def _assert_unchanged(diff):
    assert not diff.get("page_status")
    assert not diff.get("layout_page_changed")
    assert all(not changes for changes in diff["layout"].values())


def test_layout_ignores_rewording_with_identical_block_bounds(tmp_path):
    old = _make_pdf(tmp_path / "old.pdf", text="ALPHA")
    new = _make_pdf(tmp_path / "new.pdf", text="BRAVO")
    with fitz.open(old) as before, fitz.open(new) as after:
        assert before[0].get_text("blocks")[0][:4] == after[0].get_text("blocks")[0][:4]
        assert before[0].get_text() != after[0].get_text()

    diffs = diff_layout_files(old, new)

    assert len(diffs) == 1
    _assert_unchanged(diffs[0])


def test_layout_movement_and_size_tolerances_are_independent(tmp_path):
    old = _make_pdf(tmp_path / "old.pdf", rectangle=(230, 130, 285, 170))
    new = _make_pdf(tmp_path / "new.pdf", text_x=50, rectangle=(230, 130, 305, 170))

    diff = diff_layout_files(old, new)[0]
    assert len(diff["layout"]["moved"]) == 1
    assert diff["layout"]["moved"][0]["kind"] == "text"
    assert diff["layout"]["moved"][0]["to"][0] - diff["layout"]["moved"][0]["from"][0] == 15
    assert len(diff["layout"]["resized"]) == 1
    assert diff["layout"]["resized"][0]["kind"] == "drawing"
    assert not diff["layout"]["added"]
    assert not diff["layout"]["removed"]

    tolerate_movement = diff_layout_files(old, new, position_tolerance=15)[0]
    assert not tolerate_movement["layout"]["moved"]
    assert tolerate_movement["layout"]["resized"]
    tolerate_size = diff_layout_files(old, new, size_tolerance=20)[0]
    assert tolerate_size["layout"]["moved"]
    assert not tolerate_size["layout"]["resized"]
    _assert_unchanged(diff_layout_files(old, new, position_tolerance=15, size_tolerance=20)[0])


@pytest.mark.parametrize("size,rotation", [((440, 300), 0), ((420, 300), 90)])
def test_layout_detects_blank_page_size_and_rotation_changes(tmp_path, size, rotation):
    old = _make_pdf(tmp_path / "old.pdf", text=None)
    new = _make_pdf(tmp_path / "new.pdf", text=None, size=size, rotation=rotation)

    diff = diff_layout_files(old, new)[0]

    assert diff["layout_page_changed"] is True
    assert diff["old_rotation"] == 0
    assert diff["new_rotation"] == rotation
    if size != (420, 300):
        assert diff["old_page_size"] != diff["new_page_size"]


@pytest.mark.parametrize("option", ["position_tolerance", "size_tolerance"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_layout_rejects_invalid_tolerances(tmp_path, option, value):
    source = _make_pdf(tmp_path / "source.pdf")
    with pytest.raises(ValueError, match="tolerance"):
        diff_layout_files(source, source, **{option: value})


@pytest.mark.parametrize("base", ["old", "new"])
@pytest.mark.parametrize("old_count,new_count,status", [(1, 2, "added"), (2, 1, "removed")])
def test_layout_cli_preserves_and_highlights_blank_extra_pages(tmp_path, base, old_count, new_count, status):
    old = _make_pdf(tmp_path / "old.pdf", pages=old_count)
    new = _make_pdf(tmp_path / "new.pdf", pages=new_count)
    output = tmp_path / "nested" / "layout.pdf"
    diffs = diff_layout_files(old, new)
    assert len(diffs) == 2
    _assert_unchanged(diffs[0])
    assert diffs[1]["page_status"] == status

    result = CliRunner().invoke(app, [
        "compare-layout", old, new, "--out-overlay", str(output),
        "--base-pdf", old if base == "old" else new,
    ])

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"%PDF-")
    with fitz.open(output) as overlay:
        assert len(overlay) == 2
        assert "ALPHA" in overlay[0].get_text()
        assert f"Page {status}" in overlay[1].get_text()
        color = (0, 0.7, 0) if status == "added" else (1, 0, 0)
        assert any(item["color"] == pytest.approx(color) for item in overlay[1].get_drawings())


@pytest.mark.parametrize("rotation", [0, 90, 270])
@pytest.mark.parametrize("base", ["old", "new"])
def test_layout_cli_saves_movement_and_resize_highlights(tmp_path, base, rotation):
    old_rect, new_rect = (230, 130, 285, 170), (230, 130, 305, 170)
    old = _make_pdf(tmp_path / "old.pdf", text="ALPHA", rectangle=old_rect, rotation=rotation)
    new = _make_pdf(tmp_path / "new.pdf", text="BRAVO", text_x=50, rectangle=new_rect, rotation=rotation)
    output = tmp_path / "layout.pdf"
    with fitz.open(old) as before, fitz.open(new) as after:
        text_boxes = [page.get_text("blocks")[0][:4] for page in (before[0], after[0])]

    result = CliRunner().invoke(app, [
        "compare-layout", old, new, "--out-overlay", str(output),
        "--base-pdf", old if base == "old" else new,
    ])

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"%PDF-")
    with fitz.open(output) as overlay:
        page = overlay[0]
        assert page.rotation == rotation
        assert ("ALPHA" if base == "old" else "BRAVO") in page.get_text()
        drawings = page.get_drawings()
        for bounds in text_boxes:
            assert any(
                item["color"] == pytest.approx((1, 0.5, 0))
                and tuple(item["rect"]) == pytest.approx(bounds, abs=0.01)
                for item in drawings
            ), f"Missing orange movement highlight at {bounds}"
        for bounds in (old_rect, new_rect):
            assert any(
                item["color"] == pytest.approx((0.5, 0, 0.8))
                and tuple(item["rect"]) == pytest.approx(bounds, abs=0.01)
                for item in drawings
            ), f"Missing purple resize highlight at {bounds}"


def test_layout_cli_applies_tolerance_options_to_saved_overlay(tmp_path):
    old = _make_pdf(tmp_path / "old.pdf", rectangle=(230, 130, 285, 170))
    new = _make_pdf(tmp_path / "new.pdf", text_x=50, rectangle=(230, 130, 305, 170))
    output = tmp_path / "layout.pdf"

    result = CliRunner().invoke(app, [
        "compare-layout", old, new, "--out-overlay", str(output),
        "--position-tolerance", "15", "--size-tolerance", "20",
    ])

    assert result.exit_code == 0, result.output
    content_area = fitz.Rect(25, 120, 325, 185)
    with fitz.open(output) as overlay:
        for item in overlay[0].get_drawings():
            if item["rect"].intersects(content_area):
                assert item["color"] != pytest.approx((1, 0.5, 0))
                assert item["color"] != pytest.approx((0.5, 0, 0.8))


@pytest.mark.parametrize("option,value", [
    ("--position-tolerance", "-1"), ("--size-tolerance", "-1"),
    ("--position-tolerance", "nan"), ("--size-tolerance", "inf"),
])
def test_layout_cli_rejects_invalid_tolerances_without_output(tmp_path, option, value):
    source = _make_pdf(tmp_path / "source.pdf")
    output = tmp_path / "layout.pdf"
    result = CliRunner().invoke(app, [
        "compare-layout", source, source, "--out-overlay", str(output), option, value,
    ])
    assert result.exit_code != 0
    assert "tolerance" in result.output.lower()
    assert not output.exists()


def test_layout_cli_rejects_unrelated_base(tmp_path):
    old = _make_pdf(tmp_path / "old.pdf")
    new = _make_pdf(tmp_path / "new.pdf", text_x=50)
    unrelated = _make_pdf(tmp_path / "unrelated.pdf")
    output = tmp_path / "layout.pdf"
    result = CliRunner().invoke(app, [
        "compare-layout", old, new, "--base-pdf", unrelated, "--out-overlay", str(output),
    ])
    assert result.exit_code != 0
    assert "base" in result.output.lower()
    assert not output.exists()


@pytest.mark.parametrize("output_old", [True, False])
def test_layout_cli_cannot_overwrite_either_source(tmp_path, output_old):
    old_path, new_path = tmp_path / "old.pdf", tmp_path / "new.pdf"
    old = _make_pdf(old_path)
    new = _make_pdf(new_path, text_x=50)
    originals = [path.read_bytes() for path in (old_path, new_path)]

    result = CliRunner().invoke(app, [
        "compare-layout", old, new, "--out-overlay", old if output_old else new,
    ])

    assert result.exit_code == 1
    assert "different from the source" in result.output
    assert [path.read_bytes() for path in (old_path, new_path)] == originals
