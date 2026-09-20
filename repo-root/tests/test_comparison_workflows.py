"""Regression coverage for complete file comparison and CLI output."""

import fitz
import pytest
from typer.testing import CliRunner

from pdf_compare.cli import app
from pdf_compare.comparison import diff_pdf_files


def make_pdf(path, page_count):
    with fitz.open() as doc:
        for index in range(page_count):
            page = doc.new_page(width=200, height=160)
            if index == 0:
                page.insert_text((20, 60), "UNCHANGED")
        doc.save(path)
    return str(path)


@pytest.mark.parametrize("old_count,new_count,status", [(1, 2, "added"), (2, 1, "removed")])
def test_visual_compare_includes_blank_extra_pages(tmp_path, old_count, new_count, status):
    old = make_pdf(tmp_path / "old.pdf", old_count)
    new = make_pdf(tmp_path / "new.pdf", new_count)

    diffs = diff_pdf_files(old, new, dpi=72)

    assert len(diffs) == 2
    assert diffs[0]["geometry"]["changed"] == []
    assert diffs[1]["page_status"] == status
    assert diffs[1]["geometry"][status] == [(0, 0, 200, 160)]


@pytest.mark.parametrize("use_old_base", [False, True])
@pytest.mark.parametrize("old_count,new_count", [(1, 2), (2, 1)])
def test_grid_cli_writes_complete_overlay(tmp_path, use_old_base, old_count, new_count):
    old = make_pdf(tmp_path / "old.pdf", old_count)
    new = make_pdf(tmp_path / "new.pdf", new_count)
    output = tmp_path / "overlay.pdf"
    args = ["compare-grid", old, new, "--out-overlay", str(output), "--grid-dpi", "72"]
    if use_old_base:
        args += ["--base-pdf", old]

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    with fitz.open(output) as overlay:
        assert len(overlay) == 2
        assert "UNCHANGED" in overlay[0].get_text()
        assert overlay[1].get_drawings(), "Even a blank added/removed page must be highlighted"


def test_grid_cli_rejects_invalid_grid(tmp_path):
    old = make_pdf(tmp_path / "old.pdf", 1)
    result = CliRunner().invoke(app, ["compare-grid", old, old, "--grid-rows", "0"])
    assert result.exit_code == 2


def test_grid_cli_reports_missing_input(tmp_path):
    missing = str(tmp_path / "missing.pdf")
    result = CliRunner().invoke(app, ["compare-grid", missing, missing])
    assert result.exit_code == 1
    assert "Comparison failed" in result.output


@pytest.mark.parametrize("size", [(612, 792), (2592, 3456)])
def test_default_visual_comparison_detects_single_character_on_large_sheets(tmp_path, size):
    paths = [tmp_path / "old.pdf", tmp_path / "new.pdf"]
    for path, text in zip(paths, ["REV A", "REV B"]):
        with fitz.open() as pdf:
            page = pdf.new_page(width=size[0], height=size[1])
            page.insert_text((30, 70), text, fontsize=11)
            pdf.save(path)
    diffs = diff_pdf_files(*(str(path) for path in paths))
    assert diffs[0]["geometry"]["changed"], "A changed character must not disappear in page whitespace"


def test_grid_cli_rejects_unrelated_base_and_preserves_sources(tmp_path):
    old = make_pdf(tmp_path / "old.pdf", 1)
    new = make_pdf(tmp_path / "new.pdf", 2)
    unrelated = make_pdf(tmp_path / "unrelated.pdf", 1)
    original = (tmp_path / "new.pdf").read_bytes()
    result = CliRunner().invoke(app, [
        "compare-grid", old, new, "--base-pdf", unrelated, "--out-overlay", new,
    ])
    assert result.exit_code == 1
    assert "base must be the old or new" in result.output
    assert (tmp_path / "new.pdf").read_bytes() == original


@pytest.mark.parametrize("output_old", [True, False])
def test_grid_cli_cannot_overwrite_either_source(tmp_path, output_old):
    old = make_pdf(tmp_path / "old.pdf", 1)
    new = make_pdf(tmp_path / "new.pdf", 2)
    originals = [(tmp_path / name).read_bytes() for name in ("old.pdf", "new.pdf")]
    result = CliRunner().invoke(app, [
        "compare-grid", old, new, "--out-overlay", old if output_old else new, "--grid-dpi", "72",
    ])
    assert result.exit_code == 1
    assert "different from the source" in result.output
    assert [(tmp_path / name).read_bytes() for name in ("old.pdf", "new.pdf")] == originals
