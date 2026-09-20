"""Opt-in PostgreSQL, CLI and Streamlit comparison workflow regressions.

Set PDF_COMPARE_TEST_DATABASE_URL to a disposable PostgreSQL database. Only the
uniquely named documents created by each test are removed during cleanup.
"""

import os
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from typer.testing import CliRunner

from pdf_compare.cli import app as cli
from pdf_compare.compare_new import diff_documents
from pdf_compare.db_backend import DatabaseBackend
from pdf_compare.pdf_extract import pdf_to_vectormap


@pytest.fixture
def revisions(tmp_path):
    url = os.getenv("PDF_COMPARE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set PDF_COMPARE_TEST_DATABASE_URL to run PostgreSQL workflow tests")
    backend = DatabaseBackend(url)
    prefix = "compare-test-" + uuid4().hex
    ids = [prefix + "-old", prefix + "-new"]
    paths = [tmp_path / "old.pdf", tmp_path / "new.pdf"]
    try:
        for revised, (doc_id, path) in enumerate(zip(ids, paths)):
            with fitz.open() as pdf:
                page = pdf.new_page(width=300, height=200)
                page.draw_line((30, 90), (130 if revised else 230, 90))
                page.insert_text((30, 70), "REV B" if revised else "REV A")
                pdf.new_page(width=300, height=200).insert_text((30, 70), "UNCHANGED")
                if not revised:
                    pdf.new_page(width=240, height=180).insert_text((30, 70), "REMOVED PAGE")
                pdf.save(path)
            backend.upsert_vectormap(pdf_to_vectormap(str(path), doc_id=doc_id, workers=1))
        yield backend, ids, paths, url
    finally:
        for doc_id in ids:
            backend.delete_document(doc_id)
        backend.engine.dispose()


def test_extraction_database_comparison_and_cli_overlay(revisions, tmp_path):
    backend, (old_id, new_id), paths, url = revisions
    diffs = diff_documents(backend, old_id, new_id)
    assert len(diffs) == 3
    assert diffs[0]["geometry"]["added"] and diffs[0]["geometry"]["removed"]
    assert [t["text"] for t in diffs[0]["text"]["added"]] == ["REV B"]
    assert [t["text"] for t in diffs[0]["text"]["removed"]] == ["REV A"]
    assert not any(diffs[1]["geometry"].values())
    assert not any(diffs[1]["text"].values())
    assert diffs[2]["page_status"] == "removed"
    assert all(not any(d["geometry"].values()) and not any(d["text"].values())
               for d in diff_documents(backend, old_id, old_id))

    output = tmp_path / "vector-overlay.pdf"
    result = CliRunner().invoke(cli, [
        "compare", old_id, new_id, "--db-url", url, "--out-overlay", str(output),
    ])
    assert result.exit_code == 0, result.output
    with fitz.open(output) as overlay:
        assert len(overlay) == 3
        assert "REV B" in overlay[0].get_text()
        assert "REMOVED PAGE" in overlay[2].get_text()

    unrelated = tmp_path / "unrelated.pdf"
    unrelated.write_bytes(paths[0].read_bytes())
    original = paths[1].read_bytes()
    result = CliRunner().invoke(cli, [
        "compare", old_id, new_id, "--db-url", url,
        "--base-pdf", str(unrelated), "--out-overlay", str(paths[1]),
    ])
    assert result.exit_code == 1
    assert "base must be the old or new" in result.output
    assert paths[1].read_bytes() == original


def test_streamlit_compare_export_rerun_and_selection_change(revisions, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    _, (old_id, new_id), paths, url = revisions
    ui_dir = Path(__file__).resolve().parents[1] / "ui"
    monkeypatch.syspath_prepend(str(ui_dir))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "ui-data"))
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ui_dir / "streamlit_app.py"), default_timeout=30).run()
        assert not app.exception
        by_label = lambda elements, label: next(e for e in elements if e.label == label)
        by_label(app.selectbox, "Old document (baseline)").select(f"{old_id} - {paths[0].name}")
        by_label(app.selectbox, "New document (revised)").select(f"{new_id} - {paths[1].name}")
        by_label(app.selectbox, "Comparison method").select("Vector and text").run()
        by_label(app.button, "Compare documents").click().run()
        assert not app.exception
        assert not app.error
        assert len(app.session_state["user_last_diffs"]) == 3
        by_label(app.button, "Create overlay PDF").click().run()
        assert not app.error
        assert app.session_state["user_last_overlay"][1].startswith(b"%PDF")
        app.run()
        assert len(app.get("download_button")) >= 1
        assert any("page status" in table.value for table in app.dataframe)

        by_label(app.selectbox, "Comparison method").select("Visual (raster)").run()
        assert app.session_state["user_last_diffs"] == []
        assert app.session_state["user_last_overlay"] is None
        by_label(app.button, "Compare documents").click().run()
        assert not app.exception
        assert not app.error
        assert len(app.session_state["user_last_diffs"]) == 3
        assert app.session_state["user_last_diffs"][0]["geometry"]["changed"]
        by_label(app.button, "Create overlay PDF").click().run()
        assert not app.error
        assert app.session_state["user_last_overlay"][1].startswith(b"%PDF")

        by_label(app.number_input, "Minimum changed area per grid cell (%)").set_value(1.0).run()
        assert app.session_state["user_last_diffs"] == []
        assert app.session_state["user_last_overlay"] is None

        by_label(app.selectbox, "Comparison method").select("Layout").run()
        by_label(app.button, "Compare documents").click().run()
        assert not app.exception
        assert not app.error
        diffs = app.session_state["user_last_diffs"]
        assert len(diffs) == 3
        assert diffs[0]["layout"]["resized"]
        assert any("moved blocks" in table.value for table in app.dataframe)
        by_label(app.button, "Create overlay PDF").click().run()
        assert not app.error
        filename, content = app.session_state["user_last_overlay"]
        assert filename.startswith("layout_overlay_")
        with fitz.open(stream=content, filetype="pdf") as overlay:
            assert len(overlay) == 3
            assert "REMOVED PAGE" in overlay[2].get_text()
        app.run()
        assert app.session_state["user_last_overlay"][1] == content
        by_label(app.number_input, "Position tolerance (points)").set_value(5.0).run()
        assert app.session_state["user_last_diffs"] == []
        assert app.session_state["user_last_overlay"] is None

        by_label(app.selectbox, "New document (revised)").select(f"{old_id} - {paths[0].name}").run()
        assert app.session_state["user_last_diffs"] == []
        assert not any(b.label == "Create overlay PDF" for b in app.button)
    finally:
        st.cache_resource.clear()
