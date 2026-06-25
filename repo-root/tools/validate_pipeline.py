#!/usr/bin/env python3
"""
End-to-end pipeline validation on real infrastructure.

Run this on a machine with a PostgreSQL DATABASE_URL set. It exercises the full
stack the audit touched — extraction, DB round-trip (incl. text provenance),
full-text search, diff, overlay, and optionally OCR + GPU — printing PASS/FAIL
per stage so you can confirm "that shit is working" on your hardware.

Usage:
    export DATABASE_URL=postgresql://user:pass@localhost:5432/pdfcompare   # (or set in env)
    python tools/validate_pipeline.py                # core + DB stages (synthetic PDFs)
    python tools/validate_pipeline.py --ocr          # also run an OCR pass (uses GPU if available)
    python tools/validate_pipeline.py --keep         # don't delete the test docs afterward

Exit code is non-zero if any stage fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable when run as `python tools/validate_pipeline.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402

results: list[tuple[str, bool, str]] = []


def stage(name: str):
    """Decorator-ish context: run a callable, record PASS/FAIL + detail."""
    def run(fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
            print(f"[PASS] {name} {('- ' + str(detail)) if detail else ''}")
            return True
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name} - {type(exc).__name__}: {exc}")
            return False
    return run


def _make_pdf(path: str, *, text: str, rect: tuple[float, float, float, float]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.draw_rect(fitz.Rect(*rect), color=(0, 0, 0), width=1)
    page.insert_text(fitz.Point(rect[0] + 10, rect[1] + 40), text, fontsize=12)
    doc.save(path)
    doc.close()


def _make_image_only_pdf(path: str, text: str) -> None:
    """A page with no native text layer (forces the OCR path)."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=120)
    page.insert_text(fitz.Point(20, 70), text, fontsize=28)
    pix = page.get_pixmap(dpi=200)
    out = fitz.open()
    op = out.new_page(width=page.rect.width, height=page.rect.height)
    op.insert_image(op.rect, pixmap=pix)
    out.save(path)
    doc.close()
    out.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true", help="also run an OCR pass")
    ap.add_argument("--keep", action="store_true", help="keep test documents in the DB")
    args = ap.parse_args()

    # --- Environment report ---
    from pdf_compare.analyzers.highres_ocr import resolve_ocr_engine, HAVE_EASYOCR
    engine, gpu = resolve_ocr_engine()
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    print("=" * 64)
    print("Environment")
    print(f"  resolve_ocr_engine() -> engine={engine}, gpu={gpu}")
    print(f"  EasyOCR installed: {HAVE_EASYOCR}   torch.cuda: {cuda}")
    print(f"  DATABASE_URL set: {bool(os.getenv('DATABASE_URL'))}")
    print("=" * 64)

    tmp = Path(tempfile.mkdtemp(prefix="validate_pipeline_"))
    old_pdf = str(tmp / "old.pdf")
    new_pdf = str(tmp / "new.pdf")
    _make_pdf(old_pdf, text="VALVE V-101", rect=(50, 50, 250, 150))
    _make_pdf(new_pdf, text="VALVE V-102", rect=(60, 60, 260, 160))

    from pdf_compare.pdf_extract import pdf_to_vectormap
    from pdf_compare.db_backend import create_backend
    from pdf_compare.store_new import (
        upsert_vectormap, list_documents, get_vectormap, delete_document,
    )
    from pdf_compare.search_new import search_text
    from pdf_compare.compare_new import diff_documents
    from pdf_compare.overlay import write_overlay

    state: dict = {}

    @stage("extract (geometry + native text + provenance)")
    def _():
        vm = pdf_to_vectormap(old_pdf, workers=1)
        vm_new = pdf_to_vectormap(new_pdf, workers=1)
        assert vm.pages[0].geoms, "no geometry extracted"
        assert any("VALVE" in t.text for t in vm.pages[0].texts), "native text missing"
        assert all(t.source == "native" for t in vm.pages[0].texts), "bad provenance"
        state["vm_old"], state["vm_new"] = vm, vm_new
        return f"old={vm.meta.doc_id[:8]} geoms={len(vm.pages[0].geoms)} texts={len(vm.pages[0].texts)}"

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("\nDATABASE_URL not set - skipping all DB stages.")
        print("Set it and re-run to validate upsert/get/search/diff/overlay.")
        return _summary(skipped_db=True)

    @stage("db connect")
    def _():
        state["backend"] = create_backend(db_url)
        return "connected"

    @stage("upsert (both docs)")
    def _():
        upsert_vectormap(state["backend"], state["vm_old"])
        upsert_vectormap(state["backend"], state["vm_new"])

    @stage("list_documents")
    def _():
        docs = list_documents(state["backend"])
        ids = {d[0] for d in docs}
        assert state["vm_old"].meta.doc_id in ids and state["vm_new"].meta.doc_id in ids
        return f"{len(docs)} doc(s)"

    @stage("get_vectormap round-trip (provenance preserved)")
    def _():
        rt = get_vectormap(state["backend"], state["vm_old"].meta.doc_id)
        assert rt and rt.pages[0].geoms, "round-trip lost geometry"
        assert all(t.source in ("native", "ocr") for t in rt.pages[0].texts)

    @stage("search_text (PostgreSQL FTS)")
    def _():
        rows = search_text(state["backend"], "VALVE")
        assert rows, "FTS returned no rows for 'VALVE'"
        return f"{len(rows)} hit(s)"

    @stage("diff_documents")
    def _():
        diffs = diff_documents(state["backend"], state["vm_old"].meta.doc_id, state["vm_new"].meta.doc_id)
        assert diffs, "no diffs returned"
        state["diffs"] = diffs
        return f"{len(diffs)} page(s)"

    @stage("write_overlay")
    def _():
        out = str(tmp / "overlay.pdf")
        write_overlay(new_pdf, state["diffs"], out)
        assert Path(out).exists() and Path(out).stat().st_size > 0
        return out

    if args.ocr:
        @stage("OCR pass (image-only page -> source=ocr)")
        def _():
            img_pdf = str(tmp / "scan.pdf")
            _make_image_only_pdf(img_pdf, "PUMP P-200")
            vm = pdf_to_vectormap(img_pdf, workers=1, enable_ocr=True, ocr_dpi=300)
            ocr_texts = [t for p in vm.pages for t in p.texts if t.source == "ocr"]
            assert ocr_texts, "OCR produced no text (check EasyOCR/Tesseract install)"
            joined = " ".join(t.text for t in ocr_texts)
            return f"engine={engine} gpu={gpu} got: {joined!r}"

    if not args.keep:
        @stage("cleanup (delete test docs)")
        def _():
            delete_document(state["backend"], state["vm_old"].meta.doc_id)
            delete_document(state["backend"], state["vm_new"].meta.doc_id)

    return _summary()


def _summary(skipped_db: bool = False) -> int:
    print("=" * 64)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{total} stage(s) passed" + ("  (DB stages skipped)" if skipped_db else ""))
    print("=" * 64)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
