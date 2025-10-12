import typer
import os
from .pdf_extract import pdf_to_vectormap
from .db_backend import DatabaseBackend
from .store_new import upsert_vectormap, list_documents
from .search_new import search_text as search_text_fts
from .compare_new import diff_documents
from .overlay import write_overlay
from .raster_grid import raster_grid_changed_boxes
from .analyzers.highres_ocr import highres_ocr, HighResOCRConfig
import fitz

app = typer.Typer(add_completion=False)

# Helper to get database connection
def get_db_connection(db_url: str | None = None):
    """
    Get database connection from DATABASE_URL env var or provided URL.

    Args:
        db_url: Optional database URL override

    Returns:
        DatabaseBackend instance
    """
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        typer.echo(
            "❌ DATABASE_URL not set!\n\n"
            "This CLI requires PostgreSQL. Set the DATABASE_URL environment variable:\n\n"
            "  export DATABASE_URL=postgresql://user:password@host:5432/dbname\n\n"
            "For local development:\n"
            "  export DATABASE_URL=postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare\n",
            err=True
        )
        raise typer.Exit(code=1)

    return DatabaseBackend(url)

@app.command()
def ingest(
    pdf: str,
    doc_id: str | None = None,
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL URL (or use DATABASE_URL env var)"),
    use_server_mode: bool = typer.Option(
        False,
        "--server",
        help="Use server-optimized extraction (no Streamlit compatibility, better performance)"
    )
):
    """
    Ingest a PDF into the PostgreSQL database.

    Use --server flag for server/Docker deployments (faster, no Streamlit compatibility).

    Example:
        export DATABASE_URL=postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare
        compare-pdf-revs ingest document.pdf
    """
    if use_server_mode or os.getenv("PDF_SERVER_MODE"):
        from .pdf_extract_server import pdf_to_vectormap_server
        vm = pdf_to_vectormap_server(pdf, doc_id=doc_id)
    else:
        vm = pdf_to_vectormap(pdf, doc_id=doc_id)

    backend = get_db_connection(db_url)
    upsert_vectormap(backend, vm)
    typer.echo(f"✅ Ingested {vm.meta.path} as {vm.meta.doc_id} with {vm.meta.page_count} pages.")

@app.command()
def search_text(
    q: str,
    doc_id: str | None = None,
    page: int | None = None,
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL URL (or use DATABASE_URL env var)"),
    limit: int = typer.Option(100, help="Maximum number of results")
):
    """
    Search for text in ingested PDF documents.

    Example:
        export DATABASE_URL=postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare
        compare-pdf-revs search-text "error message" --doc-id doc123 --limit 50
    """
    backend = get_db_connection(db_url)
    rows = search_text_fts(backend, q, doc_id, page, limit=limit)

    if not rows:
        typer.echo("No results found.")
        return

    typer.echo(f"Found {len(rows)} results:\n")
    for r in rows:
        typer.echo(f"  Doc: {r[0]} | Page: {r[1]} | Text: {r[2][:80]}...")

@app.command()
def compare_grid(
    old_pdf: str,
    new_pdf: str,
    out_overlay: str = "grid_diff.pdf",
    base_pdf: str | None = None,
    grid_dpi: int = 400,
    grid_rows: int = 12,
    grid_cols: int = 16,
    grid_ratio: float = 0.03,
):
    """
    Raster grid compare (changed regions only). Does not use the DB.
    """
    doc_old = doc_new = None
    try:
        doc_old = fitz.open(old_pdf)
        doc_new = fitz.open(new_pdf)
        page_count = min(doc_old.page_count, doc_new.page_count)
    finally:
        try:
            if doc_old: doc_old.close()
        except Exception:
            pass
        try:
            if doc_new: doc_new.close()
        except Exception:
            pass

    diffs = []
    for p in range(page_count):
        boxes = raster_grid_changed_boxes(
            old_pdf, new_pdf, p,
            dpi=grid_dpi, rows=grid_rows, cols=grid_cols,
            method="hybrid", cell_change_ratio=grid_ratio, merge_adjacent=True
        )
        diffs.append({
            "page": p + 1,
            "geometry": {"added": [], "removed": [], "changed": boxes},
            "text": {"added": [], "removed": [], "moved": []},
        })

    base = base_pdf or new_pdf
    write_overlay(base, diffs, out_overlay)
    print(f"Overlay written: {out_overlay}")


@app.command()
def compare(
    old_id: str,
    new_id: str,
    out_overlay: str | None = None,
    base_pdf: str | None = None,
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL URL (or use DATABASE_URL env var)"),

    # --- OCR augment knobs ---
    with_ocr: bool = typer.Option(False, help="Run OCR augment on NEW doc before vector/text diff"),
    ocr_mode: str = typer.Option("sparse", help="sparse|all|changed-cells"),
    ocr_dpi: int = 500,
    ocr_min_conf: int = 60,
    ocr_psm: int = 11,

    # --- parameters used when ocr_mode == 'changed-cells' ---
    changed_cells_dpi: int = 400,
    changed_cells_rows: int = 12,
    changed_cells_cols: int = 16,
    changed_cells_ratio: float = 0.03,
):
    """
    Vector/Text compare using PostgreSQL database. Optionally OCR the NEW document first.

    ocr_mode:
      - sparse        -> OCR only pages with very little native text (fast)
      - all           -> OCR every page
      - changed-cells -> Use raster grid to find changed regions, OCR only those tiles (fast + focused)

    Example:
        export DATABASE_URL=postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare
        compare-pdf-revs compare old_doc new_doc --with-ocr --ocr-mode sparse
    """
    backend = get_db_connection(db_url)

    if with_ocr:
        typer.echo("⚠️  OCR augmentation with PostgreSQL backend not yet implemented in CLI.")
        typer.echo("    Use the Streamlit UI or run ocr-augment command separately.")
        typer.echo("    Proceeding with comparison without OCR...\n")

    # Proceed with vector/text diff (DB-backed)
    diffs = diff_documents(backend, old_id, new_id)
    typer.echo(f"✅ Compared {len(diffs)} pages.")
    if out_overlay and base_pdf:
        write_overlay(base_pdf, diffs, out_overlay)
        print(f"Wrote overlay to {out_overlay}")


@app.command()
def ocr_augment(
    doc_id: str,
    dpi: int = 500,
    min_conf: int = 60,
    only_sparse: bool = True,
    db_url: str | None = typer.Option(None, "--db-url", help="PostgreSQL URL (or use DATABASE_URL env var)")
):
    """
    Add OCR text runs to existing document in the PostgreSQL database.

    - only_sparse=True → OCR only pages with little/no native text.

    Example:
        export DATABASE_URL=postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare
        compare-pdf-revs ocr-augment doc_id123 --only-sparse
    """
    backend = get_db_connection(db_url)

    # Get document info
    with backend.get_session() as session:
        from .db_models import Document, TextRow
        doc = session.query(Document).filter(Document.doc_id == doc_id).first()
        if not doc:
            typer.echo(f"❌ Document {doc_id} not found", err=True)
            raise typer.Exit(code=1)

        pdf_path = doc.path
        page_count = doc.page_count

        # Determine which pages to OCR
        pages = list(range(1, page_count + 1))
        if only_sparse:
            sparse = []
            for p in pages:
                n = session.query(TextRow).filter(
                    TextRow.doc_id == doc_id,
                    TextRow.page_number == p
                ).count()
                if n < 5:  # heuristic
                    sparse.append(p)
            pages = sparse

    if not pages:
        typer.echo("ℹ️  No pages need OCR.")
        return

    typer.echo(f"📄 Running OCR on {len(pages)} pages...")
    cfg = HighResOCRConfig(dpi=dpi, psm=11, min_conf=min_conf, max_workers=6, ram_budget_mb=10240)

    with backend.get_session() as session:
        from .db_models import TextRow
        for p in pages:
            runs = highres_ocr(pdf_path, p-1, cfg)
            typer.echo(f"  Page {p}/{page_count}: {len(runs)} OCR spans")

            # Insert OCR text runs
            for r in runs:
                text_row = TextRow(
                    doc_id=doc_id,
                    page_number=p,
                    text=r["text"],
                    bbox=str(list(r["bbox"])),
                    font=None,
                    size=None,
                    source="ocr"
                )
                session.add(text_row)
            session.commit()

    typer.echo(f"✅ OCR augment complete for {doc_id}")


if __name__ == "__main__":
    app()
