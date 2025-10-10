import typer
import os
from .pdf_extract import pdf_to_vectormap
from .store import open_db, upsert_vectormap, list_documents
from .search import search_text as search_text_fts
from .compare import diff_documents
from .overlay import write_overlay
from .raster_grid import raster_grid_changed_boxes
from .analyzers.highres_ocr import highres_ocr, HighResOCRConfig
import fitz

app = typer.Typer(add_completion=False)

@app.command()
def ingest(
    pdf: str,
    db: str = "vectormap.sqlite",
    doc_id: str | None = None,
    use_server_mode: bool = typer.Option(
        False,
        "--server",
        help="Use server-optimized extraction (no Streamlit compatibility, better performance)"
    )
):
    """
    Ingest a PDF into the vector database.

    Use --server flag for server/Docker deployments (faster, no Streamlit compatibility).
    """
    if use_server_mode or os.getenv("PDF_SERVER_MODE"):
        from .pdf_extract_server import pdf_to_vectormap_server
        vm = pdf_to_vectormap_server(pdf, doc_id=doc_id)
    else:
        vm = pdf_to_vectormap(pdf, doc_id=doc_id)

    conn = open_db(db)
    upsert_vectormap(conn, vm)
    typer.echo(f"Ingested {vm.meta.path} as {vm.meta.doc_id} with {vm.meta.page_count} pages.")

@app.command()
def search_text(q: str, db: str="vectormap.sqlite", doc_id: str|None=None, page: int|None=None):
    from .search import search_text as st
    conn = open_db(db)
    rows = st(conn, q, doc_id, page, limit=100)
    for r in rows:
        print(r)

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
    db: str = "vectormap.sqlite",
    out_overlay: str | None = None,
    base_pdf: str | None = None,

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
    Vector/Text compare (DB). Optionally OCR the NEW document first.

    ocr_mode:
      - sparse        -> OCR only pages with very little native text (fast)
      - all           -> OCR every page
      - changed-cells -> Use raster grid to find changed regions, OCR only those tiles (fast + focused)
    """
    conn = open_db(db)

    if with_ocr:
        _ensure_source_column(conn)

        # NEW doc path + pages
        row = conn.execute(
            "SELECT path, page_count FROM documents WHERE doc_id=?",
            (new_id,)
        ).fetchone()
        if not row:
            raise typer.Exit(code=1)
        new_pdf, page_count = row[0], row[1]

        # choose pages
        pages = list(range(1, page_count + 1))
        changed_boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = {}

        if ocr_mode == "sparse":
            sparse = []
            for p in pages:
                n = conn.execute(
                    "SELECT COUNT(*) FROM text_rows WHERE doc_id=? AND page_number=?",
                    (new_id, p)
                ).fetchone()[0]
                if n < 5:  # heuristic threshold for "low native text"
                    sparse.append(p)
            pages = sparse

        elif ocr_mode == "changed-cells":
            # Need OLD pdf path to compute changed tiles
            row2 = conn.execute(
                "SELECT path FROM documents WHERE doc_id=?",
                (old_id,)
            ).fetchone()
            if not row2:
                raise typer.Exit(code=1)
            old_pdf = row2[0]

            for i in range(page_count):  # i is 0-based
                boxes = raster_grid_changed_boxes(
                    old_pdf, new_pdf, i,
                    dpi=changed_cells_dpi,
                    rows=changed_cells_rows,
                    cols=changed_cells_cols,
                    method="hybrid",
                    cell_change_ratio=changed_cells_ratio,
                    merge_adjacent=True,
                )
                if boxes:
                    changed_boxes_by_page[i + 1] = boxes  # store as 1-based
            pages = list(changed_boxes_by_page.keys())

        # run OCR
        cfg = HighResOCRConfig(dpi=ocr_dpi, psm=ocr_psm, min_conf=ocr_min_conf)
        if not pages:
            print(f"OCR: no pages selected (mode: {ocr_mode})")
        else:
            print(f"OCR mode={ocr_mode}; pages={pages}")
            for p in pages:
                if ocr_mode == "changed-cells":
                    tiles = changed_boxes_by_page.get(p, [])
                    spans = highres_ocr(new_pdf, p - 1, cfg, tiles_pdf=tiles)
                else:
                    spans = highres_ocr(new_pdf, p - 1, cfg)
                _insert_ocr_spans(conn, new_id, p, spans)
                print(f"OCR page {p}/{page_count}: {len(spans)} spans")
            conn.commit()
            print("OCR augment complete.")

    # Proceed with vector/text diff (DB-backed)
    diffs = diff_documents(conn, old_id, new_id)
    print(f"Compared {len(diffs)} pages.")
    if out_overlay and base_pdf:
        write_overlay(base_pdf, diffs, out_overlay)
        print(f"Wrote overlay to {out_overlay}")


@app.command()
def ocr_augment(doc_id: str, db: str = "vectormap.sqlite",
                dpi: int = 500, min_conf: int = 60, only_sparse: bool = True):
    """
    Add OCR text runs to existing document in the DB.
    - only_sparse=True → OCR only pages with little/no native text.
    """
    conn = open_db(db)
    # 1) get PDF path + page_count
    row = conn.execute("SELECT path, page_count FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if not row:
        raise typer.Exit(code=1)
    pdf_path, page_count = row

    # 2) determine which pages to OCR
    pages = range(1, page_count + 1)
    if only_sparse:
        sparse = []
        for p in pages:
            n = conn.execute("SELECT COUNT(*) FROM text_rows WHERE doc_id=? AND page_number=?", (doc_id, p)).fetchone()[0]
            if n < 5:   # heuristic
                sparse.append(p)
        pages = sparse

    if not pages:
        typer.echo("No pages need OCR."); return

    cfg = HighResOCRConfig(dpi=dpi, psm=11, min_conf=min_conf, max_workers=6, ram_budget_mb=10240)
    for p in pages:
        # returns list of {"text": str, "bbox": (x0,y0,x1,y1), ...} in PDF coords
        runs = highres_ocr(pdf_path, p-1, cfg)
        # upsert into text_rows with source='ocr'
        for r in runs:
            conn.execute(
                "INSERT INTO text_rows (doc_id,page_number,text,bbox,font,size,source) VALUES (?,?,?,?,?,?,?)",
                (doc_id, p, r["text"], str(list(r["bbox"])), None, None, "ocr")
            )
        conn.commit()
    typer.echo(f"OCR augment complete for {doc_id} (pages: {list(pages)})")


def _ensure_source_column(conn):
    # safe to run each time
    try:
        conn.execute("ALTER TABLE text_rows ADD COLUMN source TEXT DEFAULT 'native'")
    except Exception:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_text_rows_source ON text_rows(source)")
    conn.commit()


def _insert_ocr_spans(conn, doc_id: str, page_1based: int, spans: list[dict]):
    for r in spans:
        conn.execute(
            "INSERT INTO text_rows (doc_id,page_number,text,bbox,font,size,source) VALUES (?,?,?,?,?,?,?)",
            (doc_id, page_1based, r["text"], str(list(r["bbox"])), None, None, "ocr")
        )
if __name__ == "__main__":
    app()
