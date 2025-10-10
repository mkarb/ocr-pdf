import typer
from .pdf_extract import pdf_to_vectormap
from .store import open_db, upsert_vectormap, list_documents
from .search import search_text as search_text_fts
from .compare import diff_documents
from .overlay import write_overlay
from .raster_grid import raster_grid_changed_boxes
import fitz

app = typer.Typer(add_completion=False)

@app.command()
def ingest(pdf: str, db: str = "vectormap.sqlite", doc_id: str | None = None):
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
    doc_old = fitz.open(old_pdf)
    doc_new = fitz.open(new_pdf)
    page_count = min(doc_old.page_count, doc_new.page_count)
    doc_old.close(); doc_new.close()

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
def compare(old_id: str, new_id: str, db: str="vectormap.sqlite", out_overlay: str|None=None, base_pdf: str|None=None):
    conn = open_db(db)
    diffs = diff_documents(conn, old_id, new_id)
    print(diffs)
    if out_overlay and base_pdf:
        write_overlay(base_pdf, diffs, out_overlay)
        print(f"Wrote overlay to {out_overlay}")

if __name__ == "__main__":
    app()
