import typer
from .pdf_extract import pdf_to_vectormap
from .store import open_db, upsert_vectormap, list_documents
from .search import search_text as search_text_fts
from .compare import diff_documents
from .overlay import write_overlay

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
def compare(old_id: str, new_id: str, db: str="vectormap.sqlite", out_overlay: str|None=None, base_pdf: str|None=None):
    conn = open_db(db)
    diffs = diff_documents(conn, old_id, new_id)
    print(diffs)
    if out_overlay and base_pdf:
        write_overlay(base_pdf, diffs, out_overlay)
        print(f"Wrote overlay to {out_overlay}")

if __name__ == "__main__":
    app()
