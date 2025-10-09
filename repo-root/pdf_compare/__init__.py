# pdf_compare/__init__.py
from .pdf_extract import pdf_to_vectormap
from .store import open_db, upsert_vectormap, list_documents
from .search import search_text
from .compare import diff_documents
from .overlay import write_overlay

__all__ = [
    "pdf_to_vectormap", "open_db", "upsert_vectormap", "list_documents",
    "search_text", "diff_documents", "write_overlay",
]
