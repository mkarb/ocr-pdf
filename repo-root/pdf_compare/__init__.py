# pdf_compare/__init__.py
# PostgreSQL-only imports (SQLite support removed)
from .pdf_extract import pdf_to_vectormap
from .store_new import open_db, upsert_vectormap, list_documents, DatabaseBackend
from .search_new import search_text
from .compare_new import diff_documents
from .layout_compare import diff_layout_files
from .overlay import write_overlay

__all__ = [
    "pdf_to_vectormap",
    "open_db",
    "upsert_vectormap",
    "list_documents",
    "DatabaseBackend",
    "search_text",
    "diff_documents",
    "diff_layout_files",
    "write_overlay",
]
