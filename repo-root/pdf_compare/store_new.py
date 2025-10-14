"""
New database store with SQLAlchemy backend support.
Provides backwards-compatible API with both SQLite and PostgreSQL support.
"""

from __future__ import annotations
import os
from typing import List, Tuple, Union
from .models import VectorMap
from .db_backend import DatabaseBackend, create_backend


# Module-level backend instance
_backend: DatabaseBackend | None = None


def open_db(database_url: str) -> DatabaseBackend:
    """
    Open database connection with SQLAlchemy backend.

    Args:
        database_url: Database URL string
            - SQLite (legacy): Pass file path, will convert to sqlite:///path
            - SQLite (explicit): "sqlite:///path/to/file.db"
            - PostgreSQL: "postgresql://user:pass@host:port/dbname"

    Returns:
        DatabaseBackend instance
    """
    global _backend

    # Handle legacy file path format
    if not database_url.startswith(("sqlite://", "postgresql://")):
        # Convert file path to SQLite URL
        database_url = f"sqlite:///{database_url}"

    # Create backend if needed
    if _backend is None or _backend.database_url != database_url:
        if _backend:
            _backend.close()
        _backend = create_backend(database_url)

    return _backend


def upsert_vectormap(conn_or_backend: Union[DatabaseBackend, any], vm: VectorMap) -> None:
    """
    Store a VectorMap into the database.

    Args:
        conn_or_backend: DatabaseBackend instance or legacy sqlite3.Connection
        vm: VectorMap to store
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        conn_or_backend.upsert_vectormap(vm)
    else:
        # Legacy sqlite3 connection - import old implementation
        from .store import upsert_vectormap as legacy_upsert
        legacy_upsert(conn_or_backend, vm)


def list_documents(conn_or_backend: Union[DatabaseBackend, any]) -> List[Tuple[str, str, int]]:
    """
    List all documents in the database.

    Returns:
        List of (doc_id, path, page_count) tuples
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.list_documents()
    else:
        # Legacy sqlite3 connection
        from .store import list_documents as legacy_list
        return legacy_list(conn_or_backend)


def delete_document(conn_or_backend: Union[DatabaseBackend, any], doc_id: str) -> bool:
    """
    Delete a document and all its associated data from the database.

    Args:
        conn_or_backend: DatabaseBackend instance or legacy sqlite3.Connection
        doc_id: Document ID to delete

    Returns:
        True if document was deleted, False if not found
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.delete_document(doc_id)
    else:
        # Legacy sqlite3 connection - not implemented
        raise NotImplementedError("delete_document not supported for legacy SQLite connections")


def delete_all_documents(conn_or_backend: Union[DatabaseBackend, any]) -> int:
    """
    Delete all documents from the database.

    Args:
        conn_or_backend: DatabaseBackend instance or legacy sqlite3.Connection

    Returns:
        Number of documents deleted
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.delete_all_documents()
    else:
        # Legacy sqlite3 connection - not implemented
        raise NotImplementedError("delete_all_documents not supported for legacy SQLite connections")


def export_document_text(conn_or_backend: Union[DatabaseBackend, any], doc_id: str, format: str = "txt") -> str:
    """
    Export all text content from a document for debugging.

    Args:
        conn_or_backend: DatabaseBackend instance
        doc_id: Document ID to export
        format: Output format ("txt" or "json")

    Returns:
        Formatted text content
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.export_document_text(doc_id, format)
    else:
        raise NotImplementedError("export_document_text not supported for legacy SQLite connections")


def get_document_text_with_coords(conn_or_backend: Union[DatabaseBackend, any], doc_id: str):
    """
    Get all text with coordinates for creating searchable PDFs.
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.get_document_text_with_coords(doc_id)
    else:
        raise NotImplementedError("get_document_text_with_coords not supported for legacy")


# Export backwards-compatible API
__all__ = ["open_db", "upsert_vectormap", "list_documents", "delete_document", "delete_all_documents", "export_document_text", "get_document_text_with_coords", "DatabaseBackend"]
