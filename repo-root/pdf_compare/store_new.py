"""
PostgreSQL database store with SQLAlchemy backend.
Provides a clean API for storing and retrieving PDF vector data.
"""

from __future__ import annotations
import os
from typing import List, Tuple
from .models import VectorMap
from .db_backend import DatabaseBackend, create_backend


# Module-level backend instance
_backend: DatabaseBackend | None = None


def open_db(database_url: str) -> DatabaseBackend:
    """
    Open PostgreSQL database connection with SQLAlchemy backend.

    Args:
        database_url: PostgreSQL connection string
            Format: "postgresql://user:pass@host:port/dbname"

    Returns:
        DatabaseBackend instance

    Raises:
        ValueError: If database_url is not a PostgreSQL URL
    """
    global _backend

    # Validate PostgreSQL URL
    if not database_url.startswith("postgresql://"):
        raise ValueError(
            f"Invalid database URL. Expected PostgreSQL URL (postgresql://...), got: {database_url[:20]}..."
        )

    # Create backend if needed
    if _backend is None or _backend.database_url != database_url:
        if _backend:
            _backend.close()
        _backend = create_backend(database_url)

    return _backend


def upsert_vectormap(backend: DatabaseBackend, vm: VectorMap) -> None:
    """
    Store a VectorMap into the database.

    Args:
        backend: DatabaseBackend instance
        vm: VectorMap to store
    """
    backend.upsert_vectormap(vm)


def list_documents(backend: DatabaseBackend) -> List[Tuple[str, str, int]]:
    """
    List all documents in the database.

    Args:
        backend: DatabaseBackend instance

    Returns:
        List of (doc_id, path, page_count) tuples
    """
    return backend.list_documents()


def delete_document(backend: DatabaseBackend, doc_id: str) -> bool:
    """
    Delete a document and all its associated data from the database.

    Args:
        backend: DatabaseBackend instance
        doc_id: Document ID to delete

    Returns:
        True if document was deleted, False if not found
    """
    return backend.delete_document(doc_id)


def delete_all_documents(backend: DatabaseBackend) -> int:
    """
    Delete all documents from the database.

    Args:
        backend: DatabaseBackend instance

    Returns:
        Number of documents deleted
    """
    return backend.delete_all_documents()


def export_document_text(backend: DatabaseBackend, doc_id: str, format: str = "txt") -> str:
    """
    Export all text content from a document for debugging.

    Args:
        backend: DatabaseBackend instance
        doc_id: Document ID to export
        format: Output format ("txt" or "json")

    Returns:
        Formatted text content
    """
    return backend.export_document_text(doc_id, format)


def get_document_text_with_coords(backend: DatabaseBackend, doc_id: str):
    """
    Get all text with coordinates for creating searchable PDFs.

    Args:
        backend: DatabaseBackend instance
        doc_id: Document ID

    Returns:
        List of (page_number, text, (x0,y0,x1,y1), source) tuples
    """
    return backend.get_document_text_with_coords(doc_id)


def get_vectormap(backend: DatabaseBackend, doc_id: str, page_numbers: list[int] | None = None) -> VectorMap | None:
    """Retrieve a stored VectorMap for the given document."""

    return backend.get_vectormap(doc_id, page_numbers=page_numbers)


# Export public API
__all__ = [
    "open_db",
    "upsert_vectormap",
    "list_documents",
    "delete_document",
    "delete_all_documents",
    "export_document_text",
    "get_document_text_with_coords",
    "get_vectormap",
    "DatabaseBackend",
]
