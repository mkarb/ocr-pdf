"""
Search functions with SQLAlchemy backend support.
"""

from typing import Optional, List, Tuple
from .db_backend import DatabaseBackend


def search_text(
    backend: DatabaseBackend,
    q: str,
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    limit: int = 100
) -> List[Tuple]:
    """
    Full-text search across documents.

    Args:
        backend: DatabaseBackend instance
        q: Search query
        doc_id: Optional document ID filter
        page: Optional page number filter
        limit: Maximum number of results

    Returns:
        List of (doc_id, page_number, text, bbox, font, size) tuples
    """
    return backend.search_text(q, doc_id, page, limit)
