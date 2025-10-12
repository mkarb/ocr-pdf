"""
Search functions with SQLAlchemy backend support.
"""

from typing import Optional, List, Tuple, Union
from .db_backend import DatabaseBackend


def search_text(
    conn_or_backend: Union[DatabaseBackend, any],
    q: str,
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    limit: int = 100
) -> List[Tuple]:
    """
    Full-text search across documents.

    Args:
        conn_or_backend: DatabaseBackend or legacy sqlite3.Connection
        q: Search query
        doc_id: Optional document ID filter
        page: Optional page number filter
        limit: Maximum number of results

    Returns:
        List of (doc_id, page_number, text, bbox, font, size) tuples
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        return conn_or_backend.search_text(q, doc_id, page, limit)
    else:
        # Legacy sqlite3 connection
        from .search import search_text as legacy_search
        return legacy_search(conn_or_backend, q, doc_id, page, limit)
