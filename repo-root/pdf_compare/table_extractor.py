"""
High-level helpers for extracting tables from stored PDF documents.

This module wraps the lower-level analyzer in ``pdf_compare.analyzers.table_extractor``
and centralises the repeated boilerplate required to ensure a vectormap is available
for a document before running table extraction. Callers only need to supply a
``DatabaseBackend`` instance, a document identifier, and the path to the PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .db_backend import DatabaseBackend
from .store_new import get_vectormap, upsert_vectormap
from .pdf_extract import pdf_to_vectormap
from .analyzers.table_extractor import (
    Table,
    TableExtractionConfig,
    TableExtractor,
)

LOGGER = logging.getLogger(__name__)


def _coerce_page_numbers(page_numbers: Optional[Sequence[int]]) -> Optional[List[int]]:
    """Return a sanitised list of 1-based page numbers or ``None``."""
    if page_numbers is None:
        return None

    cleaned: List[int] = []
    for page in page_numbers:
        if page is None:
            continue
        if page < 1:
            raise ValueError(f"Page numbers must be 1-based; received {page!r}")
        cleaned.append(int(page))

    if not cleaned:
        return None

    seen = set()
    unique_pages: List[int] = []
    for page in cleaned:
        if page not in seen:
            unique_pages.append(page)
            seen.add(page)
    return unique_pages


def ensure_vectormap(
    backend: DatabaseBackend,
    doc_id: str,
    doc_path: str | Path,
    *,
    page_numbers: Optional[Sequence[int]] = None,
    regenerate_if_missing: bool = True,
    workers: int = 1,
    enable_ocr: bool = True,
) -> Tuple[Optional[object], bool]:
    """
    Ensure a vectormap exists for the requested document.

    Returns the vectormap (or ``None``) and a boolean indicating whether a new
    extraction was performed.
    """
    resolved_path = str(Path(doc_path))
    normalized_pages = _coerce_page_numbers(page_numbers)

    vectormap = get_vectormap(backend, doc_id, page_numbers=normalized_pages)
    if vectormap and getattr(vectormap, "pages", None):
        return vectormap, False

    if not regenerate_if_missing:
        return vectormap, False

    vectormap = pdf_to_vectormap(
        resolved_path,
        doc_id=doc_id,
        workers=workers,
        enable_ocr=enable_ocr,
    )

    try:
        upsert_vectormap(backend, vectormap)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to persist regenerated vectormap for %s: %s", doc_id, exc)

    return vectormap, True


def extract_tables(
    backend: DatabaseBackend,
    doc_id: str,
    doc_path: str | Path,
    config: TableExtractionConfig,
    *,
    page_numbers: Optional[Sequence[int]] = None,
    vectormap: Optional[object] = None,
    regenerate_vectormap: bool = True,
    workers: int = 1,
) -> List[Table]:
    """
    Extract tables for a document using ``TableExtractor``.

    Parameters mirror :func:`ensure_vectormap`; callers typically pass the values
    collected from the user interface.
    """
    normalized_pages = _coerce_page_numbers(page_numbers)
    resolved_path = str(Path(doc_path))

    if vectormap is None:
        vectormap, _ = ensure_vectormap(
            backend,
            doc_id,
            resolved_path,
            page_numbers=normalized_pages,
            regenerate_if_missing=regenerate_vectormap,
            workers=workers,
        )

    extractor = TableExtractor(config)
    page_indices = None
    if normalized_pages is not None:
        page_indices = [page - 1 for page in normalized_pages]

    return extractor.extract_all_tables(
        resolved_path,
        page_indices=page_indices,
        vector_map=vectormap,
    )


def extract_bom_tables(
    backend: DatabaseBackend,
    doc_id: str,
    doc_path: str | Path,
    config: TableExtractionConfig,
    *,
    page_numbers: Optional[Sequence[int]] = None,
    vectormap: Optional[object] = None,
    regenerate_vectormap: bool = True,
    workers: int = 1,
) -> List[Table]:
    """Convenience wrapper that filters :func:`extract_tables` results to BOM tables."""
    tables = extract_tables(
        backend,
        doc_id,
        doc_path,
        config,
        page_numbers=page_numbers,
        vectormap=vectormap,
        regenerate_vectormap=regenerate_vectormap,
        workers=workers,
    )
    return [table for table in tables if table.table_type == "bom"]


__all__ = [
    "extract_tables",
    "extract_bom_tables",
    "ensure_vectormap",
]
