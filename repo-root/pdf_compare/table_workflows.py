"""
Workflow helpers for table extraction tasks used by the Streamlit UI.

These helpers encapsulate the non-UI logic required to prepare table extraction
configurations, validate page selections, and transform extracted tables for
export routines (CSV / JSON). Moving this logic out of the UI keeps the
Streamlit layer focused on presentation concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .db_backend import DatabaseBackend
from .table_extractor import extract_tables
from .analyzers.table_extractor import Table, TableExtractionConfig


DEFAULT_BOM_KEYWORDS = [
    "PARTS LIST",
    "BILL OF MATERIALS",
    "BOM",
    "MATERIAL LIST",
    "ITEM",
    "QTY",
    "QUANTITY",
    "PART NUMBER",
    "DESCRIPTION",
    "PART NO",
    "PART#",
    "P/N",
    "PN",
    "MATL",
    "MATERIAL",
]


class PageSelectionError(ValueError):
    """Raised when a page selection string is invalid or out of range."""


def parse_page_selection(pages: str, *, total_pages: int) -> List[int]:
    """
    Parse a comma-separated list of 1-based page numbers and validate range.

    Returns a sorted list of unique page numbers. Empty input defaults to the
    first page.
    """
    cleaned = pages.strip()
    if not cleaned:
        return [1]

    try:
        parsed = {int(part.strip()) for part in cleaned.split(",") if part.strip()}
    except ValueError as exc:  # pragma: no cover - validation
        raise PageSelectionError("Invalid page numbers. Use format like '1' or '1,3,5'.") from exc

    if not parsed:
        raise PageSelectionError("No valid page numbers provided.")

    invalid = sorted(num for num in parsed if num < 1 or num > total_pages)
    if invalid:
        joined = ", ".join(str(num) for num in invalid)
        raise PageSelectionError(f"Page number(s) out of range: {joined}")

    return sorted(parsed)


@dataclass
class BOMExtractionResult:
    """Container for quick BOM extraction."""

    bom_tables: List[Table]
    all_tables: List[Table]
    fallback_used: bool


def run_quick_bom_extraction(
    backend: DatabaseBackend,
    doc_id: str,
    doc_path: str | Path,
    *,
    dpi: int,
    page_numbers: Sequence[int],
) -> BOMExtractionResult:
    """
    Run BOM extraction for the selected pages and return both BOM tables and
    the underlying table list for diagnostics/export.
    """
    config = TableExtractionConfig(
        dpi=dpi,
        ocr_min_conf=40,
        enable_line_detection=True,
        enable_whitespace_detection=True,
        bom_keywords=DEFAULT_BOM_KEYWORDS,
    )

    tables = extract_tables(
        backend,
        doc_id,
        doc_path,
        config,
        page_numbers=page_numbers,
        workers=1,
    )

    bom_tables = [table for table in tables if table.table_type == "bom"]
    fallback_used = False
    if not bom_tables:
        bom_tables = tables
        fallback_used = bool(tables)

    return BOMExtractionResult(
        bom_tables=bom_tables,
        all_tables=tables,
        fallback_used=fallback_used,
    )


def run_table_extraction(
    backend: DatabaseBackend,
    doc_id: str,
    doc_path: str | Path,
    *,
    dpi: int,
    ocr_min_conf: int,
    page_numbers: Optional[Sequence[int]],
) -> List[Table]:
    """Perform configurable table extraction for the Advanced workflow."""
    config = TableExtractionConfig(
        dpi=dpi,
        ocr_min_conf=ocr_min_conf,
    )

    return extract_tables(
        backend,
        doc_id,
        doc_path,
        config,
        page_numbers=page_numbers,
        workers=1,
    )


def summarise_tables(tables: Iterable[Table]) -> Dict[str, List]:
    """Return a columnar summary dictionary for display in dataframes."""
    tables_list = list(tables)
    return {
        "Table ID": [table.table_id for table in tables_list],
        "Type": [table.table_type for table in tables_list],
        "Page": [table.page for table in tables_list],
        "Rows": [len(table.rows) for table in tables_list],
        "Columns": [len(table.headers) for table in tables_list],
        "Detection": [table.metadata.get("detection_method", "unknown") for table in tables_list],
    }


def tables_to_csv(tables: Sequence[Table]) -> str:
    """Convert tables into a combined CSV string suitable for download."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("pandas is required for CSV export.") from exc

    frames = []
    for table in tables:
        df = table.to_dataframe()
        df.insert(0, "Table ID", table.table_id)
        df.insert(1, "Table Type", table.table_type)
        df.insert(2, "Page", table.page)
        frames.append(df)

    if not frames:
        return ""

    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False)


__all__ = [
    "BOMExtractionResult",
    "DEFAULT_BOM_KEYWORDS",
    "PageSelectionError",
    "parse_page_selection",
    "run_quick_bom_extraction",
    "run_table_extraction",
    "summarise_tables",
    "tables_to_csv",
]
