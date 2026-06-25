from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional

# --------------------------------------------------------------------
# Shared alias
# --------------------------------------------------------------------
BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 in PDF user space

# --------------------------------------------------------------------
# Ingest / storage models (used by pdf_extract, store, etc.)
# --------------------------------------------------------------------
class GeoKind(Enum):
    STROKE = auto()
    FILL = auto()

@dataclass(frozen=True)
class VectorGeom:
    kind: GeoKind
    wkb: bytes           # shapely geometry serialized as WKB
    bbox: BBox

@dataclass(frozen=True)
class TextRun:
    text: str
    bbox: BBox
    font: Optional[str]
    size: Optional[float]
    source: str = "native"   # provenance: 'native' (PDF text) or 'ocr'

@dataclass(frozen=True)
class PageVectors:
    page_number: int     # 1-based
    width: float
    height: float
    rotation: int        # 0/90/180/270
    geoms: List[VectorGeom]
    texts: List[TextRun]

@dataclass(frozen=True)
class DocMeta:
    doc_id: str          # stable id (hash of content or provided)
    path: str
    page_count: int

@dataclass(frozen=True)
class VectorMap:
    meta: DocMeta
    pages: List[PageVectors]


def text_run_from_dict(d: dict) -> TextRun:
    """Build a TextRun from an extraction-worker dict, preserving `source`."""
    return TextRun(
        text=d["text"],
        bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
        font=d.get("font"),
        size=d.get("size"),
        source=d.get("source") or "native",
    )


__all__ = [
    "BBox", "GeoKind", "VectorGeom", "TextRun", "PageVectors", "DocMeta", "VectorMap",
    "text_run_from_dict",
]
