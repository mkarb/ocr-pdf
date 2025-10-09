from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 in PDF user space

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
