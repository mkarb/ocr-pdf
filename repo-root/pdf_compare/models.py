from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Literal

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

# --------------------------------------------------------------------
# Diff payload models (consumed by overlay.py, UI) – additive
# --------------------------------------------------------------------
@dataclass(frozen=True)
class GeometryDiff:
    added: List[BBox] = field(default_factory=list)
    removed: List[BBox] = field(default_factory=list)
    changed: List[BBox] = field(default_factory=list)   # for raster/grid diffs

@dataclass(frozen=True)
class TextAdd:
    text: str
    bbox: BBox
    source: Optional[Literal["native","ocr"]] = None     # optional provenance

@dataclass(frozen=True)
class TextMoved:
    text: str
    from_bbox: BBox
    to_bbox: BBox

@dataclass(frozen=True)
class TextDiff:
    added: List[TextAdd] = field(default_factory=list)
    removed: List[TextAdd] = field(default_factory=list)
    moved: List[TextMoved] = field(default_factory=list)

@dataclass(frozen=True)
class PageDiff:
    page: int                   # 1-based page index
    geometry: GeometryDiff
    text: TextDiff

@dataclass(frozen=True)
class RasterGridConfig:
    dpi: int = 400
    rows: int = 12
    cols: int = 16
    cell_change_ratio: float = 0.03   # 3% pixels changed to flag cell
    method: str = "hybrid"            # or "ssim"
    merge_adjacent: bool = True

# --------------------------------------------------------------------
# Helpers to convert between typed PageDiff and overlay.py dicts
# --------------------------------------------------------------------
def page_diff_from_dict(d: dict) -> PageDiff:
    g = d.get("geometry", {})
    t = d.get("text", {})
    return PageDiff(
        page=int(d["page"]),
        geometry=GeometryDiff(
            added=g.get("added", []) or [],
            removed=g.get("removed", []) or [],
            changed=g.get("changed", []) or [],
        ),
        text=TextDiff(
            added=[TextAdd(text=x["text"], bbox=tuple(x["bbox"]), source=x.get("source"))
                   for x in (t.get("added", []) or [])],
            removed=[TextAdd(text=x["text"], bbox=tuple(x["bbox"]), source=x.get("source"))
                     for x in (t.get("removed", []) or [])],
            moved=[TextMoved(text=m["text"], from_bbox=tuple(m["from"]), to_bbox=tuple(m["to"]))
                   for m in (t.get("moved", []) or [])],
        ),
    )

def page_diff_to_overlay_dict(pd: PageDiff) -> dict:
    return {
        "page": pd.page,
        "geometry": {
            "added": pd.geometry.added,
            "removed": pd.geometry.removed,
            "changed": pd.geometry.changed,
        },
        "text": {
            "added": [{"text": x.text, "bbox": x.bbox, **({"source": x.source} if x.source else {})}
                      for x in pd.text.added],
            "removed": [{"text": x.text, "bbox": x.bbox, **({"source": x.source} if x.source else {})}
                        for x in pd.text.removed],
            "moved": [{"text": m.text, "from": m.from_bbox, "to": m.to_bbox}
                      for m in pd.text.moved],
        },
    }

def diffs_to_overlay_dicts(pages: List[PageDiff]) -> List[dict]:
    return [page_diff_to_overlay_dict(p) for p in pages]

__all__ = [
    # ingest/storage
    "BBox", "GeoKind", "VectorGeom", "TextRun", "PageVectors", "DocMeta", "VectorMap",
    # diff/config
    "GeometryDiff", "TextAdd", "TextMoved", "TextDiff", "PageDiff", "RasterGridConfig",
    # helpers
    "page_diff_from_dict", "page_diff_to_overlay_dict", "diffs_to_overlay_dicts",
]
