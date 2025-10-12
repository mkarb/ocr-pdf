"""
Document comparison with SQLAlchemy backend support.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Union
import json
import numpy as np
from shapely.wkb import loads as wkb_loads
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException

from .db_backend import DatabaseBackend

# Tolerances
GEO_TOL: float = 0.15
TEXT_MOVE_TOL: float = 0.75
AREA_EPS: float = 1e-2


# Geometry matching helpers (reused from original)
def _as_list(arr_or_seq):
    """Normalize numpy arrays / sequences to a Python list."""
    if isinstance(arr_or_seq, np.ndarray):
        return arr_or_seq.tolist()
    return list(arr_or_seq)


def _query_hits(tree: STRtree | None, gb: BaseGeometry, backing: List[BaseGeometry]) -> List[BaseGeometry]:
    """Return geometries from STRtree.query."""
    if tree is None or gb is None or gb.is_empty:
        return []
    raw = tree.query(gb)
    hits = _as_list(raw)
    if not hits:
        return []

    first = hits[0]
    is_geom = hasattr(first, "geom_type") or isinstance(first, BaseGeometry)
    if not is_geom:
        out: List[BaseGeometry] = []
        for idx in hits:
            ii = int(idx)
            if 0 <= ii < len(backing):
                out.append(backing[ii])
        return out

    return hits


def _geom_matches(gb: BaseGeometry, candidates: List[BaseGeometry]) -> bool:
    """Return True if buffered geom matches any candidate within tolerance."""
    if gb is None or gb.is_empty:
        return False
    for h in candidates:
        if h is None or h.is_empty:
            continue
        try:
            if not gb.envelope.intersects(h.envelope):
                continue
            if gb.intersects(h):
                sym = gb.symmetric_difference(h)
                if sym.is_empty or sym.area < AREA_EPS:
                    return True
            if gb.difference(h).area < AREA_EPS or h.difference(gb).area < AREA_EPS:
                return True
        except GEOSException:
            continue
    return False


def diff_documents(
    conn_or_backend: Union[DatabaseBackend, any],
    old_id: str,
    new_id: str,
    pages: List[int] | None = None
) -> List[Dict]:
    """
    Compare two ingested documents page by page (vector + text).
    Returns a list of per-page diff dicts compatible with overlay.py.
    """
    if isinstance(conn_or_backend, DatabaseBackend):
        backend = conn_or_backend

        # Get page counts from backend
        with backend.SessionLocal() as session:
            from .db_models import Document
            old_doc = session.get(Document, old_id)
            new_doc = session.get(Document, new_id)

            if not old_doc or not new_doc:
                raise ValueError("Unknown doc_id(s) supplied to diff_documents")

            pc_old, pc_new = old_doc.page_count, new_doc.page_count

        max_pages = min(pc_old, pc_new)
        if pages is None:
            pages = list(range(1, max_pages + 1))
        return [diff_pages(backend, old_id, new_id, p) for p in pages]
    else:
        # Legacy sqlite3 connection
        from .compare import diff_documents as legacy_diff
        return legacy_diff(conn_or_backend, old_id, new_id, pages)


def diff_pages(
    conn_or_backend: Union[DatabaseBackend, any],
    old_id: str,
    new_id: str,
    page: int
) -> Dict:
    """Compare a single page between two documents."""

    if isinstance(conn_or_backend, DatabaseBackend):
        backend = conn_or_backend

        # Load geometries
        a_geom_wkbs = backend.load_page_geoms(old_id, page)
        b_geom_wkbs = backend.load_page_geoms(new_id, page)
        a_geoms = [wkb_loads(wkb) for wkb in a_geom_wkbs]
        b_geoms = [wkb_loads(wkb) for wkb in b_geom_wkbs]

        # Load text
        a_txt = backend.load_page_texts(old_id, page)
        b_txt = backend.load_page_texts(new_id, page)
    else:
        # Legacy path
        from .compare import diff_pages as legacy_diff_page
        return legacy_diff_page(conn_or_backend, old_id, new_id, page)

    # Early no-op
    if not a_geoms and not b_geoms and not a_txt and not b_txt:
        return {
            "page": page,
            "geometry": {"added": [], "removed": [], "changed": []},
            "text": {"added": [], "removed": [], "moved": []},
        }

    # Geometry diff
    a_buf = [g.buffer(GEO_TOL) for g in a_geoms] if a_geoms else []
    b_buf = [g.buffer(GEO_TOL) for g in b_geoms] if b_geoms else []

    tree_a = STRtree(a_buf) if a_buf else None
    tree_b = STRtree(b_buf) if b_buf else None

    removed_geo, added_geo = [], []

    # removed: A not in B
    for g, gb in zip(a_geoms, a_buf):
        hits = _query_hits(tree_b, gb, b_buf)
        if not hits or not _geom_matches(gb, hits):
            removed_geo.append(g)

    # added: B not in A
    for g, gb in zip(b_geoms, b_buf):
        hits = _query_hits(tree_a, gb, a_buf)
        if not hits or not _geom_matches(gb, hits):
            added_geo.append(g)

    # Text diff
    used_b: set[int] = set()
    removed_text: List[Dict] = []
    added_text: List[Dict] = []
    moved_text: List[Dict] = []

    def _center(bb):
        x0, y0, x1, y1 = bb
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    for i, (t, bb) in enumerate(a_txt):
        ac = _center(bb)
        match_j = None
        best_dist = None
        for j, (t2, bb2) in enumerate(b_txt):
            if j in used_b or t2 != t:
                continue
            bc = _center(bb2)
            dx, dy = ac[0] - bc[0], ac[1] - bc[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist, match_j = dist, j

        if match_j is None:
            removed_text.append({"text": t, "bbox": bb})
        else:
            used_b.add(match_j)
            if best_dist is not None and best_dist > TEXT_MOVE_TOL:
                moved_text.append({"text": t, "from": bb, "to": b_txt[match_j][1]})

    for j, (t2, bb2) in enumerate(b_txt):
        if j not in used_b:
            added_text.append({"text": t2, "bbox": bb2})

    return {
        "page": page,
        "geometry": {
            "added": [g.bounds for g in added_geo],
            "removed": [g.bounds for g in removed_geo],
            "changed": [],
        },
        "text": {
            "added": added_text,
            "removed": removed_text,
            "moved": moved_text,
        },
    }
