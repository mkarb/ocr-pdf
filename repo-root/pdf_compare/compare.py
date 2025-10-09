# compare.py
from __future__ import annotations
from typing import List, Tuple, Dict
import sqlite3
import json
import numpy as np
import shapely
from shapely.wkb import loads as wkb_loads
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException
from .raster_diff import raster_diff_boxes
# --- Enforce Shapely 2.x without extra deps
try:
    _major = int(shapely.__version__.split(".", 1)[0])
except Exception:
    _major = 2  # assume ok if version string is odd
if _major < 2:
    raise RuntimeError(f"Shapely 2.x required; found {shapely.__version__}")

# Tolerances
GEO_TOL: float = 0.15
TEXT_MOVE_TOL: float = 0.75
AREA_EPS: float = 1e-2

# -----------------------
# Loading helpers
# -----------------------
def _load_page_geoms(conn: sqlite3.Connection, doc_id: str, page: int) -> List[BaseGeometry]:
    rows = conn.execute(
        "SELECT wkb FROM geometry WHERE doc_id=? AND page_number=?", (doc_id, page)
    ).fetchall()
    return [wkb_loads(r[0]) for r in rows]

def _load_page_texts(conn: sqlite3.Connection, doc_id: str, page: int) -> List[Tuple[str, Tuple[float,float,float,float]]]:
    rows = conn.execute(
        "SELECT text, bbox FROM text_rows WHERE doc_id=? AND page_number=?", (doc_id, page)
    ).fetchall()
    out: List[Tuple[str, Tuple[float,float,float,float]]] = []
    for t, b in rows:
        try:
            x0, y0, x1, y1 = map(float, json.loads(b))
        except Exception:
            x0, y0, x1, y1 = map(float, b.strip("[]").split(","))
        out.append((t, (x0, y0, x1, y1)))
    return out

# -----------------------
# Geometry matching
# -----------------------
def _as_list(arr_or_seq):
    """Normalize numpy arrays / sequences to a Python list."""
    if isinstance(arr_or_seq, np.ndarray):
        return arr_or_seq.tolist()
    return list(arr_or_seq)

def _query_hits(tree: STRtree | None, gb: BaseGeometry, backing: List[BaseGeometry]) -> List[BaseGeometry]:
    """
    Return geometries from STRtree.query. Works whether query returns indices (np.int64)
    or geometry objects. `backing` must be the same sequence used to build the tree.
    """
    if tree is None or gb is None or gb.is_empty:
        return []
    raw = tree.query(gb)                         # ndarray[int] OR ndarray[geom]
    hits = _as_list(raw)
    if not hits:
        return []

    # If first element isn't a Shapely geometry, treat as indices and map
    first = hits[0]
    is_geom = hasattr(first, "geom_type") or isinstance(first, BaseGeometry)
    if not is_geom:
        out: List[BaseGeometry] = []
        for idx in hits:
            ii = int(idx)                        # handles np.int64 cleanly
            if 0 <= ii < len(backing):
                out.append(backing[ii])
        return out

    # Already geometry objects
    return hits


def _geom_matches(gb: BaseGeometry, candidates: List[BaseGeometry]) -> bool:
    """Return True if buffered geom `gb` matches any candidate within tolerance."""
    if gb is None or gb.is_empty:
        return False
    for h in candidates:
        if h is None or h.is_empty:
            continue
        try:
            # quick reject on envelopes
            if not gb.envelope.intersects(h.envelope):
                continue
            # robust overlap checks
            if gb.intersects(h):
                sym = gb.symmetric_difference(h)
                if sym.is_empty or sym.area < AREA_EPS:
                    return True
            # near-equality fallback
            if gb.difference(h).area < AREA_EPS or h.difference(gb).area < AREA_EPS:
                return True
        except GEOSException:
            # Skip degenerate/invalid cases
            continue
    return False

# -----------------------
# Public API
# -----------------------
def diff_documents(conn: sqlite3.Connection, old_id: str, new_id: str, pages: List[int] | None = None):
    """
    Compare two ingested documents page by page.
    Returns a list of per-page diff dicts.
    """
    row_old = conn.execute("SELECT page_count FROM documents WHERE doc_id=?", (old_id,)).fetchone()
    row_new = conn.execute("SELECT page_count FROM documents WHERE doc_id=?", (new_id,)).fetchone()
    if row_old is None or row_new is None:
        raise ValueError("Unknown doc_id(s) supplied to diff_documents")
    pc_old, pc_new = row_old[0], row_new[0]

    max_pages = min(pc_old, pc_new)
    if pages is None:
        pages = list(range(1, max_pages + 1))
    return [diff_pages(conn, old_id, new_id, p) for p in pages]

def diff_pages(conn: sqlite3.Connection, old_id: str, new_id: str, page: int) -> Dict:
    # ---------------- Load content ----------------
    a_geoms = _load_page_geoms(conn, old_id, page)
    b_geoms = _load_page_geoms(conn, new_id, page)
    a_txt = _load_page_texts(conn, old_id, page)
    b_txt = _load_page_texts(conn, new_id, page)

    # Early no-op: nothing on either side at all
    if not a_geoms and not b_geoms and not a_txt and not b_txt:
        return {
            "page": page,
            "geometry": {"added": [], "removed": []},
            "text": {"added": [], "removed": [], "moved": []},
        }

    # ---------------- Geometry diff ----------------
    a_buf = [g.buffer(GEO_TOL) for g in a_geoms] if a_geoms else []
    b_buf = [g.buffer(GEO_TOL) for g in b_geoms] if b_geoms else []

    tree_a = STRtree(a_buf) if a_buf else None
    tree_b = STRtree(b_buf) if b_buf else None

    removed_geo, added_geo = [], []

    #    removed: A not in B
    for g, gb in zip(a_geoms, a_buf):
        hits = _query_hits(tree_b, gb, b_buf)  # backing = b_buf (tree_b was built on b_buf)
        # Sanity check: after mapping, hits must be geometries
        if hits and not (hasattr(hits[0], "geom_type") or isinstance(hits[0], BaseGeometry)):
            raise RuntimeError(f"Non-geometry in hits after mapping (removed loop): {type(hits[0]).__name__}")
        if not hits or not _geom_matches(gb, hits):
            removed_geo.append(g)

    # added: B not in A
    for g, gb in zip(b_geoms, b_buf):
        hits = _query_hits(tree_a, gb, a_buf)  # backing = a_buf (tree_a was built on a_buf)
        # Sanity check: after mapping, hits must be geometries
        if hits and not (hasattr(hits[0], "geom_type") or isinstance(hits[0], BaseGeometry)):
            raise RuntimeError(f"Non-geometry in hits after mapping (added loop): {type(hits[0]).__name__}")
        if not hits or not _geom_matches(gb, hits):
            added_geo.append(g)

    # ---------------- Text diff ----------------
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
        },
        "text": {
            "added": added_text,
            "removed": removed_text,
            "moved": moved_text,
        },
    }
