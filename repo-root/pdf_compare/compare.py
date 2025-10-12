from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import sqlite3
import json
import numpy as np
import shapely
from shapely.wkb import loads as wkb_loads
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException

try:
    from .models import page_diff_from_dict, PageDiff  # type: ignore
except Exception:
    page_diff_from_dict = None
    PageDiff = None  # type: ignore

try:
    from .page_alignment import align_pages, get_alignment_summary  # type: ignore
except Exception:
    align_pages = None  # type: ignore
    get_alignment_summary = None  # type: ignore

try:
    _major = int(shapely.__version__.split(".", 1)[0])
except Exception:
    _major = 2
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

    first = hits[0]
    is_geom = hasattr(first, "geom_type") or isinstance(first, BaseGeometry)
    if not is_geom:
        out: List[BaseGeometry] = []
        for idx in hits:
            ii = int(idx)                        # handle np.int64
            if 0 <= ii < len(backing):
                out.append(backing[ii])
        return out

    return hits

def _geom_matches(gb: BaseGeometry, candidates: List[BaseGeometry]) -> bool:
    """Return True if buffered geom `gb` matches any candidate within tolerance."""
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

# -----------------------
# Public API (dict-based)
# -----------------------
def diff_documents(conn: sqlite3.Connection, old_id: str, new_id: str, pages: List[int] | None = None):
    """
    Compare two ingested documents page by page (vector + text).
    Returns a list of per-page diff dicts compatible with overlay.py.

    Note: This function assumes page numbers align (1:1, 2:2, etc.).
    For documents with inserted/deleted pages, use diff_documents_aligned() instead.
    """
    row_old = conn.execute("SELECT page_count FROM documents WHERE doc_id=?", (old_id,)).fetchone()
    row_new = conn.execute("SELECT page_count FROM documents WHERE doc_id=?", (new_id,)).fetchone()
    if row_old is None or row_new is None:
        raise ValueError("Unknown doc_id(s) supplied to diff_documents")
    pc_old, pc_new = row_old[0], row_new[0]

    max_pages = min(pc_old, pc_new)
    if pages is None:
        pages = list(range(1, max_pages + 1))
    return [diff_pages(conn, old_id, new_id, p, p) for p in pages]

def diff_documents_typed(conn: sqlite3.Connection, old_id: str, new_id: str, pages: List[int] | None = None):
    if page_diff_from_dict is None:
        raise RuntimeError("Typed models not available; ensure pdf_compare.models defines PageDiff helpers.")
    return [page_diff_from_dict(d) for d in diff_documents(conn, old_id, new_id, pages)]


# -----------------------
# Aligned comparison (handles page insertions/deletions)
# -----------------------
def diff_documents_aligned(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    alignment_method: str = "dynamic",
    similarity_threshold: float = 0.5
) -> Tuple[List[Dict], List[Tuple[Optional[int], Optional[int], float]]]:
    """
    Compare two documents with intelligent page alignment.

    Handles cases where pages are inserted, deleted, or reordered.
    Uses content-based matching to align pages before comparison.

    Args:
        conn: Database connection
        old_id: Old document ID
        new_id: New document ID
        alignment_method: "greedy" (faster) or "dynamic" (optimal, default)
        similarity_threshold: Minimum similarity to match pages (0.0-1.0, default 0.5)

    Returns:
        Tuple of (diffs, alignments):
            - diffs: List of page diff dicts (same format as diff_documents)
            - alignments: List of (old_page, new_page, score) tuples

    Example:
        diffs, alignments = diff_documents_aligned(conn, "old", "new")

        print("Page Alignment:")
        for old_pg, new_pg, score in alignments:
            if old_pg and new_pg:
                print(f"  Page {old_pg} → {new_pg} (match: {score:.2%})")
            elif old_pg:
                print(f"  Page {old_pg} was DELETED")
            else:
                print(f"  Page {new_pg} was INSERTED")

        print(f"\\nFound {len(diffs)} page diffs")
    """
    if align_pages is None:
        raise RuntimeError("Page alignment module not available. Check page_alignment.py import.")

    # Step 1: Compute page alignment
    alignments = align_pages(conn, old_id, new_id, alignment_method, similarity_threshold)

    # Step 2: Compare aligned pages
    diffs = []

    for old_page, new_page, score in alignments:
        if old_page and new_page:
            # Pages matched - compare them
            diff = diff_pages(conn, old_id, new_id, old_page, new_page)
            diff["alignment"] = {
                "old_page": old_page,
                "new_page": new_page,
                "similarity": score,
                "status": "matched"
            }
            diffs.append(diff)

        elif old_page and not new_page:
            # Page was deleted - show all content as removed
            a_geoms = _load_page_geoms(conn, old_id, old_page)
            a_txt = _load_page_texts(conn, old_id, old_page)

            diff = {
                "page": old_page,
                "geometry": {
                    "added": [],
                    "removed": [g.bounds for g in a_geoms],
                    "changed": [],
                },
                "text": {
                    "added": [],
                    "removed": [{"text": t, "bbox": bb} for t, bb in a_txt],
                    "moved": [],
                },
                "alignment": {
                    "old_page": old_page,
                    "new_page": None,
                    "similarity": 0.0,
                    "status": "deleted"
                }
            }
            diffs.append(diff)

        elif not old_page and new_page:
            # Page was inserted - show all content as added
            b_geoms = _load_page_geoms(conn, new_id, new_page)
            b_txt = _load_page_texts(conn, new_id, new_page)

            diff = {
                "page": new_page,
                "geometry": {
                    "added": [g.bounds for g in b_geoms],
                    "removed": [],
                    "changed": [],
                },
                "text": {
                    "added": [{"text": t, "bbox": bb} for t, bb in b_txt],
                    "removed": [],
                    "moved": [],
                },
                "alignment": {
                    "old_page": None,
                    "new_page": new_page,
                    "similarity": 0.0,
                    "status": "inserted"
                }
            }
            diffs.append(diff)

    return diffs, alignments


def diff_pages(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    old_page: int,
    new_page: Optional[int] = None
) -> Dict:
    """
    Compare two specific pages (possibly from different page numbers).

    Args:
        old_id: Old document ID
        new_id: New document ID
        old_page: Page number in old document
        new_page: Page number in new document (defaults to old_page if None)

    This is the updated version that supports aligned comparison.
    The old diff_pages signature (3 args) is preserved for backward compatibility.
    """
    if new_page is None:
        new_page = old_page

    # Load content
    a_geoms = _load_page_geoms(conn, old_id, old_page)
    b_geoms = _load_page_geoms(conn, new_id, new_page)
    a_txt = _load_page_texts(conn, old_id, old_page)
    b_txt = _load_page_texts(conn, new_id, new_page)

    # Early no-op
    if not a_geoms and not b_geoms and not a_txt and not b_txt:
        return {
            "page": new_page,  # Use new_page for consistency
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
        "page": new_page,  # Use new_page for consistency
        "geometry": {
            "added": [g.bounds for g in added_geo],
            "removed": [g.bounds for g in removed_geo],
            "changed": [],  # reserved for raster modes; empty in vector/text diff
        },
        "text": {
            "added": added_text,
            "removed": removed_text,
            "moved": moved_text,
        },
    }
