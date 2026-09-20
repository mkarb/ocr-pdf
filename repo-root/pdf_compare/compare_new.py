"""
Document comparison with SQLAlchemy backend support.
"""

from __future__ import annotations
from collections import defaultdict, deque
from math import hypot
from numbers import Integral
from typing import List, Dict
import numpy as np
from shapely.wkb import loads as wkb_loads
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException

from .db_backend import DatabaseBackend

# Tolerances
GEO_TOL: float = 0.15
TEXT_MOVE_TOL: float = 0.75


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


def _geom_matches(
    geom: BaseGeometry,
    candidates: List[BaseGeometry],
    buffered: BaseGeometry,
) -> bool:
    """Require every part of both geometries to lie within GEO_TOL.

    Comparing buffered areas, or accepting containment in either direction,
    hides shortened strokes and resized fills. Each original geometry must
    instead be covered by the other's tolerance buffer.
    """
    if geom is None or geom.is_empty:
        return False
    for h in candidates:
        if h is None or h.is_empty:
            continue
        try:
            if buffered.covers(h) and h.buffer(GEO_TOL).covers(geom):
                return True
        except GEOSException:
            continue
    return False


def diff_documents(
    backend: DatabaseBackend,
    old_id: str,
    new_id: str,
    pages: List[int] | None = None
) -> List[Dict]:
    """
    Compare two ingested documents page by page (vector + text).
    Returns a list of per-page diff dicts compatible with overlay.py.
    Includes trailing pages present in only one document, including blank pages.
    """
    if not isinstance(backend, DatabaseBackend):
        raise TypeError("diff_documents requires a DatabaseBackend (SQLite support removed)")

    # Get page counts from backend
    with backend.SessionLocal() as session:
        from .db_models import Document, Page
        old_doc = session.get(Document, old_id)
        new_doc = session.get(Document, new_id)

        if not old_doc or not new_doc:
            raise ValueError("Unknown doc_id(s) supplied to diff_documents")

        pc_old, pc_new = old_doc.page_count, new_doc.page_count

        max_pages = max(pc_old, pc_new)
        if pages is None:
            pages = list(range(1, max_pages + 1))
        else:
            pages = list(pages)
        for page in pages:
            if isinstance(page, bool) or not isinstance(page, Integral) or not 1 <= page <= max_pages:
                raise ValueError(f"Invalid page {page!r}; expected an integer between 1 and {max_pages}")

        page_metadata = {}
        for page in pages:
            if page <= min(pc_old, pc_new):
                continue
            added = page > pc_old
            metadata = {"page_status": "added" if added else "removed"}
            stored_page = session.get(Page, (new_id if added else old_id, page))
            if stored_page is not None and stored_page.width and stored_page.height:
                metadata["page_size"] = (stored_page.width, stored_page.height)
            page_metadata[page] = metadata

    diffs = []
    for page in pages:
        diff = _diff_page_contents(backend, old_id, new_id, page)
        diff.update(page_metadata.get(page, {}))
        diffs.append(diff)
    return diffs


def diff_pages(
    backend: DatabaseBackend,
    old_id: str,
    new_id: str,
    page: int
) -> Dict:
    """Compare one validated page, including a page present in only one PDF."""
    return diff_documents(backend, old_id, new_id, pages=[page])[0]


def _text_bbox_distance(a, b):
    """Measure movement of both corners so centered resizing is detected too."""
    return max(hypot(a[0] - b[0], a[1] - b[1]), hypot(a[2] - b[2], a[3] - b[3]))


def _match_texts(a_txt, b_txt):
    """Match identical text, reserving unchanged occurrences before moved ones."""
    exact_b = defaultdict(deque)
    for j, (text, bbox) in enumerate(b_txt):
        exact_b[(text, tuple(bbox))].append(j)

    matches = {}
    used_b = set()
    for i, (text, bbox) in enumerate(a_txt):
        exact = exact_b.get((text, tuple(bbox)))
        if exact:
            j = exact.popleft()
            matches[i] = j
            used_b.add(j)

    remaining_a = defaultdict(list)
    remaining_b = defaultdict(list)
    for i, (text, bbox) in enumerate(a_txt):
        if i not in matches:
            remaining_a[text].append(i)
    for j, (text, bbox) in enumerate(b_txt):
        if j not in used_b:
            remaining_b[text].append(j)

    # Match the closest remaining pair first, independent of database row order.
    # Exact matches above avoid quadratic work for unchanged repeated labels.
    for text, old_indices in remaining_a.items():
        candidates = sorted(
            (_text_bbox_distance(a_txt[i][1], b_txt[j][1]), i, j)
            for i in old_indices for j in remaining_b.get(text, [])
        )
        for _, i, j in candidates:
            if i not in matches and j not in used_b:
                matches[i] = j
                used_b.add(j)
    return matches, used_b


def _diff_page_contents(backend, old_id, new_id, page):
    """Compare stored content after document and page selection validation."""

    # Load geometries
    a_geom_wkbs = backend.load_page_geoms(old_id, page)
    b_geom_wkbs = backend.load_page_geoms(new_id, page)
    a_geoms = [g for wkb in a_geom_wkbs if not (g := wkb_loads(wkb)).is_empty]
    b_geoms = [g for wkb in b_geom_wkbs if not (g := wkb_loads(wkb)).is_empty]

    # Load text
    a_txt = backend.load_page_texts(old_id, page)
    b_txt = backend.load_page_texts(new_id, page)

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

    tree_a = STRtree(a_geoms) if a_geoms else None
    tree_b = STRtree(b_geoms) if b_geoms else None

    removed_geo, added_geo = [], []

    # removed: A not in B
    for g, gb in zip(a_geoms, a_buf):
        hits = _query_hits(tree_b, gb, b_geoms)
        if not hits or not _geom_matches(g, hits, gb):
            removed_geo.append(g)

    # added: B not in A
    for g, gb in zip(b_geoms, b_buf):
        hits = _query_hits(tree_a, gb, a_geoms)
        if not hits or not _geom_matches(g, hits, gb):
            added_geo.append(g)

    # Text diff
    matches, used_b = _match_texts(a_txt, b_txt)
    removed_text: List[Dict] = []
    added_text: List[Dict] = []
    moved_text: List[Dict] = []

    for i, (t, bb) in enumerate(a_txt):
        match_j = matches.get(i)
        if match_j is None:
            removed_text.append({"text": t, "bbox": bb})
        else:
            if _text_bbox_distance(bb, b_txt[match_j][1]) > TEXT_MOVE_TOL:
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
