# pdf_compare/_extract_core.py
"""
Shared low-level helpers for PDF vector extraction.

Used by both ``pdf_extract.py`` (interactive/Streamlit) and
``pdf_extract_server.py`` (container/server) so the geometry-extraction logic
lives in exactly one place. Kept dependency-light: PyMuPDF page objects in,
plain picklable dicts out — no OCR, logging, or multiprocessing policy here.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import hashlib
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, box
from shapely.wkb import dumps as wkb_dumps


def hash_file(path: Path) -> str:
    """SHA256 hash (first 16 chars) for a stable doc_id."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cubic_sample(p0, p1, p2, p3, n: int) -> List[Tuple[float, float]]:
    """Sample a cubic Bezier curve with n points."""
    t = np.linspace(0.0, 1.0, n)
    xs = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    ys = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return list(zip(xs.tolist(), ys.tolist()))


def adaptive_bezier_samples(p0, p1, p2, p3, base_samples: int) -> int:
    """
    Calculate adaptive sample count based on curve's bounding box diagonal.
    Longer curves get more samples; short curves get fewer.
    """
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    diagonal = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    samples = max(4, min(base_samples, int(diagonal / 2) + 1))
    return samples


def _append_stroke(out: List[Dict[str, Any]], ls: LineString,
                   min_segment_len: float, simplify_tolerance: Optional[float]) -> None:
    """Simplify (optional), length-filter, and append a stroke LineString."""
    if simplify_tolerance:
        ls = ls.simplify(simplify_tolerance, preserve_topology=True)
    if ls.length >= min_segment_len and not ls.is_empty:
        out.append({"kind": "STROKE", "wkb": wkb_dumps(ls), "bbox": ls.bounds})


def drawings_to_geoms(
    page: "fitz.Page",
    min_segment_len: float,
    min_fill_area: float,
    bezier_samples: int,
    simplify_tolerance: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Extract stroke/fill geometries from a page and return as plain dicts ready for IPC.
    Each dict: {"kind": "STROKE"|"FILL", "wkb": bytes, "bbox": (x0,y0,x1,y1)}

    Handles all four PyMuPDF path ops: "l" (line), "c" (cubic Bezier),
    "re" (rectangle), and "qu" (quad).
    """
    out: List[Dict[str, Any]] = []

    for d in page.get_drawings():
        # strokes (path segments)
        for item in d["items"]:
            op = item[0]
            if op == "l":  # line (p0, p1)
                _, p0, p1 = item
                if p0 != p1:
                    _append_stroke(out, LineString([p0, p1]), min_segment_len, simplify_tolerance)
            elif op == "c":  # cubic Bezier (p0,p1,p2,p3)
                _, p0, p1, p2, p3 = item
                n_samples = adaptive_bezier_samples(p0, p1, p2, p3, bezier_samples)
                pts = cubic_sample(p0, p1, p2, p3, n=n_samples)
                _append_stroke(out, LineString(pts), min_segment_len, simplify_tolerance)
            elif op == "re":  # rectangle: ("re", rect[, orientation])
                x0, y0, x1, y1 = item[1]
                ring = LineString([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
                _append_stroke(out, ring, min_segment_len, simplify_tolerance)
            elif op == "qu":  # quad: ("qu", quad) with corners ul/ur/lr/ll
                q = item[1]
                ring = LineString([tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll), tuple(q.ul)])
                _append_stroke(out, ring, min_segment_len, simplify_tolerance)

        # simple rect fill
        if d.get("fill") and d.get("rect"):
            x0, y0, x1, y1 = d["rect"]
            poly = box(x0, y0, x1, y1)
            if poly.area >= min_fill_area and not poly.is_empty:
                out.append({"kind": "FILL", "wkb": wkb_dumps(poly), "bbox": poly.bounds})

    return out
