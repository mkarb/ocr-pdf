# pdf_compare/pdf_extract_server.py
"""
Server-optimized PDF extraction for Docker/container deployments.

Differences from pdf_extract.py:
- No Streamlit compatibility workarounds
- Environment-based configuration (12-factor app)
- Memory limit enforcement
- Structured logging with metrics
- Resource monitoring hooks
- Batch processing support
- Horizontal scaling friendly

Usage:
    from pdf_compare.pdf_extract_server import pdf_to_vectormap_server

    vm = pdf_to_vectormap_server(
        path="drawing.pdf",
        workers=0,  # 0=auto from CPU_LIMIT env var
        memory_limit_mb=2048,  # Per-worker memory limit
    )
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Callable
import hashlib
import os
import logging
import time
from pathlib import Path

from .worker_pool import (
    configure_thread_env,
    ThrottledPoolExecutor,
    worker_init,
    get_cached_doc,
    get_optimal_workers,
)

configure_thread_env()

import numpy as np
import fitz  # PyMuPDF
from shapely.geometry import LineString, box
from shapely.wkb import dumps as wkb_dumps

from .models import (
    VectorMap, DocMeta, PageVectors, VectorGeom, GeoKind, TextRun, BBox
)

# -----------------------
# Logging Setup
# -----------------------
logger = logging.getLogger(__name__)

# -----------------------
# Environment-based Configuration
# -----------------------
DEFAULT_ROTATIONS = {0, 90, 180, 270}

# Configuration from environment variables (12-factor app pattern)
DEF_MIN_SEGMENT_LEN = float(os.getenv("PDF_MIN_SEGMENT_LEN", "0.50"))
DEF_MIN_FILL_AREA = float(os.getenv("PDF_MIN_FILL_AREA", "0.50"))
DEF_BEZIER_SAMPLES = int(os.getenv("PDF_BEZIER_SAMPLES", "24"))
DEF_SIMPLIFY_TOL = float(os.getenv("PDF_SIMPLIFY_TOL", "0")) if os.getenv("PDF_SIMPLIFY_TOL") else None

# Worker configuration (managed by worker_pool module)
# See worker_pool.get_optimal_workers() for dynamic worker allocation


# -----------------------
# Helpers
# -----------------------
def _hash_file(path: Path) -> str:
    """SHA256 hash (first 16 chars) for stable doc_id."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cubic_sample(p0, p1, p2, p3, n: int) -> List[Tuple[float, float]]:
    """Sample a cubic Bezier curve with n points."""
    t = np.linspace(0.0, 1.0, n)
    xs = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
    ys = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    return list(zip(xs.tolist(), ys.tolist()))


def _adaptive_bezier_samples(p0, p1, p2, p3, base_samples: int) -> int:
    """
    Calculate adaptive sample count based on curve's bounding box diagonal.
    Longer curves get more samples; short curves get fewer.
    """
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    diagonal = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    samples = max(4, min(base_samples, int(diagonal / 2) + 1))
    return samples


def _drawings_to_geoms(
    page: "fitz.Page",
    min_segment_len: float,
    min_fill_area: float,
    bezier_samples: int,
    simplify_tolerance: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Extract stroke/fill geometries from a page and return as plain dicts ready for IPC.
    Each dict: {"kind": "STROKE"|"FILL", "wkb": bytes, "bbox": (x0,y0,x1,y1)}
    """
    out: List[Dict[str, Any]] = []

    for d in page.get_drawings():
        # strokes (path segments)
        for item in d["items"]:
            op = item[0]
            if op == "l":  # line (p0, p1)
                _, p0, p1 = item
                if p0 != p1:
                    ls = LineString([p0, p1])
                    if simplify_tolerance:
                        ls = ls.simplify(simplify_tolerance, preserve_topology=True)
                    if ls.length >= min_segment_len and not ls.is_empty:
                        out.append({
                            "kind": "STROKE",
                            "wkb": wkb_dumps(ls),
                            "bbox": ls.bounds,
                        })
            elif op == "c":  # cubic Bezier (p0,p1,p2,p3)
                _, p0, p1, p2, p3 = item
                n_samples = _adaptive_bezier_samples(p0, p1, p2, p3, bezier_samples)
                pts = _cubic_sample(p0, p1, p2, p3, n=n_samples)
                ls = LineString(pts)
                if simplify_tolerance:
                    ls = ls.simplify(simplify_tolerance, preserve_topology=True)
                if ls.length >= min_segment_len and not ls.is_empty:
                    out.append({
                        "kind": "STROKE",
                        "wkb": wkb_dumps(ls),
                        "bbox": ls.bounds,
                    })

        # simple rect fill
        if d.get("fill") and d.get("rect"):
            x0, y0, x1, y1 = d["rect"]
            poly = box(x0, y0, x1, y1)
            if poly.area >= min_fill_area and not poly.is_empty:
                out.append({
                    "kind": "FILL",
                    "wkb": wkb_dumps(poly),
                    "bbox": poly.bounds,
                })

    return out


def _extract_text(page: "fitz.Page") -> List[Dict[str, Any]]:
    """
    Extract native text spans as plain dicts: {"text": str, "bbox": (x0,y0,x1,y1), "font": str|None, "size": float|None}
    """
    runs: List[Dict[str, Any]] = []
    raw = page.get_text("dict") or {}  # ~14% faster than rawdict
    for blk in raw.get("blocks", []):
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                bbox = tuple(span["bbox"])  # type: ignore
                runs.append({"text": txt, "bbox": bbox, "font": span.get("font"), "size": span.get("size")})
    return runs


# Note: Worker initialization and document caching are now handled by worker_pool module


def _extract_page_job_server(
    pdf_path: str,
    page_index: int,
    min_segment_len: float,
    min_fill_area: float,
    bezier_samples: int,
    simplify_tolerance: Optional[float],
) -> Dict[str, Any]:
    """
    Server-optimized worker: Extract one page → return pure Python dict.
    Uses process-level cached document to avoid repeated file opens.
    Includes timing metrics for monitoring.
    """
    start_time = time.perf_counter()

    try:
        doc = get_cached_doc(pdf_path)
        pg = doc[page_index]
        rotation = pg.rotation if pg.rotation in DEFAULT_ROTATIONS else 0

        geoms = _drawings_to_geoms(
            pg,
            min_segment_len=min_segment_len,
            min_fill_area=min_fill_area,
            bezier_samples=bezier_samples,
            simplify_tolerance=simplify_tolerance,
        )
        texts = _extract_text(pg)

        elapsed = time.perf_counter() - start_time

        out = {
            "page_number": page_index + 1,
            "width": float(pg.rect.width),
            "height": float(pg.rect.height),
            "rotation": int(rotation),
            "geoms": geoms,
            "texts": texts,
            "metrics": {
                "elapsed_sec": elapsed,
                "geom_count": len(geoms),
                "text_count": len(texts),
            }
        }
        return out
    except Exception as e:
        logger.error(f"Failed to extract page {page_index} from {pdf_path}: {e}")
        raise


# -----------------------
# Public API
# -----------------------
def pdf_to_vectormap_server(
    path: str,
    doc_id: str | None = None,
    *,
    workers: int = 0,  # 0=auto from CPU_LIMIT
    min_segment_len: float = DEF_MIN_SEGMENT_LEN,
    min_fill_area: float = DEF_MIN_FILL_AREA,
    bezier_samples: int = DEF_BEZIER_SAMPLES,
    simplify_tolerance: Optional[float] = DEF_SIMPLIFY_TOL,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VectorMap:
    """
    Server-optimized parallel PDF extraction for Docker/container environments.

    Args:
        path: Path to PDF file
        doc_id: Optional document ID (defaults to file hash)
        workers: Number of worker processes (0=auto from CPU_LIMIT env var)
        min_segment_len: Minimum line segment length to keep
        min_fill_area: Minimum fill area to keep
        bezier_samples: Base samples for Bezier curves (adaptive)
        simplify_tolerance: Geometry simplification tolerance (None=disabled)
        progress_callback: Optional callback(completed_pages, total_pages)

    Returns:
        VectorMap with extracted geometry and text

    Environment Variables:
        CPU_LIMIT: Max CPU cores to use (default: os.cpu_count())
        PDF_MIN_SEGMENT_LEN: Default min segment length (default: 0.50)
        PDF_MIN_FILL_AREA: Default min fill area (default: 0.50)
        PDF_BEZIER_SAMPLES: Default Bezier samples (default: 24)
        PDF_SIMPLIFY_TOL: Default simplify tolerance (default: None)
    """
    start_time = time.perf_counter()
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Determine page_count without keeping the file handle open
    d = fitz.open(path)
    page_count = d.page_count
    d.close()

    if doc_id is None:
        doc_id = _hash_file(p)

    logger.info(f"Starting extraction: doc_id={doc_id}, pages={page_count}, path={path}")

    # Determine optimal worker count
    workers = get_optimal_workers(
        requested_workers=workers,
        page_count=page_count,
        enable_ocr=False,  # Server mode doesn't use OCR
        force_serial=False,
    )

    logger.info(f"Using {workers} worker(s) for {page_count} page(s)")

    page_dicts: List[Dict[str, Any]] = []

    if workers == 1:
        # Serial fallback
        for i in range(page_count):
            page_dicts.append(
                _extract_page_job_server(
                    str(p),
                    i,
                    min_segment_len=min_segment_len,
                    min_fill_area=min_fill_area,
                    bezier_samples=bezier_samples,
                    simplify_tolerance=simplify_tolerance,
                )
            )
            if progress_callback:
                progress_callback(i + 1, page_count)
    else:
        # Use ThrottledPoolExecutor for memory-bounded parallel processing
        with ThrottledPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            # Helper to convert page_index to worker arguments
            def item_to_args(page_idx: int) -> tuple:
                return (
                    str(p),
                    page_idx,
                    min_segment_len,
                    min_fill_area,
                    bezier_samples,
                    simplify_tolerance,
                )

            page_dicts = pool.submit_throttled(
                worker_func=_extract_page_job_server,
                items=range(page_count),
                progress_callback=progress_callback,
                item_to_args=item_to_args,
            )

    # Convert dicts → dataclasses and sort by page_number
    pages: List[PageVectors] = []
    total_geoms = 0
    total_texts = 0

    for r in page_dicts:
        metrics = r.pop("metrics", {})
        total_geoms += metrics.get("geom_count", 0)
        total_texts += metrics.get("text_count", 0)

        geoms_dc = [
            VectorGeom(
                kind=GeoKind[g["kind"]],
                wkb=g["wkb"],
                bbox=tuple(g["bbox"]),
            )
            for g in r["geoms"]
        ]
        texts_dc = [
            TextRun(
                text=t["text"],
                bbox=tuple(t["bbox"]),
                font=t.get("font"),
                size=t.get("size"),
            )
            for t in r["texts"]
        ]
        pages.append(
            PageVectors(
                page_number=r["page_number"],
                width=r["width"],
                height=r["height"],
                rotation=r["rotation"],
                geoms=geoms_dc,
                texts=texts_dc,
            )
        )

    pages.sort(key=lambda pg: pg.page_number)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Extraction complete: doc_id={doc_id}, "
        f"pages={page_count}, geoms={total_geoms}, texts={total_texts}, "
        f"elapsed={elapsed:.2f}s, pages_per_sec={page_count/elapsed:.2f}"
    )

    meta = DocMeta(doc_id=doc_id, path=str(p.resolve()), page_count=page_count)
    return VectorMap(meta=meta, pages=pages)


# Alias for backwards compatibility with pdf_extract.py
pdf_to_vectormap = pdf_to_vectormap_server
