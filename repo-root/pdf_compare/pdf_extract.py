# pdf_compare/pdf_extract.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict
import hashlib
import os
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
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely.wkb import dumps as wkb_dumps

from .models import (
    VectorMap, DocMeta, PageVectors, VectorGeom, GeoKind, TextRun, BBox
)

# -----------------------
# Defaults / Tunables
# -----------------------
DEFAULT_ROTATIONS = {0, 90, 180, 270}

# Sensible defaults (can be overridden via pdf_to_vectormap args)
DEF_MIN_SEGMENT_LEN = 0.50     # drop ultra-short stroke segments (< 0.5 user units)
DEF_MIN_FILL_AREA   = 0.50     # drop tiny filled rects (< 0.5 sq units)
DEF_BEZIER_SAMPLES  = 24       # samples per cubic segment (higher = smoother)
DEF_SIMPLIFY_TOL    = None     # e.g., 0.05..0.15 to reduce oversampled paths

# Worker configuration (managed by worker_pool module)
# See worker_pool.get_optimal_workers() for dynamic worker allocation


# -----------------------
# Helpers
# -----------------------
def _hash_file(path: Path) -> str:
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
    # Estimate curve extent via control point bounding box
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    diagonal = np.hypot(max(xs) - min(xs), max(ys) - min(ys))

    # Scale samples: 1 sample per ~2 units of diagonal length, min 4, max base_samples
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
                # Use adaptive sampling: short curves get fewer samples
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


def _extract_text(page: "fitz.Page", pdf_path: Optional[str] = None, page_index: Optional[int] = None, enable_ocr: bool = False) -> List[Dict[str, Any]]:
    """
    Extract native text spans as plain dicts: {"text": str, "bbox": (x0,y0,x1,y1), "font": str|None, "size": float|None}
    Optionally runs OCR if enable_ocr=True and minimal native text is found.
    """
    runs: List[Dict[str, Any]] = []
    raw = page.get_text("dict") or {}
    for blk in raw.get("blocks", []):
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                bbox = tuple(span["bbox"])  # type: ignore
                runs.append({"text": txt, "bbox": bbox, "font": span.get("font"), "size": span.get("size"), "source": "native"})

    # Run OCR if enabled and little native text was found
    if enable_ocr and len(runs) < 20 and pdf_path and page_index is not None:
        try:
            from .analyzers import highres_ocr, HighResOCRConfig

            # Calculate dynamic DPI based on page dimensions
            # Large engineering drawings need higher DPI for small text
            page_width = page.rect.width
            page_height = page.rect.height
            page_diagonal = (page_width ** 2 + page_height ** 2) ** 0.5

            # DPI scaling strategy:
            # - Small pages (letter/A4, ~800-1200pts diagonal): 400 DPI
            # - Medium pages (tabloid, ~1200-1800pts): 500 DPI
            # - Large pages (engineering drawings, >1800pts): 600-800 DPI
            # - Extra large pages (>3000pts): 800+ DPI
            if page_diagonal < 1200:
                dpi = 400
            elif page_diagonal < 1800:
                dpi = 500
            elif page_diagonal < 2500:
                dpi = 600
            elif page_diagonal < 3500:
                dpi = 700
            else:
                dpi = 800

            # PSM 11 = Sparse text, finds as much text as possible (best for drawings)
            config = HighResOCRConfig(dpi=dpi, psm=11, min_conf=60)

            import sys
            print(f"OCR: page {page_index+1} size={page_width:.0f}x{page_height:.0f} pts, diagonal={page_diagonal:.0f}, using DPI={dpi}", file=sys.stderr)

            ocr_results = highres_ocr(pdf_path, page_index, config)

            # Add OCR results
            for ocr_text in ocr_results:
                runs.append({
                    "text": ocr_text["text"],
                    "bbox": ocr_text["bbox"],
                    "font": None,
                    "size": None,
                    "source": "ocr"
                })

            print(f"OCR: extracted {len(ocr_results)} text items from page {page_index+1}", file=sys.stderr)

        except Exception as e:
            # Log OCR errors but don't break extraction
            import sys
            print(f"OCR warning for page {page_index}: {e}", file=sys.stderr)

    return runs


# Note: Worker initialization and document caching are now handled by worker_pool module

def _extract_page_job(
    pdf_path: str,
    page_index: int,
    min_segment_len: float,
    min_fill_area: float,
    bezier_samples: int,
    simplify_tolerance: Optional[float],
    enable_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Isolated worker: Extract one page → return pure Python dict.
    Uses process-level cached document to avoid repeated file opens.
    Avoids passing PyMuPDF/Shapely objects across processes.
    """
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
    texts = _extract_text(pg, pdf_path=pdf_path, page_index=page_index, enable_ocr=enable_ocr)

    out = {
        "page_number": page_index + 1,
        "width": float(pg.rect.width),
        "height": float(pg.rect.height),
        "rotation": int(rotation),
        "geoms": geoms,   # list of dicts
        "texts": texts,   # list of dicts
    }
    # Note: Don't close doc - it's cached for reuse
    return out


# -----------------------
# Public API
# -----------------------
def pdf_to_vectormap(
    path: str,
    doc_id: str | None = None,
    *,
    workers: int = 0,                         # 0=auto (cores-1)
    min_segment_len: float = DEF_MIN_SEGMENT_LEN,
    min_fill_area: float = DEF_MIN_FILL_AREA,
    bezier_samples: int = DEF_BEZIER_SAMPLES,
    simplify_tolerance: Optional[float] = DEF_SIMPLIFY_TOL,
    enable_ocr: bool = False,                 # Enable OCR for engineering drawings
) -> VectorMap:
    """
    Parallel, high-throughput ingest.
    - workers: number of processes to use (0 = auto; 1 = serial)
    - min_segment_len: drop tiny line segments
    - min_fill_area: drop tiny rect fills
    - bezier_samples: sampling density for cubic curves
    - simplify_tolerance: if set, simplifies strokes to reduce oversampled paths
    - enable_ocr: if True, runs OCR on pages with minimal native text
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Determine page_count without keeping the file handle open
    d = fitz.open(path)
    page_count = d.page_count
    d.close()

    if doc_id is None:
        doc_id = _hash_file(p)

    # Detect Streamlit environment - force serial for small docs or OCR
    force_serial = False
    try:
        import streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            # Streamlit + OCR + multiprocessing = context issues → force serial
            # Also force serial for very small docs (1-2 pages)
            if enable_ocr or page_count <= 2:
                force_serial = True
    except (ImportError, AttributeError):
        pass  # Not running under Streamlit

    # Determine optimal worker count
    workers = get_optimal_workers(
        requested_workers=workers,
        page_count=page_count,
        enable_ocr=enable_ocr,
        force_serial=force_serial,
    )

    # Log worker allocation
    import sys
    if enable_ocr:
        ocr_mode = f" (OCR enabled, DPI will be auto-adjusted per page)"
    else:
        ocr_mode = ""
    print(f"Processing {page_count} page(s) with {workers} worker(s){ocr_mode}", file=sys.stderr)

    # Map pages in parallel (Windows-safe spawn context)
    page_dicts: List[Dict[str, Any]] = []
    if workers == 1:
        # serial fallback
        for i in range(page_count):
            page_dicts.append(
                _extract_page_job(
                    str(p),
                    i,
                    min_segment_len=min_segment_len,
                    min_fill_area=min_fill_area,
                    bezier_samples=bezier_samples,
                    simplify_tolerance=simplify_tolerance,
                    enable_ocr=enable_ocr,
                )
            )
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
                    enable_ocr,
                )

            page_dicts = pool.submit_throttled(
                worker_func=_extract_page_job,
                items=range(page_count),
                progress_callback=None,  # No callback for client mode
                item_to_args=item_to_args,
            )

    # Convert dicts → dataclasses and sort by page_number
    pages: List[PageVectors] = []
    for r in page_dicts:
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

    meta = DocMeta(doc_id=doc_id, path=str(p.resolve()), page_count=page_count)
    return VectorMap(meta=meta, pages=pages)
