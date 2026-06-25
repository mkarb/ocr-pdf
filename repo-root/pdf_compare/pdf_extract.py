# pdf_compare/pdf_extract.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
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

import fitz  # PyMuPDF

from ._extract_core import hash_file as _hash_file, drawings_to_geoms as _drawings_to_geoms
from .models import (
    VectorMap, DocMeta, PageVectors, VectorGeom, GeoKind, text_run_from_dict
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
# Geometry helpers (_hash_file, _drawings_to_geoms) live in _extract_core and are
# imported above so pdf_extract.py and pdf_extract_server.py share one implementation.


def _extract_text(page: "fitz.Page", pdf_path: Optional[str] = None, page_index: Optional[int] = None, enable_ocr: bool = False, ocr_dpi: int = 400, ocr_engine: Optional[str] = None, ocr_use_gpu: Optional[bool] = None) -> List[Dict[str, Any]]:
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

    # Debug logging for OCR decision
    if enable_ocr:
        import sys
        print(f"OCR: page {page_index+1 if page_index is not None else '?'}: Found {len(runs)} native text spans (threshold: 20)", file=sys.stderr)
        if len(runs) >= 20:
            print(f"OCR: page {page_index+1}: Skipping OCR (sufficient native text found)", file=sys.stderr)

    # Run OCR if enabled and little native text was found
    if enable_ocr and len(runs) < 20 and pdf_path and page_index is not None:
        try:
            from .analyzers import highres_ocr, tiled_ocr, HighResOCRConfig, resolve_ocr_engine

            # Resolve engine + GPU: explicit caller choice (e.g. the UI dropdown) wins,
            # then OCR_ENGINE/OCR_USE_GPU env, then CUDA autodetect; falls back to
            # Tesseract if EasyOCR isn't installed so OCR still runs on a non-GPU host.
            resolved_engine, resolved_gpu = resolve_ocr_engine(ocr_engine, ocr_use_gpu)

            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height

            # Use user-requested DPI (no capping here - tiled OCR handles it)
            dpi = ocr_dpi

            # Check if tiling is needed at requested DPI
            TESSERACT_PIXEL_LIMIT = 29000
            zoom = dpi / 72.0
            pixel_width = page_width * zoom
            pixel_height = page_height * zoom
            needs_tiling = (pixel_width > TESSERACT_PIXEL_LIMIT or pixel_height > TESSERACT_PIXEL_LIMIT)

            import sys
            if needs_tiling:
                # Use tiled OCR for large pages
                print(f"OCR: page {page_index+1} size={page_width:.0f}x{page_height:.0f} pts: Using tiled OCR at {dpi} DPI (page too large for single tile), engine={resolved_engine} gpu={resolved_gpu}", file=sys.stderr)

                ocr_results, report = tiled_ocr(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    dpi=dpi,
                    psm=11,
                    min_conf=50,  # Lowered from 60 to capture more text (some may be lower confidence)
                    overlap_pct=0.35,  # Increased from 0.20 to better capture text at boundaries
                    skip_empty=True,
                    return_report=True,
                    use_dual_psm=True,  # Use both PSM 11 and PSM 6 for better text capture (Tesseract only)
                    engine=resolved_engine,
                    use_gpu=resolved_gpu,
                )

                print(f"OCR: page {page_index+1}: Tiled OCR complete - {report.tiles_processed}/{report.total_tiles} tiles processed, {report.tiles_skipped_empty} skipped, {len(ocr_results)} text items, {report.duplicates_removed} duplicates removed", file=sys.stderr)

            else:
                # Use whole-page OCR for pages that fit
                print(f"OCR: page {page_index+1} size={page_width:.0f}x{page_height:.0f} pts: Using whole-page OCR at {dpi} DPI, engine={resolved_engine} gpu={resolved_gpu}", file=sys.stderr)

                config = HighResOCRConfig(dpi=dpi, psm=11, min_conf=50, engine=resolved_engine, use_gpu=resolved_gpu)
                ocr_results = highres_ocr(pdf_path, page_index, config)

                print(f"OCR: page {page_index+1}: Extracted {len(ocr_results)} text items", file=sys.stderr)

            # Add OCR results to runs
            for ocr_text in ocr_results:
                runs.append({
                    "text": ocr_text.get("text", ""),
                    "bbox": ocr_text.get("bbox", (0, 0, 0, 0)),
                    "font": None,
                    "size": None,
                    "source": "ocr"
                })

        except Exception as e:
            # Log OCR errors but don't break extraction
            import sys
            import traceback
            print(f"OCR warning for page {page_index}: {e}", file=sys.stderr)
            print(f"OCR traceback: {traceback.format_exc()}", file=sys.stderr)

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
    ocr_dpi: int = 400,
    ocr_engine: Optional[str] = None,
    ocr_use_gpu: Optional[bool] = None,
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
    texts = _extract_text(pg, pdf_path=pdf_path, page_index=page_index, enable_ocr=enable_ocr, ocr_dpi=ocr_dpi, ocr_engine=ocr_engine, ocr_use_gpu=ocr_use_gpu)

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
    ocr_dpi: int = 400,                       # DPI for OCR rendering (tiled OCR splits oversized pages)
    ocr_engine: Optional[str] = None,         # "easyocr"|"tesseract"; None=auto (env/CUDA)
    ocr_use_gpu: Optional[bool] = None,       # None=auto (CUDA autodetect for EasyOCR)
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
                    ocr_dpi=ocr_dpi,
                    ocr_engine=ocr_engine,
                    ocr_use_gpu=ocr_use_gpu,
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
                    ocr_dpi,
                    ocr_engine,
                    ocr_use_gpu,
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
        texts_dc = [text_run_from_dict(t) for t in r["texts"]]
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
