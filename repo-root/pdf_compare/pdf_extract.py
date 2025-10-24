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
from .analyzers.qwen_prompts import DEFAULT_PROMPT_MODE as DEFAULT_QWEN_PROMPT_MODE

# -----------------------
# Defaults / Tunables
# -----------------------
DEFAULT_ROTATIONS = {0, 90, 180, 270}

# Sensible defaults (can be overridden via pdf_to_vectormap args)
DEF_MIN_SEGMENT_LEN = 0.2     # drop ultra-short stroke segments (< 0.5 user units)
DEF_MIN_FILL_AREA   = 0.2     # drop tiny filled rects (< 0.5 sq units)
DEF_BEZIER_SAMPLES  = 24       # samples per cubic segment (higher = smoother)
DEF_SIMPLIFY_TOL    = None     # e.g., 0.05..0.15 to reduce oversampled paths


def _default_debug_output_dir() -> str:
    """Resolve default OCR debug output directory."""
    env_override = os.getenv("OCR_DEBUG_OUTPUT_DIR")
    if env_override:
        return os.path.expanduser(env_override)

    base_dir = Path(os.getenv("APP_DATA_DIR", Path.cwd() / "data"))
    return str((base_dir / "outputs" / "ocr-debug").absolute())

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


def _extract_text(
    page: "fitz.Page",
    pdf_path: Optional[str] = None,
    page_index: Optional[int] = None,
    enable_ocr: bool = False,
    ocr_dpi: int = 400,
    ocr_engine: str = "easyocr",
    debug_ocr: bool = False,
    debug_output_dir: Optional[str] = None,
    debug_conf_low: int = 65,
    debug_conf_high: int = 92,
    enable_layout_detection: bool = False,
    table_dpi_boost: float = 1.5,
    enable_upscaling: bool = False,
    upscale_factor: float = 1.5,
    qwen_prompt_mode: str = DEFAULT_QWEN_PROMPT_MODE,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Extract native text spans as plain dicts: {"text": str, "bbox": (x0,y0,x1,y1), "font": str|None, "size": float|None}
    Optionally runs OCR if enable_ocr=True and minimal native text is found.

    Args:
        ocr_engine: OCR engine to use - "tesseract", "easyocr", or "qwen-vl"
        debug_ocr: if True, outputs visual debugging images
        debug_output_dir: directory to save debug images
        debug_conf_low: low confidence threshold for visualization
        debug_conf_high: high confidence threshold for visualization
        enable_layout_detection: Run layout-aware OCR heuristics (tables/diagrams)
        table_dpi_boost: DPI multiplier for regions classified as tables
        enable_upscaling: Upscale tiles before OCR to enhance small text
        upscale_factor: Upscaling multiplier when enable_upscaling is True
        qwen_prompt_mode: Prompt preset to use when ocr_engine == "qwen-vl"
    """
    runs: List[Dict[str, Any]] = []
    page_ocr_stats: Optional[Dict[str, Any]] = None
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
            from .analyzers import HighResOCRConfig, TiledOCRDebugContext, highres_ocr, tiled_ocr

            # Create debug visualizer if requested
            debug_visualizer = None
            if debug_ocr:
                try:
                    from .debug import create_visualizer
                    debug_output_dir_final = (
                        os.path.expanduser(debug_output_dir)
                        if debug_output_dir
                        else _default_debug_output_dir()
                    )
                    debug_visualizer = create_visualizer(
                        enabled=True,
                        output_dir=debug_output_dir_final,
                        conf_low=debug_conf_low,
                        conf_high=debug_conf_high,
                    )
                    import sys
                    print(f"[OCR DEBUG] Created visualizer with output_dir={debug_output_dir_final}, enabled=True", file=sys.stderr, flush=True)
                except ImportError as e:
                    import sys
                    print(f"Warning: OCR debug mode requested but debug module not available: {e}", file=sys.stderr, flush=True)
                except Exception as e:
                    import sys
                    print(f"Warning: Failed to create debug visualizer: {e}", file=sys.stderr, flush=True)

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
                print(f"OCR: page {page_index+1} size={page_width:.0f}x{page_height:.0f} pts: Using tiled OCR at {dpi} DPI (page too large for single tile)", file=sys.stderr)

                min_conf_threshold = 50  # Lowered from 60 to capture more text (some may be lower confidence)
                debug_context = TiledOCRDebugContext(debug_visualizer)
                ocr_results, report = tiled_ocr(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    dpi=dpi,
                    psm=11,
                    min_conf=min_conf_threshold,
                    overlap_pct=0.35,  # Increased from 0.20 to better capture text at boundaries
                    skip_empty=False,
                    max_workers=1,  # Auto-calculated in tiled_ocr (will use optimal workers for Tesseract/EasyOCR)
                    return_report=True,
                    use_dual_psm=True,  # Use both PSM 11 and PSM 6 for better text capture (Tesseract only)
                    engine=ocr_engine,  # Use user-selected OCR engine
                    use_gpu=True,       # Enable GPU acceleration
                    debug_visualizer=debug_context,
                    enable_layout_detection=enable_layout_detection,
                    table_dpi_boost=table_dpi_boost,
                    enable_upscaling=enable_upscaling,
                    upscale_factor=upscale_factor,
                    qwen_prompt_mode=qwen_prompt_mode,
                )

                engine_summary = ", ".join(f"{k}:{v}" for k, v in sorted(report.engine_counts.items())) or "n/a"
                print(
                    f"OCR: page {page_index+1}: Tiled OCR complete - "
                    f"{report.tiles_processed}/{report.total_tiles} tiles processed, "
                    f"{report.tiles_skipped_empty} skipped, "
                    f"{len(ocr_results)} text items, "
                    f"avg_conf={report.avg_confidence:.1f}, "
                    f"low<{min_conf_threshold}={report.low_confidence_results}, "
                    f"{report.duplicates_removed} duplicates removed, "
                    f"engines=[{engine_summary}]",
                    file=sys.stderr,
                )

                page_ocr_stats = {
                    "mode": "tiled",
                    "tiles_processed": report.tiles_processed,
                    "tiles_total": report.total_tiles,
                    "tiles_skipped": report.tiles_skipped_empty,
                    "avg_confidence": report.avg_confidence,
                    "median_confidence": report.median_confidence,
                    "min_confidence": report.min_confidence,
                    "max_confidence": report.max_confidence,
                    "low_confidence_count": report.low_confidence_results,
                    "engine_counts": report.engine_counts,
                    "duplicates_removed": report.duplicates_removed,
                    "processing_time_sec": report.processing_time,
                    "min_conf_threshold": min_conf_threshold,
                }

            else:
                # Use whole-page OCR for pages that fit
                print(f"OCR: page {page_index+1} size={page_width:.0f}x{page_height:.0f} pts: Using whole-page OCR at {dpi} DPI", file=sys.stderr)

                config = HighResOCRConfig(
                    dpi=dpi,
                    psm=11,
                    min_conf=50,
                    engine=ocr_engine,
                    use_gpu=True,
                    qwen_prompt_mode=qwen_prompt_mode,
                )
                ocr_results = highres_ocr(pdf_path, page_index, config, debug_visualizer=debug_visualizer)

                print(f"OCR: page {page_index+1}: Extracted {len(ocr_results)} text items", file=sys.stderr)

                confidences = [
                    int(r.get("conf", 0))
                    for r in ocr_results
                    if r.get("conf") is not None
                ]
                if confidences:
                    avg_confidence = float(np.mean(confidences))
                    median_confidence = float(np.median(confidences))
                    min_confidence_val = int(np.min(confidences))
                    max_confidence_val = int(np.max(confidences))
                    low_confidence_count = sum(1 for c in confidences if c < config.min_conf)
                else:
                    avg_confidence = median_confidence = 0.0
                    min_confidence_val = max_confidence_val = 0
                    low_confidence_count = 0

                engine_counts = {}
                for r in ocr_results:
                    engine_id = r.get("source") or config.engine
                    engine_counts[engine_id] = engine_counts.get(engine_id, 0) + 1

                page_ocr_stats = {
                    "mode": "single-pass",
                    "avg_confidence": avg_confidence,
                    "median_confidence": median_confidence,
                    "min_confidence": min_confidence_val,
                    "max_confidence": max_confidence_val,
                    "low_confidence_count": low_confidence_count,
                    "engine_counts": engine_counts,
                    "processing_time_sec": None,
                    "min_conf_threshold": config.min_conf,
                }

            # Add OCR results to runs
            for ocr_text in ocr_results:
                # Preserve confidence from OCR engine (standardize field names)
                confidence = ocr_text.get("conf") or ocr_text.get("confidence")
                if confidence is not None and isinstance(confidence, float):
                    # Convert float (0.0-1.0) to int (0-100) if needed
                    confidence = int(confidence * 100) if confidence <= 1.0 else int(confidence)

                runs.append({
                    "text": ocr_text.get("text", ""),
                    "bbox": ocr_text.get("bbox", (0, 0, 0, 0)),
                    "font": None,
                    "size": None,
                    "source": ocr_text.get("source", "ocr"),  # Preserve OCR engine name
                    "confidence": confidence  # Preserve OCR confidence (0-100)
                })

        except Exception as e:
            # Log OCR errors but don't break extraction
            import sys
            import traceback
            print(f"OCR warning for page {page_index}: {e}", file=sys.stderr)
            print(f"OCR traceback: {traceback.format_exc()}", file=sys.stderr)

    return runs, page_ocr_stats


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
    ocr_engine: str = "easyocr",
    debug_ocr: bool = False,
    debug_output_dir: Optional[str] = None,
    debug_conf_low: int = 70,
    debug_conf_high: int = 90,
    enable_layout_detection: bool = False,
    table_dpi_boost: float = 1.5,
    enable_upscaling: bool = False,
    upscale_factor: float = 1.5,
    qwen_prompt_mode: str = DEFAULT_QWEN_PROMPT_MODE,
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
    texts, ocr_stats = _extract_text(
        pg,
        pdf_path=pdf_path,
        page_index=page_index,
        enable_ocr=enable_ocr,
        ocr_dpi=ocr_dpi,
        ocr_engine=ocr_engine,
        debug_ocr=debug_ocr,
        debug_output_dir=debug_output_dir,
        debug_conf_low=debug_conf_low,
        debug_conf_high=debug_conf_high,
        enable_layout_detection=enable_layout_detection,
        table_dpi_boost=table_dpi_boost,
        enable_upscaling=enable_upscaling,
        upscale_factor=upscale_factor,
        qwen_prompt_mode=qwen_prompt_mode,
    )

    out = {
        "page_number": page_index + 1,
        "width": float(pg.rect.width),
        "height": float(pg.rect.height),
        "rotation": int(rotation),
        "geoms": geoms,   # list of dicts
        "texts": texts,   # list of dicts
        "ocr_stats": ocr_stats,
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
    ocr_dpi: int = 400,                       # User-requested DPI for OCR (will be auto-capped)
    ocr_engine: str = "easyocr",              # OCR engine: "tesseract", "easyocr", or "qwen-vl"
    debug_ocr: bool = False,                  # Enable OCR visual debugging
    debug_output_dir: Optional[str] = None,   # Output directory for debug images
    debug_conf_low: int = 70,                 # Low confidence threshold for visualization
    debug_conf_high: int = 90,                # High confidence threshold for visualization
    enable_layout_detection: bool = False,    # Enable layout-aware OCR (tables/diagrams)
    table_dpi_boost: float = 1.5,             # DPI multiplier for table regions
    enable_upscaling: bool = False,           # Enable upscaling preprocessing for small text
    upscale_factor: float = 1.5,              # Upscaling multiplier (1.5-2.0 recommended)
    qwen_prompt_mode: str = DEFAULT_QWEN_PROMPT_MODE,         # Prompt preset when using Qwen-VL
) -> VectorMap:
    """
    Parallel, high-throughput ingest.
    - workers: number of processes to use (0 = auto; 1 = serial)
    - min_segment_len: drop tiny line segments
    - min_fill_area: drop tiny rect fills
    - bezier_samples: sampling density for cubic curves
    - simplify_tolerance: if set, simplifies strokes to reduce oversampled paths
    - enable_ocr: if True, runs OCR on pages with minimal native text
    - ocr_engine: OCR engine to use - "tesseract", "easyocr", or "qwen-vl"
    - debug_ocr: if True, outputs visual debugging images at each OCR stage
    - debug_output_dir: directory to save debug images (default: APP_DATA_DIR/outputs/ocr-debug)
    - debug_conf_low: low confidence threshold for visualization (default: 70)
    - debug_conf_high: high confidence threshold for visualization (default: 90)
    - enable_layout_detection: enable layout-aware heuristics (tables/diagrams)
    - table_dpi_boost: DPI multiplier applied to table regions when layout detection is enabled
    - enable_upscaling: upscale tiles before OCR to enhance small text
    - upscale_factor: scaling multiplier when upscaling is enabled
    - qwen_prompt_mode: Qwen2-VL prompt preset ("sparse", "layout", or "table")
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
                    debug_ocr=debug_ocr,
                    debug_output_dir=debug_output_dir,
                    debug_conf_low=debug_conf_low,
                    debug_conf_high=debug_conf_high,
                    enable_layout_detection=enable_layout_detection,
                    table_dpi_boost=table_dpi_boost,
                    enable_upscaling=enable_upscaling,
                    upscale_factor=upscale_factor,
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
                    debug_ocr,
                    debug_output_dir,
                    debug_conf_low,
                    debug_conf_high,
                    enable_layout_detection,
                    table_dpi_boost,
                    enable_upscaling,
                    upscale_factor,
                    qwen_prompt_mode,
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
                confidence=t.get("confidence"),  # Preserve confidence from OCR
                source=t.get("source"),  # Preserve source (native/ocr)
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
                ocr_stats=r.get("ocr_stats"),
            )
        )

    pages.sort(key=lambda pg: pg.page_number)

    meta = DocMeta(doc_id=doc_id, path=str(p.resolve()), page_count=page_count)
    return VectorMap(meta=meta, pages=pages)

