from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import fitz  # PyMuPDF
import cv2
import os
import pytesseract
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import debug visualizer
try:
    from ..debug.ocr_visualizer import OCRVisualizer, create_visualizer
    HAVE_VISUALIZER = True
except ImportError:
    HAVE_VISUALIZER = False
    OCRVisualizer = None
    create_visualizer = None

if os.name == "nt" and "tesseract.exe" not in pytesseract.pytesseract.tesseract_cmd.lower():
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default):
        pytesseract.pytesseract.tesseract_cmd = default

# Try to import rapidfuzz for text similarity
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False

# Try to import EasyOCR for GPU-accelerated OCR
try:
    import easyocr
    HAVE_EASYOCR = True
    # Initialize EasyOCR reader (lazy loading, only when needed)
    _EASYOCR_READER = None
except ImportError:
    HAVE_EASYOCR = False
    _EASYOCR_READER = None

# Try to import Qwen2-VL client for high-accuracy VLM-based OCR via microservice
try:
    from .qwen_vl_ocr_client import get_qwen_vl_ocr_client, VLLMServiceUnavailable
    HAVE_QWEN_VL = True
    _QWEN_VL_OCR = None
except ImportError:
    HAVE_QWEN_VL = False
    _QWEN_VL_OCR = None
    VLLMServiceUnavailable = Exception  # Fallback

from .qwen_prompts import DEFAULT_PROMPT_MODE as DEFAULT_QWEN_PROMPT_MODE

# Qwen2-VL tiling thresholds
QWEN_VL_MAX_SIDE = 1024  # Vision model accepts up to 4K per side
QWEN_VL_SUBTILE_OVERLAP_RATIO = 0.12  # 8% overlap to keep context on seams

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 in PDF user space

@dataclass(frozen=True)
class HighResOCRConfig:
    dpi: int = 500            # 500–600 for fine diagrams
    psm: int = 11             # sparse text, as lines (Tesseract only)
    min_conf: int = 50        # filter weak OCR results
    lang: str = "eng"
    engine: str = "tesseract" # OCR engine: "tesseract", "easyocr", or "qwen-vl"
    use_gpu: bool = True      # Use GPU if available (EasyOCR, Qwen-VL)
    # The following are here for future page batching; for a single page they are inert
    max_workers: int = 1
    ram_budget_mb: int = 4096
    # Preprocessing enhancements for small text
    upscale_factor: float = 1.0      # Upscaling multiplier (1.0 = no upscaling, 1.5-2.0 recommended for small text)
    enable_sharpening: bool = False  # Apply unsharp mask after upscaling
    qwen_prompt_mode: str = DEFAULT_QWEN_PROMPT_MODE  # Prompt mode for Qwen2-VL (layout|sparse|table)

# ========================
# Tiled OCR Data Structures
# ========================

@dataclass
class TileConfig:
    """Configuration for tile-based OCR."""
    rows: int                    # Number of tile rows
    cols: int                    # Number of tile columns
    overlap_pct: float           # Overlap percentage (0.0-1.0)
    skip_empty: bool             # Skip tiles with < min_content_pct
    min_content_pct: float       # Minimum content to process tile (0.01 = 1%)
    white_threshold: int = 250   # Pixel value considered "white"

@dataclass
class TileBounds:
    """Bounds for a single tile in both pixel and PDF coordinates."""
    row: int                     # Tile row index
    col: int                     # Tile column index

    # Pixel coordinates (in full-page rendered image space)
    px0: int
    py0: int
    px1: int
    py1: int

    # PDF coordinates (for final results)
    pdf_x0: float
    pdf_y0: float
    pdf_x1: float
    pdf_y1: float

    # Metadata
    has_content: bool = True     # False if tile is mostly blank
    tile_id: str = ""            # "row_col" identifier

@dataclass
class OCRResult:
    """Single OCR text result with metadata."""
    text: str
    bbox: Tuple[float, float, float, float]  # PDF coordinates
    confidence: int
    tile_id: str = ""            # Which tile produced this (for debugging)
    source: str = "ocr"

@dataclass
class TileOCRReport:
    """Diagnostic report for tiled OCR operation."""
    total_tiles: int
    tiles_processed: int
    tiles_skipped_empty: int
    total_results: int
    duplicates_removed: int
    processing_time: float
    grid_size: Tuple[int, int]
    dpi_used: int
    overlap_pct: float
    avg_confidence: float = 0.0
    median_confidence: float = 0.0
    min_confidence: int = 0
    max_confidence: int = 0
    low_confidence_results: int = 0
    engine_counts: Dict[str, int] = field(default_factory=dict)

def _render_page_gray(pdf_path: str, page_index: int, dpi: int) -> tuple[np.ndarray, float]:
    """Return grayscale image and zoom factor for a page at given DPI."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB if pix.n==3
    # Convert to numpy
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    if pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return gray, zoom

def _get_easyocr_reader(lang: str = "en", use_gpu: bool = True):
    """Get or initialize EasyOCR reader (singleton pattern for efficiency)."""
    global _EASYOCR_READER

    if not HAVE_EASYOCR:
        raise ImportError("EasyOCR not installed. Install with: pip install easyocr")

    # Convert Tesseract language codes to EasyOCR format
    lang_map = {
        "eng": "en",
        "spa": "es",
        "fra": "fr",
        "deu": "de",
        "ita": "it",
        "por": "pt",
        "rus": "ru",
        "jpn": "ja",
        "chi_sim": "ch_sim",
        "chi_tra": "ch_tra",
        "kor": "ko",
        "ara": "ar"
    }
    easyocr_lang = lang_map.get(lang, lang)

    # Initialize reader if not already done
    if _EASYOCR_READER is None:
        import sys
        print(f"OCR: Initializing EasyOCR reader (lang={easyocr_lang}, GPU={'enabled' if use_gpu else 'disabled'})...", file=sys.stderr)
        _EASYOCR_READER = easyocr.Reader([easyocr_lang], gpu=use_gpu)
        print(f"OCR: EasyOCR reader initialized", file=sys.stderr)

    return _EASYOCR_READER


def _ocr_tile_easyocr(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """
    Run EasyOCR on a grayscale tile. Returns list of dicts:
    {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)

    If tile exceeds EasyOCR's size limits, recursively splits into sub-tiles.
    """
    height, width = gray.shape
    EASYOCR_MAX_DIM = 32000  # OpenCV limit is 32767, use safety margin

    # Check if tile exceeds EasyOCR's size limits
    if width > EASYOCR_MAX_DIM or height > EASYOCR_MAX_DIM:
        import sys
        print(f"[EasyOCR] Tile {width}x{height}px exceeds {EASYOCR_MAX_DIM}px limit - splitting into sub-tiles", file=sys.stderr)

        # Split into 2x2 grid (4 sub-tiles)
        mid_x = width // 2
        mid_y = height // 2

        sub_tiles = [
            (gray[0:mid_y, 0:mid_x], 0, 0),           # Top-left
            (gray[0:mid_y, mid_x:width], mid_x, 0),   # Top-right
            (gray[mid_y:height, 0:mid_x], 0, mid_y),  # Bottom-left
            (gray[mid_y:height, mid_x:width], mid_x, mid_y)  # Bottom-right
        ]

        # Recursively process each sub-tile
        all_results = []
        for sub_tile, offset_x, offset_y in sub_tiles:
            sub_results = _ocr_tile_easyocr(sub_tile, cfg)  # Recursive call

            # Offset bboxes to original tile coordinates
            for result in sub_results:
                x0, y0, x1, y1 = result["bbox"]
                result["bbox"] = (x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y)
                all_results.append(result)

        return all_results

    # Normal processing for tiles under size limit
    reader = _get_easyocr_reader(lang=cfg.lang, use_gpu=cfg.use_gpu)

    # EasyOCR works best with minimal preprocessing
    # Just apply light denoising to reduce noise
    proc = cv2.bilateralFilter(gray, 5, 50, 50)

    # EasyOCR returns: [[bbox, text, confidence], ...]
    # bbox is [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
    results = reader.readtext(proc, detail=1, paragraph=False)

    out = []
    for bbox_points, text, confidence in results:
        text = text.strip()
        conf_pct = int(confidence * 100)

        if not text or conf_pct < cfg.min_conf:
            continue

        # Convert bbox points to x0,y0,x1,y1 format
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        out.append({
            "text": text,
            "bbox": (float(x0), float(y0), float(x1), float(y1)),
            "conf": conf_pct,
            "source": "ocr-easyocr"
        })

    return out


def _preprocess_tile_upscale(gray: np.ndarray, upscale_factor: float = 1.5, enable_sharpening: bool = True) -> Tuple[np.ndarray, float]:
    """
    Upscale and enhance a tile for better OCR accuracy on small text.

    Args:
        gray: Grayscale input image
        upscale_factor: Upscaling multiplier (1.5-2.0 recommended)
        enable_sharpening: Apply unsharp mask after upscaling

    Returns:
        (upscaled_image, actual_scale_factor)
    """
    if upscale_factor <= 1.0:
        return gray, 1.0

    # Calculate new dimensions
    new_width = int(gray.shape[1] * upscale_factor)
    new_height = int(gray.shape[0] * upscale_factor)

    # Upscale using Lanczos interpolation (best for text)
    upscaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    upscaled = clahe.apply(upscaled)

    # Optional: Sharpen using unsharp mask
    if enable_sharpening:
        # Gaussian blur
        blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
        # Unsharp mask: original + (original - blurred) * strength
        upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        # Clip to valid range
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

    return upscaled, upscale_factor


def _ocr_tile_tesseract(gray: np.ndarray, cfg: HighResOCRConfig, upscale_factor: float = 1.0, enable_sharpening: bool = False) -> List[Dict]:
    """
    Run Tesseract on a grayscale tile. Returns list of dicts:
    {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)
    """
    # Optional upscaling for small text
    if upscale_factor > 1.0:
        gray, actual_scale = _preprocess_tile_upscale(gray, upscale_factor, enable_sharpening)
    else:
        actual_scale = 1.0

    # Enhanced preprocessing for engineering drawings (shaded tables, dense grids)
    # 1. Denoise with bilateral filter (preserves edges better than Gaussian)
    base = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Remove soft background shading that often appears in table columns
    background = cv2.medianBlur(base, 21)
    shade_removed = cv2.absdiff(base, background)
    shade_removed = cv2.normalize(shade_removed, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Adaptive thresholding with dynamic window size (larger windows for wide cells)
    min_dim = min(shade_removed.shape[:2])
    if min_dim > 600:
        block_size = 41
    elif min_dim > 400:
        block_size = 31
    elif min_dim > 200:
        block_size = 21
    else:
        block_size = 11
    if block_size % 2 == 0:
        block_size += 1  # Block size must be odd

    proc = cv2.adaptiveThreshold(
        shade_removed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        2,
    )

    # 4. Remove vertical ruling lines that confuse OCR confidences
    vertical_kernel_len = max(3, int(proc.shape[0] * 0.05))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    vertical_lines = cv2.morphologyEx(proc, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    proc = cv2.subtract(proc, vertical_lines)

    # 5. Morphological close to reconnect characters broken by previous steps
    kernel = np.ones((2, 2), np.uint8)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

    # Use image_to_data to get boxes + confidences
    # Add OEM 3 (Default, LSTM) for better accuracy
    ts_cfg = f"-l {cfg.lang} --psm {cfg.psm} --oem 3"
    data = pytesseract.image_to_data(proc, config=ts_cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = int(data["conf"][i])
        except Exception:
            conf = -1
        if not txt or conf < cfg.min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        # Scale bounding boxes back to original size if upscaled
        if actual_scale > 1.0:
            x, y, w, h = x / actual_scale, y / actual_scale, w / actual_scale, h / actual_scale

        out.append({
            "text": txt,
            "bbox": (float(x), float(y), float(x+w), float(y+h)),
            "conf": conf,
            "source": "ocr-tesseract"
        })
    return out


def _qwen_vl_client() -> Any:
    """Ensure the Qwen2-VL client is initialized and available."""
    global _QWEN_VL_OCR

    if not HAVE_QWEN_VL:
        raise ImportError("Qwen-VL client not available")

    if _QWEN_VL_OCR is None:
        import sys
        print("OCR: Connecting to vLLM service for Qwen2-VL OCR...", file=sys.stderr)
        _QWEN_VL_OCR = get_qwen_vl_ocr_client()

        if not _QWEN_VL_OCR.is_available():
            print("OCR: WARNING - vLLM service not available, will fallback to EasyOCR/Tesseract", file=sys.stderr)
            raise VLLMServiceUnavailable("vLLM service not available")

        print("OCR: Connected to vLLM service", file=sys.stderr)

    return _QWEN_VL_OCR


def _ocr_tile_qwen_vl_base(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """Call the Qwen2-VL service for a tile guaranteed to be within size limits."""
    client = _qwen_vl_client()

    try:
        results = client.extract_text_from_image(
            gray,
            focus_technical=True,
            prompt_mode=cfg.qwen_prompt_mode,
        )
    except VLLMServiceUnavailable:
        import sys
        print("OCR: vLLM service unavailable, fallback required", file=sys.stderr)
        raise

    out: List[Dict[str, Any]] = []
    for item in results:
        conf = int(item.get("confidence", 1.0) * 100)
        if conf < cfg.min_conf:
            continue

        out.append({
            "text": item["text"],
            "bbox": item["bbox"],
            "conf": conf,
            "source": "ocr-qwen-vl"
        })

    return out


def _qwen_generate_slices(length: int) -> List[Tuple[int, int]]:
    """Create overlapping slices that keep segment length within the Qwen limit."""
    if length <= QWEN_VL_MAX_SIDE:
        return [(0, length)]

    overlap = max(32, int(QWEN_VL_MAX_SIDE * QWEN_VL_SUBTILE_OVERLAP_RATIO))
    step = max(1, QWEN_VL_MAX_SIDE - overlap)

    slices: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + QWEN_VL_MAX_SIDE)
        slices.append((start, end))
        if end >= length:
            break
        start = end - overlap

    return slices


def _ocr_tile_qwen_vl(gray: np.ndarray, cfg: HighResOCRConfig, *, _depth: int = 0) -> List[Dict]:
    """
    Run Qwen2-VL on a grayscale tile via vLLM microservice.
    Automatically sub-tiles oversized inputs so we maintain high DPI without exceeding
    the 4K vision-model limit.
    """
    height, width = gray.shape[:2]
    max_dim = max(height, width)

    if max_dim <= QWEN_VL_MAX_SIDE:
        return _ocr_tile_qwen_vl_base(gray, cfg)

    # Subdivide oversized tiles with overlap to preserve context
    y_slices = _qwen_generate_slices(height)
    x_slices = _qwen_generate_slices(width)
    aggregated: List[Dict[str, Any]] = []

    for y0, y1 in y_slices:
        for x0, x1 in x_slices:
            sub_gray = gray[y0:y1, x0:x1]
            sub_results = _ocr_tile_qwen_vl(sub_gray, cfg, _depth=_depth + 1)
            for item in sub_results:
                x_start, y_start, x_end, y_end = item["bbox"]
                aggregated.append({
                    **item,
                    "bbox": (
                        x_start + x0,
                        y_start + y0,
                        x_end + x0,
                        y_end + y0,
                    ),
                })

    # Deduplicate overlaps introduced by sub-tiling
    return merge_and_deduplicate(aggregated)


def _ocr_tile(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """
    Run OCR on a grayscale tile using configured engine.
    Returns list of dicts: {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)
    """
    if cfg.engine == "qwen-vl":
        return _ocr_tile_qwen_vl(gray, cfg)
    elif cfg.engine == "easyocr":
        return _ocr_tile_easyocr(gray, cfg)
    else:
        return _ocr_tile_tesseract(gray, cfg, cfg.upscale_factor, cfg.enable_sharpening)

def highres_ocr(
    pdf_path: str,
    page_index: int,
    cfg: HighResOCRConfig,
    tiles_pdf: Optional[List[BBox]] = None,
    debug_visualizer: Optional['OCRVisualizer'] = None,
) -> List[Dict]:
    """
    OCR one page at high DPI with optional visual debugging.

    Args:
        pdf_path: Path to PDF file
        page_index: Page index (0-based)
        cfg: OCR configuration
        tiles_pdf: Optional list of regions to OCR (in PDF coordinates)
        debug_visualizer: Optional OCRVisualizer for debug output

    Returns:
        List of {"text": str, "bbox": (x0,y0,x1,y1), "conf": int} in PDF coords
    """
    start_time = time.time()

    # Initialize debug visualizer if not provided
    if debug_visualizer is None and HAVE_VISUALIZER:
        debug_visualizer = create_visualizer(enabled=False)

    if debug_visualizer:
        debug_visualizer.reset(page_index + 1)

    # Render page
    gray, zoom = _render_page_gray(pdf_path, page_index, cfg.dpi)

    # STAGE 1: Save original
    if debug_visualizer:
        debug_visualizer.save_original(gray, cfg.dpi)

    # STAGE 2: Grayscale (already done, save it)
    if debug_visualizer:
        debug_visualizer.save_grayscale(gray)

    # Preprocessing info for debug
    preprocessing_info = {
        "engine": cfg.engine,
        "psm": cfg.psm,
        "min_conf": cfg.min_conf,
        "dpi": cfg.dpi
    }

    # STAGE 3: Preprocessing (for Tesseract, show preprocessing steps)
    if cfg.engine == "tesseract" and debug_visualizer:
        base = cv2.bilateralFilter(gray, 9, 75, 75)
        background = cv2.medianBlur(base, 21)
        shade_removed = cv2.absdiff(base, background)
        shade_removed = cv2.normalize(shade_removed, None, 0, 255, cv2.NORM_MINMAX)

        min_dim = min(shade_removed.shape[:2])
        if min_dim > 600:
            block_size = 41
        elif min_dim > 400:
            block_size = 31
        elif min_dim > 200:
            block_size = 21
        else:
            block_size = 11
        if block_size % 2 == 0:
            block_size += 1

        proc = cv2.adaptiveThreshold(
            shade_removed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2,
        )

        vertical_kernel_len = max(3, int(proc.shape[0] * 0.05))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
        vertical_lines = cv2.morphologyEx(proc, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        proc = cv2.subtract(proc, vertical_lines)

        kernel = np.ones((2, 2), np.uint8)
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

        preprocessing_info.update({
            "bilateral_filter": "9x9, sigma=75",
            "background_removal": "median 21x21",
            "adaptive_threshold": f"Gaussian, block={block_size}",
            "vertical_line_removal": f"open kernel=(1,{vertical_kernel_len})",
            "morphology": "Close, 2x2 kernel"
        })

        debug_visualizer.save_preprocessed(proc, preprocessing_info)

    h, w = gray.shape[:2]
    results: List[Dict] = []

    if tiles_pdf:
        # OCR only specified PDF-space tiles
        for (x0, y0, x1, y1) in tiles_pdf:
            px0, py0 = int(x0 * zoom), int(y0 * zoom)
            px1, py1 = int(x1 * zoom), int(y1 * zoom)
            # clamp
            px0 = max(0, min(w, px0)); px1 = max(0, min(w, px1))
            py0 = max(0, min(h, py0)); py1 = max(0, min(h, py1))
            if px1 <= px0 or py1 <= py0:
                continue
            tile = gray[py0:py1, px0:px1]
            spans = _ocr_tile(tile, cfg)
            for s in spans:
                tx0, ty0, tx1, ty1 = s["bbox"]
                # map tile-pixel bbox back to PDF coords
                X0, Y0 = (px0 + tx0) / zoom, (py0 + ty0) / zoom
                X1, Y1 = (px0 + tx1) / zoom, (py0 + ty1) / zoom
                results.append({
                    "text": s["text"],
                    "bbox": (X0, Y0, X1, Y1),
                    "conf": s.get("conf", 0),
                    "source": s.get("source", "ocr")
                })
    else:
        # Otherwise OCR the whole page
        spans = _ocr_tile(gray, cfg)
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            results.append({
                "text": s["text"],
                "bbox": (x0/zoom, y0/zoom, x1/zoom, y1/zoom),
                "conf": s.get("conf", 0),
                "source": s.get("source", "ocr")
            })

    if debug_visualizer:
        processing_time = time.time() - start_time
        _emit_debug_visuals(
            debug_visualizer,
            gray,
            results,
            zoom,
            include_initial=False,
            processing_time=processing_time,
        )
    return results


# ========================
# Tiled OCR Implementation
# ========================

def calculate_tile_grid(
    page_width: float,
    page_height: float,
    dpi: int,
    max_tile_pixels: int = 29000,  # Default for Tesseract/EasyOCR (override to 4096 for Qwen-VL)
    overlap_pct: float = 0.20
) -> TileConfig:
    """
    Calculate optimal tile grid for a page.

    Args:
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Target rendering DPI
        max_tile_pixels: Maximum pixels per tile dimension
        overlap_pct: Overlap percentage between tiles

    Returns:
        TileConfig with grid dimensions and settings
    """
    # Calculate full page pixel dimensions at target DPI
    zoom = dpi / 72.0
    pixel_width = int(page_width * zoom)
    pixel_height = int(page_height * zoom)

    # If page fits in one tile, no tiling needed
    if pixel_width <= max_tile_pixels and pixel_height <= max_tile_pixels:
        return TileConfig(
            rows=1,
            cols=1,
            overlap_pct=0.0,  # No overlap needed for single tile
            skip_empty=False,
            min_content_pct=0.0
        )

    # Calculate grid size needed, accounting for overlap
    # Each tile will have overlap on both sides, so the effective max size is reduced
    # Effective tile size = max_tile_pixels / (1 + overlap_pct)
    effective_max = max_tile_pixels / (1 + overlap_pct)

    cols = math.ceil(pixel_width / effective_max)
    rows = math.ceil(pixel_height / effective_max)

    return TileConfig(
        rows=rows,
        cols=cols,
        overlap_pct=overlap_pct,
        skip_empty=True,
        min_content_pct=0.01  # 1% content threshold
    )


def generate_tile_bounds(
    config: TileConfig,
    page_width: float,
    page_height: float,
    dpi: int
) -> List[TileBounds]:
    """
    Generate bounding boxes for all tiles with overlap.

    Args:
        config: TileConfig from calculate_tile_grid()
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        dpi: Rendering DPI

    Returns:
        List of TileBounds, one per tile, ordered left-to-right, top-to-bottom
    """
    zoom = dpi / 72.0
    pixel_width = int(page_width * zoom)
    pixel_height = int(page_height * zoom)

    tiles: List[TileBounds] = []

    # Calculate base tile size (without overlap)
    base_tile_width = pixel_width / config.cols
    base_tile_height = pixel_height / config.rows

    # Calculate overlap in pixels
    overlap_w = int(base_tile_width * config.overlap_pct)
    overlap_h = int(base_tile_height * config.overlap_pct)

    for row in range(config.rows):
        for col in range(config.cols):
            # Base tile boundaries (no overlap)
            base_px0 = int(col * base_tile_width)
            base_py0 = int(row * base_tile_height)
            base_px1 = pixel_width if col == config.cols - 1 else int((col + 1) * base_tile_width)
            base_py1 = pixel_height if row == config.rows - 1 else int((row + 1) * base_tile_height)

            # Add overlap
            px0 = max(0, base_px0 - (overlap_w if col > 0 else 0))
            py0 = max(0, base_py0 - (overlap_h if row > 0 else 0))
            px1 = min(pixel_width, base_px1 + (overlap_w if col < config.cols - 1 else 0))
            py1 = min(pixel_height, base_py1 + (overlap_h if row < config.rows - 1 else 0))

            # Convert to PDF coordinates
            pdf_x0 = px0 / zoom
            pdf_y0 = py0 / zoom
            pdf_x1 = px1 / zoom
            pdf_y1 = py1 / zoom

            tile = TileBounds(
                row=row,
                col=col,
                px0=px0,
                py0=py0,
                px1=px1,
                py1=py1,
                pdf_x0=pdf_x0,
                pdf_y0=pdf_y0,
                pdf_x1=pdf_x1,
                pdf_y1=pdf_y1,
                has_content=True,  # Will be determined later
                tile_id=f"{row}_{col}"
            )
            tiles.append(tile)

    return tiles


def render_tile(
    pdf_path: str,
    page_index: int,
    tile_bounds: TileBounds,
    zoom: float
) -> np.ndarray:
    """
    Render a specific tile region from PDF page.

    Args:
        pdf_path: Path to PDF
        page_index: Page index (0-based)
        tile_bounds: TileBounds defining region to render
        zoom: Zoom factor for rendering

    Returns:
        Grayscale numpy array of tile region
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # Create clip rectangle in PDF coordinates
    clip_rect = fitz.Rect(
        tile_bounds.pdf_x0,
        tile_bounds.pdf_y0,
        tile_bounds.pdf_x1,
        tile_bounds.pdf_y1
    )

    # Render only the clipped region
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

    # Convert to numpy
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()

    # Convert to grayscale if needed
    if pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    return gray


def detect_tile_content(
    tile_image: np.ndarray,
    white_threshold: int = 250,
    min_content_pct: float = 0.01
) -> bool:
    """
    Detect if tile has sufficient content to warrant OCR.

    Args:
        tile_image: Grayscale tile image
        white_threshold: Pixel value considered white/blank
        min_content_pct: Minimum content ratio (0.01 = 1%)

    Returns:
        True if tile has content, False if mostly blank
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(tile_image, (5, 5), 0)

    # Threshold: pixels below white_threshold are content
    _, thresh = cv2.threshold(blurred, white_threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological close to connect nearby content
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Calculate content ratio
    content_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    content_ratio = content_pixels / total_pixels

    return content_ratio >= min_content_pct


def process_single_tile(
    pdf_path: str,
    page_index: int,
    tile_bounds: TileBounds,
    zoom: float,
    ocr_config: HighResOCRConfig,
    skip_empty: bool,
    min_content_pct: float,
    white_threshold: int
) -> List[Dict[str, Any]]:
    """
    Process a single tile: render, check content, OCR, map coordinates.

    Args:
        pdf_path: Path to PDF
        page_index: Page index
        tile_bounds: Tile region to process
        zoom: Zoom factor
        ocr_config: OCR configuration (DPI, PSM, etc.)
        skip_empty: Skip if tile is blank
        min_content_pct: Content threshold
        white_threshold: White pixel threshold

    Returns:
        List of OCRResult-like dicts with coordinates in PDF space
    """
    # Render tile
    tile_img = render_tile(pdf_path, page_index, tile_bounds, zoom)

    # Track whether the tile contains content when skip_empty=True
    tile_bounds.has_content = True
    if skip_empty:
        has_content = detect_tile_content(tile_img, white_threshold, min_content_pct)
        tile_bounds.has_content = has_content
        if not has_content:
            return []

    # Run OCR on tile
    tile_results = _ocr_tile(tile_img, ocr_config)

    # Map tile-local coordinates to PDF coordinates
    results = []
    for item in tile_results:
        # item["bbox"] is in tile-local pixel coordinates
        tx0, ty0, tx1, ty1 = item["bbox"]

        # Map to PDF coordinates
        # Tile starts at (tile_bounds.pdf_x0, tile_bounds.pdf_y0)
        # Tile pixel coords are relative to tile's top-left corner
        pdf_x0 = tile_bounds.pdf_x0 + (tx0 / zoom)
        pdf_y0 = tile_bounds.pdf_y0 + (ty0 / zoom)
        pdf_x1 = tile_bounds.pdf_x0 + (tx1 / zoom)
        pdf_y1 = tile_bounds.pdf_y0 + (ty1 / zoom)

        results.append({
            "text": item["text"],
            "bbox": (pdf_x0, pdf_y0, pdf_x1, pdf_y1),
            "conf": item.get("conf", 0),
            "tile_id": tile_bounds.tile_id
        })

    return results


def merge_and_deduplicate(
    results: List[Dict[str, Any]],
    overlap_iou_threshold: float = 0.5,
    text_similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Remove duplicate OCR results from tile overlaps.

    Args:
        results: List of OCR result dicts from all tiles
        overlap_iou_threshold: IOU threshold for bbox overlap (0.5 = 50%)
        text_similarity_threshold: Text similarity threshold (0.8 = 80%)

    Returns:
        Deduplicated list of results, sorted by reading order (top-to-bottom, left-to-right)
    """
    if len(results) <= 1:
        return results

    # Sort by reading order (y0, then x0)
    sorted_results = sorted(results, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Track which results to keep
    keep_mask = [True] * len(sorted_results)

    for i in range(len(sorted_results)):
        if not keep_mask[i]:
            continue

        for j in range(i + 1, len(sorted_results)):
            if not keep_mask[j]:
                continue

            # Calculate bbox IOU
            box1 = sorted_results[i]["bbox"]
            box2 = sorted_results[j]["bbox"]

            # Calculate intersection
            x0_i = max(box1[0], box2[0])
            y0_i = max(box1[1], box2[1])
            x1_i = min(box1[2], box2[2])
            y1_i = min(box1[3], box2[3])

            if x1_i <= x0_i or y1_i <= y0_i:
                # No intersection
                continue

            intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - intersection_area

            iou = intersection_area / union_area if union_area > 0 else 0

            # Check text similarity
            text1 = sorted_results[i]["text"]
            text2 = sorted_results[j]["text"]

            if HAVE_RAPIDFUZZ:
                text_sim = fuzz.ratio(text1, text2) / 100.0
            else:
                # Fallback: exact match only
                text_sim = 1.0 if text1 == text2 else 0.0

            # If similar enough, keep the one with higher confidence
            if iou >= overlap_iou_threshold and text_sim >= text_similarity_threshold:
                conf1 = sorted_results[i].get("conf", 0)
                conf2 = sorted_results[j].get("conf", 0)

                if conf2 > conf1:
                    keep_mask[i] = False
                    break  # i is removed, move to next i
                else:
                    keep_mask[j] = False

    # Return kept results
    return [r for i, r in enumerate(sorted_results) if keep_mask[i]]


def _emit_debug_visuals(
    visualizer: Optional['OCRVisualizer'],
    base_image: np.ndarray,
    results_pdf: List[Dict[str, Any]],
    zoom: float,
    *,
    include_initial: bool = False,
    initial_dpi: Optional[int] = None,
    processing_time: Optional[float] = None,
) -> None:
    """
    Emit standard debug visualizations for OCR stages.
    """
    if not visualizer or not visualizer.config.enabled:
        return

    results_px = [
        {
            "text": res.get("text", ""),
            "bbox": [int(round(v * zoom)) for v in res.get("bbox", (0, 0, 0, 0))],
            "conf": res.get("conf", 0),
        }
        for res in results_pdf
    ]

    if include_initial:
        if initial_dpi is not None:
            visualizer.save_original(base_image, initial_dpi)
        visualizer.save_grayscale(base_image)

    visualizer.save_detections(base_image, results_px)
    visualizer.save_final_with_text(base_image, results_px)
    visualizer.save_confidence_heatmap(
        base_image.shape[1],
        base_image.shape[0],
        results_px
    )

    if processing_time is not None:
        visualizer.generate_summary_report(results_pdf, processing_time)


def tiled_ocr(
    pdf_path: str,
    page_index: int,
    dpi: int,
    psm: int = 11,
    min_conf: int = 60,
    lang: str = "eng",
    overlap_pct: float = 0.20,
    skip_empty: bool = True,
    min_content_pct: float = 0.01,
    max_workers: int = 1,
    return_report: bool = False,
    use_dual_psm: bool = True,  # Try both PSM 11 and PSM 6 (Tesseract only)
    engine: str = "tesseract",  # OCR engine: "tesseract" or "easyocr"
    use_gpu: bool = True,       # Use GPU if available (EasyOCR only)
    debug_visualizer: Optional['OCRVisualizer'] = None,  # Optional debug visualizer
    enable_layout_detection: bool = False,  # Enable layout-aware OCR
    table_dpi_boost: float = 1.5,  # DPI multiplier for table regions (1.5 = 1.5x higher DPI)
    enable_upscaling: bool = False,  # Enable upscaling preprocessing for small text
    upscale_factor: float = 1.5,  # Upscaling factor (1.5-2.0 recommended)
    qwen_prompt_mode: str = DEFAULT_QWEN_PROMPT_MODE,  # Prompt preset when engine == "qwen-vl"
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], TileOCRReport]:
    """
    Perform tiled OCR on a large PDF page.

    Args:
        pdf_path: Path to PDF file
        page_index: Page index (0-based)
        dpi: Target DPI for rendering
        psm: Tesseract page segmentation mode (11 = sparse text, 6 = uniform block) - Tesseract only
        min_conf: Minimum OCR confidence threshold (0-100)
        lang: OCR language ("eng" for Tesseract, "en" for EasyOCR)
        overlap_pct: Overlap between tiles (0.15-0.40 recommended)
        skip_empty: Skip tiles with minimal content
        min_content_pct: Minimum content ratio to process tile
        max_workers: Number of parallel workers (1 = serial)
        return_report: Return diagnostic report with results
        use_dual_psm: If True, runs both PSM 11 and PSM 6, merges results (Tesseract only)
        engine: OCR engine to use ("tesseract" or "easyocr")
        use_gpu: Enable GPU acceleration (EasyOCR only, requires CUDA)
        enable_layout_detection: Use layout-aware heuristics (tables/diagrams)
        table_dpi_boost: DPI multiplier applied to layout-detected tables
        enable_upscaling: Upscale tiles before OCR to enhance small text
        upscale_factor: Upscaling multiplier when enable_upscaling is True
        qwen_prompt_mode: Prompt preset when engine == "qwen-vl"

    Returns:
        List of OCR results as dicts: {"text": str, "bbox": tuple, "conf": int}
        Optionally returns (results, report) if return_report=True
    """
    start_time = time.time()

    # Initialize debug visualizer if not provided
    if debug_visualizer is None and HAVE_VISUALIZER:
        debug_visualizer = create_visualizer(enabled=False)

    if debug_visualizer:
        debug_visualizer.reset(page_index + 1)

    # Get page dimensions
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    page_width = page.rect.width
    page_height = page.rect.height
    doc.close()

    # Layout detection (if enabled)
    layout_regions = []
    layout_dpi = 400  # Default layout detection DPI
    if enable_layout_detection:
        import sys
        try:
            from .layout_detector import LayoutDetector

            # Adaptive LAYOUT_DPI based on page size
            # Large engineering drawings need higher DPI for accurate table grid detection
            page_size_inches = max(page_width / 72.0, page_height / 72.0)
            layout_dpi = 200 if page_size_inches > 50 else 150

            print(f"[Layout] Page size: {page_width/72.0:.1f}\" x {page_height/72.0:.1f}\" → Using {layout_dpi} DPI for layout detection", file=sys.stderr)
            layout_gray, layout_zoom = _render_page_gray(pdf_path, page_index, layout_dpi)

            # Detect regions
            detector_debug = bool(debug_visualizer and debug_visualizer.config.enabled)
            detector = LayoutDetector(debug=detector_debug)
            layout_regions = detector.analyze_page(layout_gray)

            print(f"[Layout] Detected {len(layout_regions)} regions:", file=sys.stderr)
            for i, region in enumerate(layout_regions):
                print(f"  {i+1}. {region.region_type} @ {region.bbox} (conf={region.confidence:.1f}%)", file=sys.stderr)

            # Save layout visualization to debug output
            if debug_visualizer and debug_visualizer.config.enabled:
                try:
                    debug_visualizer.save_layout_regions(layout_gray, layout_regions)
                except Exception as viz_error:
                    print(f"[Layout] Warning: Failed to save layout visualization: {viz_error}", file=sys.stderr)

        except ImportError as e:
            print(f"[Layout] Warning: layout_detector module not available ({e}). Proceeding with standard OCR.", file=sys.stderr)
            enable_layout_detection = False  # Disable if module unavailable
        except Exception as e:
            print(f"[Layout] Error during layout detection ({e}). Proceeding with standard OCR.", file=sys.stderr)
            enable_layout_detection = False  # Disable on any error
            layout_regions = []  # Clear any partial results

    # Engine-specific tile size limits
    # Qwen-VL (vision model): Small tiles (4K) to avoid decompression bomb errors
    # EasyOCR/Tesseract: Use Tesseract's limit (29K) - they benefit from high DPI
    if engine == "qwen-vl":
        max_tile_pixels = 4096
        import sys
        print(f"OCR: Using Qwen-VL with 4K tile limit (vision model)", file=sys.stderr)
    else:
        # EasyOCR and Tesseract both benefit from high DPI and can handle large tiles
        max_tile_pixels = 29000  # Tesseract's original limit - works well for both

    # Calculate tile grid
    config = calculate_tile_grid(page_width, page_height, dpi, max_tile_pixels=max_tile_pixels, overlap_pct=overlap_pct)

    # Generate tile bounds
    zoom = dpi / 72.0
    tiles = generate_tile_bounds(config, page_width, page_height, dpi)

    # Determine PSM modes to use (Tesseract only)
    import sys
    psm_modes = [psm]
    if engine == "tesseract" and use_dual_psm and psm == 11:
        psm_modes = [11, 6]  # Try sparse text (11) and uniform block (6)
        print(f"OCR: Using Tesseract with dual-PSM mode (PSM 11 + PSM 6)", file=sys.stderr)
    elif engine == "easyocr":
        print(f"OCR: Using EasyOCR (GPU={'enabled' if use_gpu else 'disabled'})", file=sys.stderr)
    else:
        print(f"OCR: Using Tesseract (PSM {psm})", file=sys.stderr)

    # Process tiles (serial for now, parallel support can be added later)
    all_results: List[Dict[str, Any]] = []

    if config.rows > 1 or config.cols > 1:
        print(f"OCR: Tiling page into {config.rows}x{config.cols} grid with {overlap_pct*100:.0f}% overlap", file=sys.stderr)
        # Log actual tile sizes for debugging
        if tiles:
            sample_tile = tiles[0]
            tile_width = sample_tile.px1 - sample_tile.px0
            tile_height = sample_tile.py1 - sample_tile.py0
            print(f"OCR: First tile size: {tile_width}x{tile_height} pixels", file=sys.stderr)

    # Helper function to check if tile overlaps with a table region
    def tile_overlaps_table(tile: TileBounds, regions: List, layout_zoom_factor: float) -> bool:
        """Check if tile overlaps with any table region (at layout DPI scale)."""
        if not regions:
            return False

        # Convert tile PDF coords to layout pixel coords
        tile_x0 = int(tile.pdf_x0 * layout_zoom_factor)
        tile_y0 = int(tile.pdf_y0 * layout_zoom_factor)
        tile_x1 = int(tile.pdf_x1 * layout_zoom_factor)
        tile_y1 = int(tile.pdf_y1 * layout_zoom_factor)

        for region in regions:
            if region.region_type != "table":
                continue

            rx0, ry0, rx1, ry1 = region.bbox

            # Check for overlap
            if not (tile_x1 < rx0 or tile_x0 > rx1 or tile_y1 < ry0 or tile_y0 > ry1):
                return True

        return False

    # Calculate layout zoom factor for coordinate conversion
    layout_zoom_factor = layout_dpi / 72.0

    # Process each tile with all PSM modes (Tesseract) or single pass (EasyOCR)
    for tile in tiles:
        tile_all_results: List[Dict[str, Any]] = []

        # Determine DPI and preprocessing for this tile based on layout
        tile_dpi = dpi
        tile_upscale = upscale_factor if enable_upscaling else 1.0

        if enable_layout_detection and tile_overlaps_table(tile, layout_regions, layout_zoom_factor):
            # Boost DPI for table regions - tables contain critical data
            tile_dpi = int(dpi * table_dpi_boost)
            import sys
            print(f"[Layout] Tile {tile.tile_id} overlaps table → DPI boosted to {tile_dpi} (base={dpi}, boost={table_dpi_boost}x)", file=sys.stderr)

        # Recalculate zoom if DPI changed
        tile_zoom = tile_dpi / 72.0

        if engine == "easyocr":
            # EasyOCR doesn't have PSM modes, just run once
            ocr_config = HighResOCRConfig(
                dpi=tile_dpi,
                psm=psm,
                min_conf=min_conf,
                lang=lang,
                engine=engine,
                use_gpu=use_gpu,
                upscale_factor=tile_upscale,
                enable_sharpening=enable_upscaling,
                qwen_prompt_mode=qwen_prompt_mode,
            )

            tile_results = process_single_tile(
                pdf_path=pdf_path,
                page_index=page_index,
                tile_bounds=tile,
                zoom=tile_zoom,  # Use boosted zoom if DPI changed
                ocr_config=ocr_config,
                skip_empty=skip_empty,
                min_content_pct=min_content_pct,
                white_threshold=config.white_threshold
            )

            # If tile was deemed empty, skip further processing
            if skip_empty and not getattr(tile, "has_content", True):
                continue

            tile_all_results.extend(tile_results)
        else:
            # Tesseract with optional dual-PSM mode
            for psm_mode in psm_modes:
                ocr_config = HighResOCRConfig(
                    dpi=tile_dpi,
                    psm=psm_mode,
                    min_conf=min_conf,
                    lang=lang,
                    engine=engine,
                    use_gpu=use_gpu,
                    upscale_factor=tile_upscale,
                    enable_sharpening=enable_upscaling,
                    qwen_prompt_mode=qwen_prompt_mode,
                )

                tile_results = process_single_tile(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    tile_bounds=tile,
                    zoom=tile_zoom,  # Use boosted zoom if DPI changed
                    ocr_config=ocr_config,
                    skip_empty=skip_empty,
                    min_content_pct=min_content_pct,
                    white_threshold=config.white_threshold
                )

                # Once a tile is classified as empty we can stop trying additional PSMs
                if skip_empty and not getattr(tile, "has_content", True):
                    tile_all_results = []
                    break

                tile_all_results.extend(tile_results)

        # Skip aggregation entirely for empty tiles when skip_empty=True
        if skip_empty and not getattr(tile, "has_content", True):
            continue

        all_results.extend(tile_all_results)

    tile_processed_flags = [
        (not skip_empty) or getattr(tile, "has_content", True)
        for tile in tiles
    ]
    tiles_skipped = tile_processed_flags.count(False)
    tiles_processed_count = len(tiles) - tiles_skipped

    # Deduplicate results from overlapping tiles
    initial_count = len(all_results)
    deduplicated_results = merge_and_deduplicate(all_results)
    duplicates_removed = initial_count - len(deduplicated_results)

    processing_time = time.time() - start_time

    # Confidence/engine statistics
    confidences = [
        int(r.get("conf", 0))
        for r in deduplicated_results
        if r.get("conf") is not None
    ]
    if confidences:
        avg_confidence = float(np.mean(confidences))
        median_confidence = float(np.median(confidences))
        min_confidence_val = int(np.min(confidences))
        max_confidence_val = int(np.max(confidences))
        low_confidence_results = sum(1 for c in confidences if c < min_conf)
    else:
        avg_confidence = 0.0
        median_confidence = 0.0
        min_confidence_val = 0
        max_confidence_val = 0
        low_confidence_results = 0

    engine_counts: Dict[str, int] = {}
    for r in deduplicated_results:
        engine_id = r.get("source") or "ocr"
        engine_counts[engine_id] = engine_counts.get(engine_id, 0) + 1

    # Create report
    report = TileOCRReport(
        total_tiles=len(tiles),
        tiles_processed=tiles_processed_count,
        tiles_skipped_empty=tiles_skipped,
        total_results=len(deduplicated_results),
        duplicates_removed=duplicates_removed,
        processing_time=processing_time,
        grid_size=(config.rows, config.cols),
        dpi_used=dpi,
        overlap_pct=overlap_pct,
        avg_confidence=avg_confidence,
        median_confidence=median_confidence,
        min_confidence=min_confidence_val,
        max_confidence=max_confidence_val,
        low_confidence_results=low_confidence_results,
        engine_counts=engine_counts,
    )

    # Generate debug output if visualizer is enabled
    if debug_visualizer and debug_visualizer.config.enabled:
        # Render full page at lower DPI for debug visualization (to keep file sizes reasonable)
        DEBUG_DPI = 150  # Much lower than OCR DPI, but sufficient for visualization
        gray_full, zoom_full = _render_page_gray(pdf_path, page_index, DEBUG_DPI)

        # Save initial stages
        debug_visualizer.save_original(gray_full, DEBUG_DPI)
        debug_visualizer.save_grayscale(gray_full)

        # Save tile grid visualization
        tile_viz_data = [
            {
                "px0": t.px0,
                "py0": t.py0,
                "px1": t.px1,
                "py1": t.py1,
                "tile_id": t.tile_id,
                "has_content": getattr(t, "has_content", True)
            }
            for t in tiles
        ]
        processed_mask = np.array(tile_processed_flags, dtype=bool)
        debug_visualizer.save_tiles(gray_full, tile_viz_data, processed_mask)

        _emit_debug_visuals(
            debug_visualizer,
            gray_full,
            deduplicated_results,
            zoom_full,
            include_initial=False,
            processing_time=processing_time,
        )

    if return_report:
        return deduplicated_results, report
    else:
        return deduplicated_results
