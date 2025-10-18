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

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 in PDF user space

@dataclass(frozen=True)
class HighResOCRConfig:
    dpi: int = 500            # 500–600 for fine diagrams
    psm: int = 11             # sparse text, as lines (Tesseract only)
    min_conf: int = 60        # filter weak OCR results
    lang: str = "eng"
    engine: str = "tesseract" # OCR engine: "tesseract" or "easyocr"
    use_gpu: bool = True      # Use GPU if available (EasyOCR only)
    # The following are here for future page batching; for a single page they are inert
    max_workers: int = 1
    ram_budget_mb: int = 4096

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
    """
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
            "conf": conf_pct
        })

    return out


def _ocr_tile_tesseract(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """
    Run Tesseract on a grayscale tile. Returns list of dicts:
    {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)
    """
    # Enhanced preprocessing for engineering drawings
    # 1. Denoise with bilateral filter (preserves edges better than Gaussian)
    proc = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Adaptive thresholding for varying lighting/contrast
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 3. Morphological operations to clean up thin lines and connect broken characters
    kernel = np.ones((2,2), np.uint8)
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
        out.append({"text": txt, "bbox": (float(x), float(y), float(x+w), float(y+h)), "conf": conf})
    return out


def _ocr_tile(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """
    Run OCR on a grayscale tile using configured engine.
    Returns list of dicts: {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)
    """
    if cfg.engine == "easyocr":
        return _ocr_tile_easyocr(gray, cfg)
    else:
        return _ocr_tile_tesseract(gray, cfg)

def highres_ocr(
    pdf_path: str,
    page_index: int,
    cfg: HighResOCRConfig,
    tiles_pdf: Optional[List[BBox]] = None,
) -> List[Dict]:
    """
    OCR one page at high DPI.
    If tiles_pdf is provided, OCR only those regions (PDF coords).
    Returns list of {"text": str, "bbox": (x0,y0,x1,y1)} in PDF coords.
    """
    gray, zoom = _render_page_gray(pdf_path, page_index, cfg.dpi)

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
                results.append({"text": s["text"], "bbox": (X0, Y0, X1, Y1)})
        return results

    # Otherwise OCR the whole page
    spans = _ocr_tile(gray, cfg)
    for s in spans:
        x0, y0, x1, y1 = s["bbox"]
        results.append({"text": s["text"], "bbox": (x0/zoom, y0/zoom, x1/zoom, y1/zoom)})
    return results


# ========================
# Tiled OCR Implementation
# ========================

def calculate_tile_grid(
    page_width: float,
    page_height: float,
    dpi: int,
    max_tile_pixels: int = 29000,
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

    # Check if tile has content
    if skip_empty and not detect_tile_content(tile_img, white_threshold, min_content_pct):
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
    use_gpu: bool = True        # Use GPU if available (EasyOCR only)
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

    Returns:
        List of OCR results as dicts: {"text": str, "bbox": tuple, "conf": int}
        Optionally returns (results, report) if return_report=True
    """
    start_time = time.time()

    # Get page dimensions
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    page_width = page.rect.width
    page_height = page.rect.height
    doc.close()

    # Calculate tile grid
    config = calculate_tile_grid(page_width, page_height, dpi, overlap_pct=overlap_pct)

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
    tiles_skipped = 0

    if config.rows > 1 or config.cols > 1:
        print(f"OCR: Tiling page into {config.rows}x{config.cols} grid with {overlap_pct*100:.0f}% overlap", file=sys.stderr)
        # Log actual tile sizes for debugging
        if tiles:
            sample_tile = tiles[0]
            tile_width = sample_tile.px1 - sample_tile.px0
            tile_height = sample_tile.py1 - sample_tile.py0
            print(f"OCR: First tile size: {tile_width}x{tile_height} pixels", file=sys.stderr)

    # Process each tile with all PSM modes (Tesseract) or single pass (EasyOCR)
    for tile in tiles:
        tile_all_results = []

        if engine == "easyocr":
            # EasyOCR doesn't have PSM modes, just run once
            ocr_config = HighResOCRConfig(
                dpi=dpi,
                psm=psm,
                min_conf=min_conf,
                lang=lang,
                engine=engine,
                use_gpu=use_gpu
            )

            tile_results = process_single_tile(
                pdf_path=pdf_path,
                page_index=page_index,
                tile_bounds=tile,
                zoom=zoom,
                ocr_config=ocr_config,
                skip_empty=skip_empty,
                min_content_pct=min_content_pct,
                white_threshold=config.white_threshold
            )

            tile_all_results.extend(tile_results)
        else:
            # Tesseract with optional dual-PSM mode
            for psm_mode in psm_modes:
                ocr_config = HighResOCRConfig(
                    dpi=dpi,
                    psm=psm_mode,
                    min_conf=min_conf,
                    lang=lang,
                    engine=engine,
                    use_gpu=use_gpu
                )

                tile_results = process_single_tile(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    tile_bounds=tile,
                    zoom=zoom,
                    ocr_config=ocr_config,
                    skip_empty=skip_empty,
                    min_content_pct=min_content_pct,
                    white_threshold=config.white_threshold
                )

                tile_all_results.extend(tile_results)

        if not tile_all_results and skip_empty:
            tiles_skipped += 1
        else:
            all_results.extend(tile_all_results)

    # Deduplicate results from overlapping tiles
    initial_count = len(all_results)
    deduplicated_results = merge_and_deduplicate(all_results)
    duplicates_removed = initial_count - len(deduplicated_results)

    processing_time = time.time() - start_time

    # Create report
    report = TileOCRReport(
        total_tiles=len(tiles),
        tiles_processed=len(tiles) - tiles_skipped,
        tiles_skipped_empty=tiles_skipped,
        total_results=len(deduplicated_results),
        duplicates_removed=duplicates_removed,
        processing_time=processing_time,
        grid_size=(config.rows, config.cols),
        dpi_used=dpi,
        overlap_pct=overlap_pct
    )

    if return_report:
        return deduplicated_results, report
    else:
        return deduplicated_results
