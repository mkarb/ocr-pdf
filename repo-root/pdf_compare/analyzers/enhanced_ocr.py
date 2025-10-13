"""
Enhanced OCR for engineering drawings with symbol recognition and caching.

Key features:
- Bilateral blur preprocessing for thin lettering
- Dual-pass OCR (psm 6 + 11) for rotated text
- Symbol library validation with fuzzy matching
- Graphical callout detection (leader lines, balloons)
- Page hash-based OCR caching
- Legend-aware validation
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from rapidfuzz import fuzz, process


# ========================
# Configuration
# ========================

@dataclass
class EnhancedOCRConfig:
    """Enhanced OCR configuration for engineering drawings."""

    # Basic settings
    dpi: int = 600  # Higher DPI for thin lettering
    min_conf: int = 60

    # Preprocessing
    use_bilateral_blur: bool = True
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50

    # Dual-pass OCR
    enable_dual_pass: bool = True
    psm_sparse: int = 11  # Sparse text (for rotated labels)
    psm_block: int = 6    # Uniform block (for text blocks)

    # Callout detection
    detect_callouts: bool = True
    callout_expansion_px: int = 10
    min_contour_area: int = 100

    # Symbol validation
    enable_symbol_validation: bool = True
    fuzzy_match_threshold: int = 85  # 0-100, higher = stricter

    # Caching
    enable_caching: bool = True
    cache_dir: Path = Path("./ocr_cache")

    # Performance
    max_workers: int = 6
    ram_budget_mb: int = 10240


# ========================
# Symbol Library
# ========================

class SymbolLibrary:
    """
    Manages expected symbols/annotations for validation.

    Supports:
    - Valve tags (V-101, V-102, etc.)
    - Instrument IDs (FT-201, PT-301, etc.)
    - Component abbreviations
    - Legend-extracted symbols
    """

    def __init__(self):
        self.symbols: Dict[str, List[str]] = {
            "valve": [],
            "instrument": [],
            "component": [],
            "legend": [],
            "custom": []
        }
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, str]:
        """Build regex patterns for common symbol types."""
        import re
        return {
            "valve": r"^[VG]-\d{2,4}[A-Z]?$",  # V-101, G-205A
            "instrument": r"^[A-Z]{2,3}-\d{2,4}[A-Z]?$",  # FT-201, PIT-301A
            "drawing_ref": r"^DWG-\d{4,6}$",  # DWG-12345
        }

    def add_from_legend(self, legend_text: str):
        """Extract symbols from legend text."""
        import re

        # Extract valve patterns
        valves = re.findall(r'\b[VG]-\d{2,4}[A-Z]?\b', legend_text)
        self.symbols["legend"].extend(valves)

        # Extract instrument patterns
        instruments = re.findall(r'\b[A-Z]{2,3}-\d{2,4}[A-Z]?\b', legend_text)
        self.symbols["legend"].extend(instruments)

    def add_custom(self, category: str, symbols: List[str]):
        """Add custom symbols to library."""
        if category not in self.symbols:
            self.symbols[category] = []
        self.symbols[category].extend(symbols)

    def validate(self, text: str, fuzzy_threshold: int = 85) -> Dict:
        """
        Validate text against symbol library.

        Returns:
            {
                "is_valid": bool,
                "category": str or None,
                "confidence": int (0-100),
                "matched_symbol": str or None,
                "is_noise": bool
            }
        """
        import re

        # Check exact matches first
        for category, symbols in self.symbols.items():
            if text in symbols:
                return {
                    "is_valid": True,
                    "category": category,
                    "confidence": 100,
                    "matched_symbol": text,
                    "is_noise": False
                }

        # Check regex patterns
        for category, pattern in self._patterns.items():
            if re.match(pattern, text):
                return {
                    "is_valid": True,
                    "category": category,
                    "confidence": 95,
                    "matched_symbol": None,
                    "is_noise": False
                }

        # Fuzzy matching
        all_symbols = []
        for symbols in self.symbols.values():
            all_symbols.extend(symbols)

        if all_symbols:
            match = process.extractOne(text, all_symbols, scorer=fuzz.ratio)
            if match and match[1] >= fuzzy_threshold:
                return {
                    "is_valid": True,
                    "category": "fuzzy",
                    "confidence": match[1],
                    "matched_symbol": match[0],
                    "is_noise": False
                }

        # Check if it's likely OCR noise
        is_noise = (
            len(text) < 2 or  # Too short
            not any(c.isalnum() for c in text) or  # No alphanumeric
            sum(c.isdigit() for c in text) / len(text) > 0.8  # Mostly digits (suspicious)
        )

        return {
            "is_valid": False,
            "category": None,
            "confidence": 0,
            "matched_symbol": None,
            "is_noise": is_noise
        }


# ========================
# OCR Caching
# ========================

class OCRCache:
    """Cache OCR results by page content hash."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _compute_page_hash(self, image: np.ndarray) -> str:
        """Compute hash of page image for cache key."""
        # Downsample for faster hashing
        small = cv2.resize(image, (256, 256))
        return hashlib.sha256(small.tobytes()).hexdigest()[:16]

    def get(self, image: np.ndarray) -> Optional[List[Dict]]:
        """Get cached OCR results if available."""
        key = self._compute_page_hash(image)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink()  # Remove corrupt cache

        return None

    def set(self, image: np.ndarray, results: List[Dict]):
        """Cache OCR results."""
        key = self._compute_page_hash(image)
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        except Exception:
            pass  # Fail silently if caching fails


# ========================
# Image Preprocessing
# ========================

def preprocess_for_ocr(image: np.ndarray, config: EnhancedOCRConfig) -> np.ndarray:
    """
    Preprocess image for better OCR results.

    - Bilateral blur to preserve edges while reducing noise
    - Adaptive thresholding for varying contrast
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Bilateral blur: reduces noise while preserving edges (thin lettering)
    if config.use_bilateral_blur:
        gray = cv2.bilateralFilter(
            gray,
            config.bilateral_d,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space
        )

    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ========================
# Callout Detection
# ========================

def detect_callouts(image: np.ndarray, config: EnhancedOCRConfig) -> List[Tuple[int, int, int, int]]:
    """
    Detect graphical callouts (leader lines, balloons) in engineering drawings.

    Returns:
        List of bounding boxes (x, y, w, h) for detected callouts
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    callouts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_contour_area:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Expand region to catch nearby text
        exp = config.callout_expansion_px
        x = max(0, x - exp)
        y = max(0, y - exp)
        w = w + 2 * exp
        h = h + 2 * exp

        callouts.append((x, y, w, h))

    return callouts


# ========================
# Dual-Pass OCR
# ========================

def run_dual_pass_ocr(
    image: np.ndarray,
    config: EnhancedOCRConfig
) -> List[Dict]:
    """
    Run OCR with two PSM modes and merge results.

    - PSM 11: Sparse text (good for rotated labels)
    - PSM 6: Uniform block (good for text blocks)

    Keeps results with higher confidence.
    """
    results = []

    # Pass 1: Sparse text (PSM 11)
    custom_config = f'--psm {config.psm_sparse} --oem 3'
    data1 = pytesseract.image_to_data(
        image,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    # Pass 2: Block text (PSM 6)
    custom_config = f'--psm {config.psm_block} --oem 3'
    data2 = pytesseract.image_to_data(
        image,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    # Merge results, keeping higher confidence
    boxes_seen = set()

    for data in [data1, data2]:
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf < config.min_conf:
                continue

            text = data['text'][i].strip()
            if not text:
                continue

            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Use box center as unique key
            box_key = (x + w//2, y + h//2)

            if box_key not in boxes_seen:
                boxes_seen.add(box_key)
                results.append({
                    "text": text,
                    "bbox": (x, y, x + w, y + h),
                    "confidence": conf,
                    "psm": config.psm_sparse if data == data1 else config.psm_block
                })

    return results


# ========================
# Main OCR Function
# ========================

def enhanced_ocr(
    pdf_path: str,
    page_index: int,
    config: EnhancedOCRConfig,
    symbol_library: Optional[SymbolLibrary] = None,
    tiles_pdf: Optional[List[Tuple[float, float, float, float]]] = None
) -> List[Dict]:
    """
    Enhanced OCR with all improvements.

    Args:
        pdf_path: Path to PDF
        page_index: 0-based page index
        config: Enhanced OCR configuration
        symbol_library: Optional symbol library for validation
        tiles_pdf: Optional list of regions to OCR (in PDF coordinates)

    Returns:
        List of OCR spans with validation metadata:
        {
            "text": str,
            "bbox": (x0, y0, x1, y1) in PDF coords,
            "confidence": int,
            "validation": {...},  # If symbol_library provided
            "source_method": str  # "dual_pass", "callout", "tile"
        }
    """
    # Render page at high DPI
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = config.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()

    # Check cache
    cache = None
    if config.enable_caching:
        cache = OCRCache(config.cache_dir)
        cached = cache.get(image)
        if cached is not None:
            return cached

    # Preprocess
    processed = preprocess_for_ocr(image, config)

    results = []

    # Detect callouts if enabled
    callout_regions = []
    if config.detect_callouts:
        callout_regions = detect_callouts(processed, config)

    # Determine OCR regions
    if tiles_pdf:
        # Use provided tiles (changed-cells mode)
        for x0, y0, x1, y1 in tiles_pdf:
            px0, py0 = int(x0 * zoom), int(y0 * zoom)
            px1, py1 = int(x1 * zoom), int(y1 * zoom)
            tile = processed[py0:py1, px0:px1]

            if config.enable_dual_pass:
                tile_results = run_dual_pass_ocr(tile, config)
            else:
                # Single-pass fallback
                custom_config = f'--psm {config.psm_sparse} --oem 3'
                data = pytesseract.image_to_data(tile, config=custom_config, output_type=pytesseract.Output.DICT)
                tile_results = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) >= config.min_conf and data['text'][i].strip():
                        tile_results.append({
                            "text": data['text'][i].strip(),
                            "bbox": (data['left'][i], data['top'][i],
                                   data['left'][i] + data['width'][i],
                                   data['top'][i] + data['height'][i]),
                            "confidence": int(data['conf'][i])
                        })

            # Convert back to PDF coordinates
            for r in tile_results:
                bx0, by0, bx1, by1 = r["bbox"]
                r["bbox"] = (
                    (px0 + bx0) / zoom,
                    (py0 + by0) / zoom,
                    (px0 + bx1) / zoom,
                    (py0 + by1) / zoom
                )
                r["source_method"] = "tile"
                results.append(r)

    elif callout_regions:
        # OCR detected callout regions
        for x, y, w, h in callout_regions:
            roi = processed[y:y+h, x:x+w]

            if config.enable_dual_pass:
                roi_results = run_dual_pass_ocr(roi, config)
            else:
                custom_config = f'--psm {config.psm_sparse} --oem 3'
                data = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)
                roi_results = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) >= config.min_conf and data['text'][i].strip():
                        roi_results.append({
                            "text": data['text'][i].strip(),
                            "bbox": (data['left'][i], data['top'][i],
                                   data['left'][i] + data['width'][i],
                                   data['top'][i] + data['height'][i]),
                            "confidence": int(data['conf'][i])
                        })

            # Convert to PDF coordinates
            for r in roi_results:
                bx0, by0, bx1, by1 = r["bbox"]
                r["bbox"] = (
                    (x + bx0) / zoom,
                    (y + by0) / zoom,
                    (x + bx1) / zoom,
                    (y + by1) / zoom
                )
                r["source_method"] = "callout"
                results.append(r)

    else:
        # Full page OCR
        if config.enable_dual_pass:
            results = run_dual_pass_ocr(processed, config)
        else:
            custom_config = f'--psm {config.psm_sparse} --oem 3'
            data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            for i in range(len(data['text'])):
                if int(data['conf'][i]) >= config.min_conf and data['text'][i].strip():
                    results.append({
                        "text": data['text'][i].strip(),
                        "bbox": (data['left'][i], data['top'][i],
                               data['left'][i] + data['width'][i],
                               data['top'][i] + data['height'][i]),
                        "confidence": int(data['conf'][i])
                    })

        # Convert to PDF coordinates
        for r in results:
            bx0, by0, bx1, by1 = r["bbox"]
            r["bbox"] = (bx0 / zoom, by0 / zoom, bx1 / zoom, by1 / zoom)
            r["source_method"] = "full_page"

    # Validate against symbol library
    if symbol_library and config.enable_symbol_validation:
        for r in results:
            r["validation"] = symbol_library.validate(
                r["text"],
                config.fuzzy_match_threshold
            )

    # Cache results
    if cache:
        cache.set(image, results)

    return results


__all__ = [
    "EnhancedOCRConfig",
    "SymbolLibrary",
    "enhanced_ocr",
]
