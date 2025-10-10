from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import fitz  # PyMuPDF
import cv2
import os, pytesseract
if os.name == "nt" and "tesseract.exe" not in pytesseract.pytesseract.tesseract_cmd.lower():
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default):
        pytesseract.pytesseract.tesseract_cmd = default

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 in PDF user space

@dataclass(frozen=True)
class HighResOCRConfig:
    dpi: int = 500            # 500–600 for fine diagrams
    psm: int = 11             # sparse text, as lines
    min_conf: int = 60        # filter weak OCR results
    lang: str = "eng"
    # The following are here for future page batching; for a single page they are inert
    max_workers: int = 1
    ram_budget_mb: int = 4096

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

def _ocr_tile(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    """
    Run Tesseract on a grayscale tile. Returns list of dicts:
    {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
    (bbox is in TILE pixel coords; caller maps to PDF coords)
    """
    # Slight denoise/contrast helps on engineering drawings
    proc = cv2.GaussianBlur(gray, (3,3), 0)
    # Use image_to_data to get boxes + confidences
    ts_cfg = f"-l {cfg.lang} --psm {cfg.psm}"
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
