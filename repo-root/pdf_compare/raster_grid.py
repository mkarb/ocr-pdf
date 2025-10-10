# pdf_compare/raster_grid.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import fitz  # PyMuPDF
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

def _render_gray(pdf_path: str, page_index: int, dpi: int) -> tuple[np.ndarray, float]:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, zoom

def _align_ecc(base: np.ndarray, mov: np.ndarray) -> np.ndarray:
    h, w = base.shape[:2]
    b = cv2.GaussianBlur(base, (5,5), 0)
    m = cv2.GaussianBlur(mov,  (5,5), 0)
    warp = np.eye(2, 3, dtype=np.float32)  # rotation+translation
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        cv2.findTransformECC(b, m, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except Exception:
        pass
    return cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _mask_diff(base: np.ndarray, aligned: np.ndarray, method: str) -> np.ndarray:
    b = cv2.GaussianBlur(base, (3,3), 0)
    m = cv2.GaussianBlur(aligned, (3,3), 0)
    if method == "ssim" and HAVE_SKIMAGE:
        _, diff = ssim(b, m, full=True)
        mask = ((1.0 - diff) > 0.15).astype(np.uint8) * 255
    else:
        d = cv2.absdiff(b, m)
        _, th1 = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
        e1 = cv2.Canny(b, 50, 120)
        e2 = cv2.Canny(m, 50, 120)
        mask = cv2.bitwise_or(th1, cv2.bitwise_xor(e1, e2))
    # light cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def raster_grid_changed_boxes(
    old_pdf_path: str,
    new_pdf_path: str,
    page_index: int,
    *,
    dpi: int = 400,
    rows: int = 12,
    cols: int = 16,
    method: str = "hybrid",            # "abs" | "ssim" | "hybrid"
    cell_change_ratio: float = 0.03,   # ≥3% changed pixels marks the cell
    merge_adjacent: bool = True
) -> List[Tuple[float, float, float, float]]:
    """
    Returns PDF-space boxes (x0,y0,x1,y1) for grid cells flagged as changed.
    page_index is 0-based.
    """
    img_old, zoom = _render_gray(old_pdf_path, page_index, dpi)
    img_new, _    = _render_gray(new_pdf_path, page_index, dpi)
    img_new = _align_ecc(img_old, img_new)

    mask = _mask_diff(img_old, img_new, method)
    H, W = mask.shape
    cell_h = H // rows
    cell_w = W // cols

    boxes = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = H if r == rows - 1 else (r + 1) * cell_h
            x1 = W if c == cols - 1 else (c + 1) * cell_w
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            ratio = (cell > 0).sum() / float(cell.size)
            if ratio >= cell_change_ratio:
                boxes.append((x0/zoom, y0/zoom, x1/zoom, y1/zoom))

    if not merge_adjacent or not boxes:
        return boxes

    # simple merge of touching/overlapping cells to clean overlays
    boxes.sort()
    merged = []
    for b in boxes:
        if not merged:
            merged.append(b); continue
        x0,y0,x1,y1 = b
        mx0,my0,mx1,my1 = merged[-1]
        touch = not (x1 < mx0 or mx1 < x0 or y1 < my0 or my1 < y0)
        if touch:
            merged[-1] = (min(mx0,x0), min(my0,y0), max(mx1,x1), max(my1,y1))
        else:
            merged.append(b)
    return merged
