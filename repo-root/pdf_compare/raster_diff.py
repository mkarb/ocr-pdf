# pdf_compare/raster_diff.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import fitz  # PyMuPDF
import cv2
import skimage
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

def _render_page_as_gray(pdf_path: str, page_index: int, dpi: int = 350):
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    doc.close()
    return gray, zoom

def _align_images(base: np.ndarray, mov: np.ndarray) -> np.ndarray:
    h, w = base.shape[:2]
    b = cv2.GaussianBlur(base, (5,5), 0)
    m = cv2.GaussianBlur(mov,  (5,5), 0)
    warp = np.eye(2, 3, dtype=np.float32)  # euclidean: rot + trans
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        cv2.findTransformECC(b, m, warp, cv2.MOTION_EUCLIDEAN, criteria)
    except Exception:
        pass
    return cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _diff_mask(base: np.ndarray, aligned: np.ndarray, method: str = "hybrid") -> np.ndarray:
    b = cv2.GaussianBlur(base, (3,3), 0)
    m = cv2.GaussianBlur(aligned, (3,3), 0)
    if method == "ssim" and HAVE_SKIMAGE:
        score, diff = ssim(b, m, full=True)
        th = (1.0 - diff) > 0.15
        mask = (th.astype(np.uint8) * 255)
    else:
        d = cv2.absdiff(b, m)
        _, th1 = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
        e1 = cv2.Canny(b, 50, 120)
        e2 = cv2.Canny(m, 50, 120)
        ex = cv2.bitwise_xor(e1, e2)
        mask = cv2.bitwise_or(th1, ex)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    return mask

def _boxes_from_mask(mask: np.ndarray, min_area: int = 120, merge_gap: int = 6):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append((x, y, x+w, y+h))
    if not boxes:
        return []
    boxes.sort()
    merged = []
    def expand(b): x0,y0,x1,y1 = b; return (x0-merge_gap, y0-merge_gap, x1+merge_gap, y1+merge_gap)
    for b in boxes:
        bx = expand(b)
        if not merged:
            merged.append(bx); continue
        x0,y0,x1,y1 = bx
        mx0,my0,mx1,my1 = merged[-1]
        if not (x1 < mx0 or mx1 < x0 or y1 < my0 or my1 < y0):
            merged[-1] = (min(mx0,x0), min(my0,y0), max(mx1,x1), max(my1,y1))
        else:
            merged.append(bx)
    return [(x0+merge_gap, y0+merge_gap, x1-merge_gap, y1-merge_gap) for (x0,y0,x1,y1) in merged]

def raster_diff_boxes(old_pdf: str, new_pdf: str, page_index: int, dpi: int = 350,
                      method: str = "hybrid", min_area: int = 120):
    img_old, zoom = _render_page_as_gray(old_pdf, page_index, dpi)
    img_new, _    = _render_page_as_gray(new_pdf, page_index, dpi)
    img_new_aligned = _align_images(img_old, img_new)
    mask = _diff_mask(img_old, img_new_aligned, method=method)
    boxes_px = _boxes_from_mask(mask, min_area=min_area)
    return [(x0/zoom, y0/zoom, x1/zoom, y1/zoom) for (x0,y0,x1,y1) in boxes_px]
