"""
Improved raster grid comparison with better sensitivity controls and diagnostics.
Addresses issues with over-sensitive change detection.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import fitz
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


def _render_gray(pdf_path: str, page_index: int, dpi: int) -> tuple[np.ndarray, float]:
    """Render PDF page as grayscale image."""
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


def _align_ecc(base: np.ndarray, mov: np.ndarray, skip_if_similar: bool = True) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Align images using ECC algorithm.

    Returns:
        Aligned image and alignment metrics
    """
    h, w = base.shape[:2]
    b = cv2.GaussianBlur(base, (5,5), 0)
    m = cv2.GaussianBlur(mov,  (5,5), 0)

    # Check if images are already very similar
    if skip_if_similar:
        diff_check = cv2.absdiff(b, m)
        similarity = 1.0 - (diff_check.mean() / 255.0)
        if similarity > 0.99:  # 99% similar
            return mov, {"skipped": True, "similarity": similarity, "translation": 0.0, "rotation": 0.0}

    warp = np.eye(2, 3, dtype=np.float32)
    metrics = {"skipped": False}

    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, warp = cv2.findTransformECC(b, m, warp, cv2.MOTION_EUCLIDEAN, criteria)

        # Calculate alignment metrics
        translation = np.sqrt(warp[0,2]**2 + warp[1,2]**2)
        rotation = np.arctan2(warp[1,0], warp[0,0]) * 180 / np.pi

        metrics.update({
            "translation": float(translation),
            "rotation": float(rotation),
            "success": True
        })

        aligned = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        metrics.update({"success": False, "error": str(e)})
        aligned = mov

    return aligned, metrics


def _mask_diff_adaptive(
    base: np.ndarray,
    aligned: np.ndarray,
    method: str = "adaptive",
    threshold: int = None
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Create difference mask with adaptive thresholding.

    Args:
        base: Base image
        aligned: Aligned comparison image
        method: "adaptive", "ssim", "hybrid", or "abs"
        threshold: Manual threshold (auto if None)
    """
    b = cv2.GaussianBlur(base, (3,3), 0)
    m = cv2.GaussianBlur(aligned, (3,3), 0)

    metrics = {}

    if method == "ssim" and HAVE_SKIMAGE:
        _, diff = ssim(b, m, full=True)
        diff_val = 1.0 - diff
        mask = ((diff_val) > 0.15).astype(np.uint8) * 255
        metrics["method"] = "ssim"
        metrics["mean_diff"] = float(diff_val.mean())

    elif method == "adaptive":
        # Calculate dynamic threshold based on image statistics
        diff = cv2.absdiff(b, m)

        # Analyze difference distribution
        diff_mean = diff.mean()
        diff_std = diff.std()
        diff_max = diff.max()

        # Adaptive threshold: mean + 2*std (catches outliers while ignoring noise)
        auto_threshold = min(int(diff_mean + 2 * diff_std), diff_max)
        auto_threshold = max(auto_threshold, 25)  # Minimum threshold of 25

        threshold_val = threshold if threshold is not None else auto_threshold

        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

        # Add edge detection for structural changes
        e1 = cv2.Canny(b, 50, 120)
        e2 = cv2.Canny(m, 50, 120)
        edge_diff = cv2.bitwise_xor(e1, e2)

        # Combine: significant pixel differences OR structural edge changes
        mask = cv2.bitwise_or(mask, edge_diff)

        metrics["method"] = "adaptive"
        metrics["diff_mean"] = float(diff_mean)
        metrics["diff_std"] = float(diff_std)
        metrics["threshold_used"] = int(threshold_val)
        metrics["diff_percentage"] = float((diff > threshold_val).sum() / diff.size * 100)

    else:  # "abs" or "hybrid"
        diff = cv2.absdiff(b, m)
        threshold_val = threshold if threshold is not None else 30  # Increased from 18
        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

        if method == "hybrid":
            e1 = cv2.Canny(b, 50, 120)
            e2 = cv2.Canny(m, 50, 120)
            mask = cv2.bitwise_or(mask, cv2.bitwise_xor(e1, e2))

        metrics["method"] = method
        metrics["threshold_used"] = int(threshold_val)
        metrics["diff_percentage"] = float((diff > threshold_val).sum() / diff.size * 100)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)  # Increased cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)   # Remove small noise

    metrics["final_change_percentage"] = float((mask > 0).sum() / mask.size * 100)

    return mask, metrics


def _detect_content_regions(img: np.ndarray, white_threshold: int = 250, min_content_ratio: float = 0.01) -> np.ndarray:
    """
    Detect regions with actual content (non-white areas).

    Args:
        img: Grayscale image
        white_threshold: Pixel value above which is considered white (0-255)
        min_content_ratio: Minimum ratio of non-white pixels to be considered content

    Returns:
        Binary mask where 255=content, 0=empty white space
    """
    # Find non-white pixels
    content_mask = (img < white_threshold).astype(np.uint8) * 255

    # Remove tiny specs (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate slightly to include borders around content
    content_mask = cv2.dilate(content_mask, kernel, iterations=2)

    return content_mask


def raster_grid_changed_boxes(
    old_pdf_path: str,
    new_pdf_path: str,
    page_index: int,
    *,
    dpi: int = 400,
    rows: int = 12,
    cols: int = 16,
    method: str = "adaptive",
    cell_change_ratio: float = 0.05,  # Increased from 0.03 to 0.05 (5%)
    merge_adjacent: bool = True,
    threshold: int = None,
    skip_empty_cells: bool = True,  # NEW: Skip cells with mostly white space
    white_threshold: int = 250,     # NEW: What pixel value is considered white
    min_content_ratio: float = 0.10, # NEW: Minimum 10% content to process cell
    return_metrics: bool = False
) -> List[Tuple[float, float, float, float]] | Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]:
    """
    Returns PDF-space boxes for grid cells with significant changes.

    Args:
        old_pdf_path: Path to old PDF
        new_pdf_path: Path to new PDF
        page_index: Page number (0-based)
        dpi: Rendering resolution
        rows: Grid rows
        cols: Grid columns
        method: "adaptive" (recommended), "ssim", "hybrid", or "abs"
        cell_change_ratio: Minimum ratio of changed pixels to flag cell (0.0-1.0)
        merge_adjacent: Merge touching cells
        threshold: Manual pixel difference threshold (None=auto)
        return_metrics: Return diagnostics

    Returns:
        List of (x0, y0, x1, y1) boxes in PDF space
        If return_metrics=True, also returns metrics dict
    """
    img_old, zoom = _render_gray(old_pdf_path, page_index, dpi)
    img_new, _    = _render_gray(new_pdf_path, page_index, dpi)

    # Check if images are identical (same document)
    if np.array_equal(img_old, img_new):
        metrics = {
            "identical": True,
            "change_percentage": 0.0,
            "boxes_found": 0
        }
        return ([], metrics) if return_metrics else []

    # Align images
    img_new_aligned, align_metrics = _align_ecc(img_old, img_new)

    # Detect content regions to skip empty white space
    content_mask = None
    cells_skipped = 0

    if skip_empty_cells:
        # Detect content in both images
        content_old = _detect_content_regions(img_old, white_threshold, min_content_ratio)
        content_new = _detect_content_regions(img_new_aligned, white_threshold, min_content_ratio)
        # Union of content regions (process if either has content)
        content_mask = cv2.bitwise_or(content_old, content_new)

    # Detect differences
    mask, diff_metrics = _mask_diff_adaptive(img_old, img_new_aligned, method, threshold)

    H, W = mask.shape
    cell_h = H // rows
    cell_w = W // cols

    boxes = []
    cell_metrics = []

    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = H if r == rows - 1 else (r + 1) * cell_h
            x1 = W if c == cols - 1 else (c + 1) * cell_w

            # Check if cell has enough content (skip mostly empty white space)
            if skip_empty_cells and content_mask is not None:
                cell_content = content_mask[y0:y1, x0:x1]
                content_ratio = (cell_content > 0).sum() / float(cell_content.size)

                if content_ratio < min_content_ratio:
                    cells_skipped += 1
                    continue  # Skip this cell - it's mostly white space

            cell = mask[y0:y1, x0:x1]

            if cell.size == 0:
                continue

            ratio = (cell > 0).sum() / float(cell.size)

            if ratio >= cell_change_ratio:
                boxes.append((x0/zoom, y0/zoom, x1/zoom, y1/zoom))
                cell_metrics.append({"row": r, "col": c, "change_ratio": float(ratio)})

    if merge_adjacent and boxes:
        boxes = _merge_adjacent_boxes(boxes)

    metrics = {
        "identical": False,
        "alignment": align_metrics,
        "diff_detection": diff_metrics,
        "boxes_found": len(boxes),
        "grid_size": (rows, cols),
        "total_cells": rows * cols,
        "cells_skipped_empty": cells_skipped,
        "cells_processed": (rows * cols) - cells_skipped,
        "efficiency_gain": f"{cells_skipped / (rows * cols) * 100:.1f}%" if cells_skipped > 0 else "0%",
        "cell_change_threshold": cell_change_ratio,
        "cells_with_changes": cell_metrics[:10]  # First 10 changed cells
    }

    if return_metrics:
        return boxes, metrics
    return boxes


def _merge_adjacent_boxes(boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """Merge touching or overlapping boxes."""
    if not boxes:
        return boxes

    boxes = sorted(boxes)
    merged = []

    for b in boxes:
        if not merged:
            merged.append(b)
            continue

        x0, y0, x1, y1 = b
        mx0, my0, mx1, my1 = merged[-1]
        touch = not (x1 < mx0 or mx1 < x0 or y1 < my0 or my1 < y0)

        if touch:
            merged[-1] = (min(mx0, x0), min(my0, y0), max(mx1, x1), max(my1, y1))
        else:
            merged.append(b)

    return merged
