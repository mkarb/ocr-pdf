"""
Improved raster grid comparison with adaptive alignment, noise reduction, and diagnostics.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import fitz  # PyMuPDF
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
    b = cv2.GaussianBlur(base, (5, 5), 0)
    m = cv2.GaussianBlur(mov, (5, 5), 0)

    # Short-circuit if images are already nearly identical
    if skip_if_similar:
        diff_check = cv2.absdiff(b, m)
        similarity = 1.0 - (diff_check.mean() / 255.0)
        if similarity > 0.99:  # 99% similar
            return mov, {
                "skipped": True,
                "similarity": similarity,
                "translation": 0.0,
                "rotation": 0.0,
            }

    warp = np.eye(2, 3, dtype=np.float32)
    metrics: Dict[str, Any] = {"skipped": False}

    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, warp = cv2.findTransformECC(b, m, warp, cv2.MOTION_EUCLIDEAN, criteria)

        translation = np.sqrt(warp[0, 2] ** 2 + warp[1, 2] ** 2)
        rotation = np.arctan2(warp[1, 0], warp[0, 0]) * 180 / np.pi

        metrics.update(
            {
                "translation": float(translation),
                "rotation": float(rotation),
                "success": True,
            }
        )

        aligned = cv2.warpAffine(
            mov,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception as exc:  # pragma: no cover - fallback path
        metrics.update({"success": False, "error": str(exc)})
        aligned = mov

    return aligned, metrics


def _mask_diff_adaptive(
    base: np.ndarray,
    aligned: np.ndarray,
    method: str = "adaptive",
    threshold: Optional[int] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Create difference mask with adaptive thresholding.

    Args:
        base: Base image
        aligned: Aligned comparison image
        method: "adaptive", "ssim", "hybrid", or "abs"
        threshold: Manual threshold (auto if None)
    """
    b = cv2.GaussianBlur(base, (3, 3), 0)
    m = cv2.GaussianBlur(aligned, (3, 3), 0)

    metrics: Dict[str, Any] = {}

    if method == "ssim" and HAVE_SKIMAGE:
        _, diff = ssim(b, m, full=True)
        diff_val = 1.0 - diff
        mask = (diff_val > 0.15).astype(np.uint8) * 255
        metrics["method"] = "ssim"
        metrics["mean_diff"] = float(diff_val.mean())

    elif method == "adaptive":
        diff = cv2.absdiff(b, m)

        diff_mean = diff.mean()
        diff_std = diff.std()
        diff_max = diff.max()

        auto_threshold = min(int(diff_mean + 2 * diff_std), int(diff_max))
        auto_threshold = max(auto_threshold, 25)

        threshold_val = threshold if threshold is not None else auto_threshold
        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

        e1 = cv2.Canny(b, 50, 120)
        e2 = cv2.Canny(m, 50, 120)
        edge_diff = cv2.bitwise_xor(e1, e2)

        mask = cv2.bitwise_or(mask, edge_diff)

        metrics["method"] = "adaptive"
        metrics["threshold"] = int(threshold_val)
        metrics["mean_diff"] = float(diff_mean)
        metrics["std_diff"] = float(diff_std)

    elif method == "hybrid":
        diff = cv2.absdiff(b, m)
        _, abs_mask = cv2.threshold(diff, threshold or 18, 255, cv2.THRESH_BINARY)
        e1 = cv2.Canny(b, 40, 110)
        e2 = cv2.Canny(m, 40, 110)
        edge_diff = cv2.bitwise_xor(e1, e2)
        mask = cv2.bitwise_or(abs_mask, edge_diff)

        metrics["method"] = "hybrid"
        metrics["threshold"] = int(threshold or 18)

    else:  # "abs"
        diff = cv2.absdiff(b, m)
        _, mask = cv2.threshold(diff, threshold or 18, 255, cv2.THRESH_BINARY)
        metrics["method"] = "abs"
        metrics["threshold"] = int(threshold or 18)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, metrics


def _detect_content_regions(
    img: np.ndarray, white_threshold: int = 250, min_content_ratio: float = 0.01
) -> np.ndarray:
    """Detect content-rich areas to skip blank cells."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, white_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Downsample for performance
    small = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cell_counts = cv2.boxFilter(small.astype(np.float32), -1, (7, 7), normalize=True)
    content = (cell_counts > min_content_ratio).astype(np.uint8) * 255
    return cv2.resize(content, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def _merge_adjacent_boxes(
    boxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """Merge touching or overlapping boxes."""
    if not boxes:
        return boxes

    boxes = sorted(boxes)
    merged: List[Tuple[float, float, float, float]] = []

    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x0, y0, x1, y1 = box
        mx0, my0, mx1, my1 = merged[-1]
        touch = not (x1 < mx0 or mx1 < x0 or y1 < my0 or my1 < y0)

        if touch:
            merged[-1] = (
                min(mx0, x0),
                min(my0, y0),
                max(mx1, x1),
                max(my1, y1),
            )
        else:
            merged.append(box)

    return merged


def raster_grid_changed_boxes(
    old_pdf_path: str,
    new_pdf_path: str,
    page_index: int,
    *,
    new_page_index: Optional[int] = None,
    dpi: int = 400,
    rows: int = 12,
    cols: int = 16,
    method: str = "adaptive",
    cell_change_ratio: float = 0.05,
    merge_adjacent: bool = True,
    threshold: Optional[int] = None,
    skip_empty_cells: bool = True,
    white_threshold: int = 250,
    min_content_ratio: float = 0.10,
    return_metrics: bool = False,
) -> (
    List[Tuple[float, float, float, float]]
    | Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]
):
    """
    Returns PDF-space boxes for grid cells with significant changes.

    Args:
        old_pdf_path: Path to old PDF
        new_pdf_path: Path to new PDF
        page_index: Page number (0-based) for the old PDF (or both PDFs if new_page_index is None)
        new_page_index: Alternate page number for the new PDF
        dpi: Rendering resolution
        rows: Grid rows
        cols: Grid columns
        method: "adaptive" (recommended), "ssim", "hybrid", or "abs"
        cell_change_ratio: Minimum ratio of changed pixels to flag cell (0.0-1.0)
        merge_adjacent: Merge touching cells
        threshold: Manual pixel difference threshold (None=auto)
        skip_empty_cells: Skip mostly-empty cells to reduce noise
        white_threshold: Pixel value considered white for content detection
        min_content_ratio: Minimum ratio of content pixels to process a cell
        return_metrics: Return diagnostics
    """
    img_old, zoom = _render_gray(old_pdf_path, page_index, dpi)
    new_idx = page_index if new_page_index is None else new_page_index
    img_new, _ = _render_gray(new_pdf_path, new_idx, dpi)

    if np.array_equal(img_old, img_new):
        metrics = {
            "identical": True,
            "change_percentage": 0.0,
            "boxes_found": 0,
        }
        return ([], metrics) if return_metrics else []

    img_new_aligned, align_metrics = _align_ecc(img_old, img_new)

    content_mask = None
    cells_skipped = 0

    if skip_empty_cells:
        content_old = _detect_content_regions(img_old, white_threshold, min_content_ratio)
        content_new = _detect_content_regions(img_new_aligned, white_threshold, min_content_ratio)
        content_mask = cv2.bitwise_or(content_old, content_new)

    mask, diff_metrics = _mask_diff_adaptive(img_old, img_new_aligned, method, threshold)

    height, width = mask.shape
    cell_h = height // rows
    cell_w = width // cols

    boxes: List[Tuple[float, float, float, float]] = []
    cell_metrics: List[Dict[str, Any]] = []

    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = height if r == rows - 1 else (r + 1) * cell_h
            x1 = width if c == cols - 1 else (c + 1) * cell_w

            if skip_empty_cells and content_mask is not None:
                cell_content = content_mask[y0:y1, x0:x1]
                content_ratio = (cell_content > 0).sum() / float(cell_content.size)
                if content_ratio < min_content_ratio:
                    cells_skipped += 1
                    continue

            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            ratio = (cell > 0).sum() / float(cell.size)

            if ratio >= cell_change_ratio:
                boxes.append((x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom))
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
        "efficiency_gain": f"{cells_skipped / (rows * cols) * 100:.1f}%"
        if cells_skipped > 0
        else "0%",
        "cell_change_threshold": cell_change_ratio,
        "cells_with_changes": cell_metrics[:10],
    }

    return (boxes, metrics) if return_metrics else boxes


def raster_grid_changed_boxes_aligned(
    old_pdf_path: str,
    new_pdf_path: str,
    old_page_index: int,
    new_page_index: int,
    *,
    dpi: int = 400,
    rows: int = 12,
    cols: int = 16,
    method: str = "adaptive",
    cell_change_ratio: float = 0.05,
    merge_adjacent: bool = True,
    threshold: Optional[int] = None,
    skip_empty_cells: bool = True,
    white_threshold: int = 250,
    min_content_ratio: float = 0.10,
    return_metrics: bool = False,
) -> (
    List[Tuple[float, float, float, float]]
    | Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]
):
    """
    Wrapper to compare different page indices while reusing the improved raster grid engine.
    """
    return raster_grid_changed_boxes(
        old_pdf_path,
        new_pdf_path,
        page_index=old_page_index,
        new_page_index=new_page_index,
        dpi=dpi,
        rows=rows,
        cols=cols,
        method=method,
        cell_change_ratio=cell_change_ratio,
        merge_adjacent=merge_adjacent,
        threshold=threshold,
        skip_empty_cells=skip_empty_cells,
        white_threshold=white_threshold,
        min_content_ratio=min_content_ratio,
        return_metrics=return_metrics,
    )
