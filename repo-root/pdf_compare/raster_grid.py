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


def _render_gray(pdf_path: str, page_index: int, dpi: float) -> tuple[np.ndarray, float]:
    """Render PDF page as grayscale image."""
    zoom = dpi / 72.0
    with fitz.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False
        )
        # Own the samples before the pixmap is released.
        img = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(pix.h, pix.w).copy()
    return img, zoom


def _page_size(pdf_path: str, page_index: int) -> tuple[float, float]:
    """Return (width, height) in PDF points for a page."""
    doc = fitz.open(pdf_path)
    try:
        if not 0 <= page_index < doc.page_count:
            raise ValueError(f"Page index {page_index} is outside {pdf_path}")
        r = doc[page_index].rect
        return r.width, r.height
    finally:
        doc.close()


def _capped_dpi(page_w: float, page_h: float, dpi: int, max_px: int) -> float:
    """
    Reduce DPI so the rendered image's longest side stays <= max_px.

    Change-detection output boxes are PDF-space (pixel / zoom), so lowering the
    render DPI does not move the result's coordinates — it just keeps a large
    E-size sheet from producing a multi-hundred-MB pixmap and an intractable ECC
    alignment on a ~17000x11000 image.
    """
    zoom = dpi / 72.0
    longest = max(page_w, page_h) * zoom
    if longest <= max_px:
        return float(dpi)
    return dpi * (max_px / longest)


def _align_ecc(base: np.ndarray, mov: np.ndarray, skip_if_similar: bool = True) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Align images using ECC algorithm.

    Returns:
        Aligned image and alignment metrics
    """
    h, w = base.shape[:2]
    # Estimate registration at bounded resolution; applying the resulting
    # transform at render resolution preserves the change detector's detail.
    scale = min(1.0, 1600.0 / max(h, w))
    estimate_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    base_small = cv2.resize(base, estimate_size, interpolation=cv2.INTER_AREA) if scale < 1 else base
    mov_small = cv2.resize(mov, estimate_size, interpolation=cv2.INTER_AREA) if scale < 1 else mov
    b = cv2.GaussianBlur(base_small, (5, 5), 0)
    m = cv2.GaussianBlur(mov_small, (5, 5), 0)

    # Short-circuit if images are already nearly identical
    if skip_if_similar:
        diff_check = cv2.absdiff(b, m)
        # Blank margins must not make shifted sparse drawings look identical.
        content = (b < 250) | (m < 250)
        similarity = 1.0 - (diff_check[content].mean() / 255.0) if content.any() else 1.0
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
        # ECC alone can settle on an unrelated edge when thin linework shifts
        # farther than its blur radius. Phase correlation supplies a coarse shift.
        shift, response = cv2.phaseCorrelate(b.astype(np.float32), m.astype(np.float32))
        if response > 0.2 and np.isfinite(shift).all():
            warp[:, 2] = shift
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, warp = cv2.findTransformECC(b, m, warp, cv2.MOTION_EUCLIDEAN, criteria)

        sx, sy = estimate_size[0] / w, estimate_size[1] / h
        warp = (np.diag([1 / sx, 1 / sy]) @ warp @ np.diag([sx, sy, 1])).astype(np.float32)
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
            # ECC returns a transform from template to moving coordinates.
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        if cv2.absdiff(base, aligned).mean() > cv2.absdiff(base, mov).mean():
            metrics.update({"success": False, "reason": "alignment_increased_difference"})
            aligned = mov
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

    if method == "ssim" and HAVE_SKIMAGE and min(b.shape) >= 3:
        window = min(7, min(b.shape))
        window -= 1 if window % 2 == 0 else 0
        _, diff = ssim(b, m, full=True, win_size=window, data_range=255)
        diff_val = 1.0 - diff
        mask = (diff_val > 0.15).astype(np.uint8) * 255
        metrics["method"] = "ssim"
        metrics["mean_diff"] = float(diff_val.mean())

    elif method == "adaptive":
        diff = cv2.absdiff(b, m)

        diff_mean = diff.mean()
        diff_std = diff.std()
        diff_max = diff.max()

        # THRESH_BINARY uses a strict greater-than comparison. A threshold
        # equal to the maximum would hide uniformly or extensively changed pages.
        auto_threshold = min(int(diff_mean + 2 * diff_std), int(diff_max) - 1)
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
        threshold_val = 18 if threshold is None else threshold
        _, abs_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        e1 = cv2.Canny(b, 40, 110)
        e2 = cv2.Canny(m, 40, 110)
        edge_diff = cv2.bitwise_xor(e1, e2)
        mask = cv2.bitwise_or(abs_mask, edge_diff)

        metrics["method"] = "hybrid"
        metrics["threshold"] = int(threshold_val)

    else:  # "abs"
        diff = cv2.absdiff(b, m)
        threshold_val = 18 if threshold is None else threshold
        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        metrics["method"] = "abs"
        metrics["threshold"] = int(threshold_val)

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
    small_size = (max(1, round(img.shape[1] / 4)), max(1, round(img.shape[0] / 4)))
    small = cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA)
    cell_counts = cv2.boxFilter(small.astype(np.float32), -1, (7, 7), normalize=True)
    content = (cell_counts > min_content_ratio).astype(np.uint8) * 255
    return cv2.resize(content, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def _merge_adjacent_boxes(
    boxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """
    Merge touching or overlapping boxes until stable.

    Unlike a single-pass merge against only the previous box, this repeatedly
    absorbs any box that touches the growing region, so chains and overlaps
    that aren't adjacent in sort order are still merged.
    """
    if not boxes:
        return boxes

    remaining = [tuple(b) for b in boxes]
    out: List[Tuple[float, float, float, float]] = []

    while remaining:
        x0, y0, x1, y1 = remaining.pop()
        absorbed = True
        while absorbed:
            absorbed = False
            keep: List[Tuple[float, float, float, float]] = []
            for bx0, by0, bx1, by1 in remaining:
                touch = not (x1 < bx0 or bx1 < x0 or y1 < by0 or by1 < y0)
                if touch:
                    x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
                    absorbed = True
                else:
                    keep.append((bx0, by0, bx1, by1))
            remaining = keep
        out.append((x0, y0, x1, y1))

    return out


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
    max_render_pixels: int = 8000,
    return_metrics: bool = False,
) -> (
    List[Tuple[float, float, float, float]]
    | Tuple[List[Tuple[float, float, float, float]], Dict[str, Any]]
):
    """
    Return changed grid cells in the new page's displayed PDF coordinates.

    The old image is aligned onto the new image so highlights follow the revised
    page. Rotated pages use their displayed orientation; overlay callers should
    declare ``coordinate_space="display"``. A page-size change flags the full new
    page, including when only a blank margin was added or removed.

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
        max_render_pixels: Cap the rendered image's longest side (large sheets
            are detected at a lower effective DPI; output boxes are unaffected)
        return_metrics: Return diagnostics
    """
    for name, value in (("rows", rows), ("cols", cols), ("max_render_pixels", max_render_pixels)):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive and finite")
    for name, value in (("cell_change_ratio", cell_change_ratio), ("min_content_ratio", min_content_ratio)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if threshold is not None and not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold must be between 0 and 255")
    if method not in {"adaptive", "ssim", "hybrid", "abs"}:
        raise ValueError(f"Unknown difference method: {method}")
    new_idx = page_index if new_page_index is None else new_page_index

    # Cap render resolution so large sheets don't OOM / stall ECC alignment.
    # Use one effective DPI for both pages so the images stay pixel-aligned.
    ow, oh = _page_size(old_pdf_path, page_index)
    nw, nh = _page_size(new_pdf_path, new_idx)
    eff_dpi = min(
        _capped_dpi(ow, oh, dpi, max_render_pixels),
        _capped_dpi(nw, nh, dpi, max_render_pixels),
    )

    img_old, zoom = _render_gray(old_pdf_path, page_index, eff_dpi)
    img_new, _ = _render_gray(new_pdf_path, new_idx, eff_dpi)
    page_size_changed = not np.allclose((ow, oh), (nw, nh), rtol=0, atol=0.001)
    common_metrics = {
        "render_dpi": round(eff_dpi, 1),
        "coordinate_space": "display",
        "base_page": "new",
        "page_size_changed": page_size_changed,
    }

    if not page_size_changed and np.array_equal(img_old, img_new):
        metrics = {
            **common_metrics,
            "identical": True,
            "change_percentage": 0.0,
            "boxes_found": 0,
        }
        return ([], metrics) if return_metrics else []

    # Preserve physical PDF units on differently sized sheets. Resizing one
    # image would distort its contents and conceal a page-size change.
    height = max(img_old.shape[0], img_new.shape[0])
    width = max(img_old.shape[1], img_new.shape[1])
    img_old = cv2.copyMakeBorder(
        img_old, 0, height - img_old.shape[0], 0, width - img_old.shape[1],
        cv2.BORDER_CONSTANT, value=255,
    )
    img_new = cv2.copyMakeBorder(
        img_new, 0, height - img_new.shape[0], 0, width - img_new.shape[1],
        cv2.BORDER_CONSTANT, value=255,
    )
    if page_size_changed:
        img_old_aligned = img_old
        align_metrics = {"skipped": True, "reason": "page_size_changed"}
    else:
        img_old_aligned, align_metrics = _align_ecc(img_new, img_old)

    content_mask = None
    cells_skipped = 0

    if skip_empty_cells:
        content_old = _detect_content_regions(img_old_aligned, white_threshold, min_content_ratio)
        content_new = _detect_content_regions(img_new, white_threshold, min_content_ratio)
        content_mask = cv2.bitwise_or(content_old, content_new)

    mask, diff_metrics = _mask_diff_adaptive(img_new, img_old_aligned, method, threshold)

    height, width = mask.shape
    # A low render cap or a tiny page can have fewer pixels than grid cells.
    effective_rows = min(rows, height)
    effective_cols = min(cols, width)
    y_edges = np.linspace(0, height, effective_rows + 1, dtype=int)
    x_edges = np.linspace(0, width, effective_cols + 1, dtype=int)

    boxes: List[Tuple[float, float, float, float]] = []
    cell_metrics: List[Dict[str, Any]] = []

    for r in range(effective_rows):
        for c in range(effective_cols):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            cell = mask[y0:y1, x0:x1]
            changed_pixels = np.count_nonzero(cell)
            ratio = changed_pixels / float(cell.size)

            if skip_empty_cells and content_mask is not None:
                cell_content = content_mask[y0:y1, x0:x1]
                content_ratio = (cell_content > 0).sum() / float(cell_content.size)
                # The blank-cell optimization must never discard a detected
                # change merely because the drawing in that cell is sparse.
                if content_ratio < min_content_ratio and (not changed_pixels or ratio < cell_change_ratio):
                    cells_skipped += 1
                    continue

            if changed_pixels and ratio >= cell_change_ratio:
                box = (min(x0 / zoom, nw), min(y0 / zoom, nh), min(x1 / zoom, nw), min(y1 / zoom, nh))
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                boxes.append(box)
                cell_metrics.append({"row": r, "col": c, "change_ratio": float(ratio)})

    if page_size_changed:
        boxes.append((0.0, 0.0, nw, nh))
    if merge_adjacent and boxes:
        boxes = _merge_adjacent_boxes(boxes)

    metrics = {
        **common_metrics,
        "identical": False,
        "change_percentage": float(np.count_nonzero(mask) / mask.size * 100),
        "alignment": align_metrics,
        "diff_detection": diff_metrics,
        "boxes_found": len(boxes),
        "grid_size": (effective_rows, effective_cols),
        "total_cells": effective_rows * effective_cols,
        "cells_skipped_empty": cells_skipped,
        "cells_processed": (effective_rows * effective_cols) - cells_skipped,
        "efficiency_gain": f"{cells_skipped / (effective_rows * effective_cols) * 100:.1f}%"
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
    max_render_pixels: int = 8000,
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
        max_render_pixels=max_render_pixels,
        return_metrics=return_metrics,
    )
