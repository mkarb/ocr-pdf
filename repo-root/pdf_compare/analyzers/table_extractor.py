"""
Table extraction for engineering drawings.

Supports extraction of:
- Bills of Materials (BOM) / Parts Lists
- Symbol/Legend tables with abbreviations
- Line type tables
- Any structured tabular data

Uses hybrid detection approach:
- Line-based detection (Hough Transform) for bordered tables
- Whitespace analysis for borderless tables
- Header keyword detection
- Column structure recognition
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Iterable
from dataclasses import dataclass, field
from pathlib import Path
import re
import json

import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from rapidfuzz import fuzz
from collections import defaultdict

from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, MultiLineString


# ========================
# Data Structures
# ========================

@dataclass
class TableCell:
    """A single table cell with its content and location."""
    row: int
    col: int
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF coords
    confidence: float = 0.0


@dataclass
class TableRow:
    """A row in a table."""
    row_index: int
    cells: List[TableCell]
    bbox: Tuple[float, float, float, float]  # Row bounding box


@dataclass
class Table:
    """A complete table structure."""
    table_id: str
    table_type: str  # "bom", "symbols", "line_types", "generic"
    page: int  # 1-based page number
    bbox: Tuple[float, float, float, float]  # Table bounding box in PDF coords
    headers: List[str] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert table to dictionary format."""
        return {
            "table_id": self.table_id,
            "table_type": self.table_type,
            "page": self.page,
            "bbox": self.bbox,
            "headers": self.headers,
            "rows": [
                {
                    "row_index": row.row_index,
                    "cells": [
                        {
                            "col": cell.col,
                            "text": cell.text,
                            "bbox": cell.bbox,
                            "confidence": cell.confidence
                        }
                        for cell in row.cells
                    ]
                }
                for row in self.rows
            ],
            "metadata": self.metadata
        }

    def to_dataframe(self):
        """
        Convert table to pandas DataFrame, preserving empty cells and table structure.

        Empty cells detected in the original table are preserved as empty strings.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        # Handle edge case: no headers
        if not self.headers:
            # Generate default headers based on max columns found
            max_cols = max((max(cell.col for cell in row.cells) if row.cells else 0) for row in self.rows) if self.rows else 0
            self.headers = [f"Column_{i}" for i in range(max_cols + 1)]

        # Handle duplicate headers by making them unique
        seen_headers = {}
        unique_headers = []
        for header in self.headers:
            if header in seen_headers:
                seen_headers[header] += 1
                unique_headers.append(f"{header}_{seen_headers[header]}")
            else:
                seen_headers[header] = 0
                unique_headers.append(header)

        num_cols = len(unique_headers)

        # Build rows as lists (more reliable than dict for pandas)
        rows_data = []

        for row in self.rows:
            # Create a dict to map column index to cell text
            row_data = {}
            for cell in row.cells:
                # Map cell column index to text (preserving empty strings)
                if cell.col < num_cols:
                    row_data[cell.col] = cell.text

            # Build complete row, filling missing columns with empty strings
            # This preserves the table structure with empty cells
            row_list = []
            for col_idx in range(num_cols):
                cell_text = row_data.get(col_idx, "")  # Empty string if cell not found
                row_list.append(cell_text)

            rows_data.append(row_list)

        # Create DataFrame from rows
        return pd.DataFrame(rows_data, columns=unique_headers)


@dataclass
class TableExtractionConfig:
    """Configuration for table extraction."""
    # Rendering
    dpi: int = 400

    # Line detection (for bordered tables)
    enable_line_detection: bool = True
    hough_threshold: int = 100
    min_line_length: int = 50
    max_line_gap: int = 10

    # Whitespace detection (for borderless tables)
    enable_whitespace_detection: bool = True
    whitespace_threshold: int = 30  # Pixels of whitespace to consider column boundary

    # Cell extraction
    min_cell_width: int = 20
    min_cell_height: int = 15
    cell_padding: int = 2

    # OCR
    ocr_psm: int = 6  # Uniform block of text
    ocr_min_conf: int = 50

    # Table classification
    bom_keywords: List[str] = field(default_factory=lambda: [
        "PARTS LIST", "BILL OF MATERIALS", "BOM", "MATERIAL LIST",
        "ITEM", "QTY", "QUANTITY", "PART NUMBER", "DESCRIPTION"
    ])
    symbol_keywords: List[str] = field(default_factory=lambda: [
        "SYMBOL", "LEGEND", "ABBREVIATION", "KEY", "LINE TYPE"
    ])

    # Validation
    min_rows: int = 2  # Minimum rows to consider valid table
    min_cols: int = 2  # Minimum columns to consider valid table

    # Grid reconstruction (vector geometry driven)
    enable_bom_grid_reconstruction: bool = True
    bom_expected_headers: List[str] = field(default_factory=lambda: [
        "CAGE CODE",
        "MATERIAL",
        "PART NAME OR DESCRIPTION",
        "PART NUMBER",
        "ZONE",
        "QTY REQD",
        "FIND NO."
    ])
    bom_line_orientation_tol: float = 1.5  # Max deviation from perfect axis alignment
    bom_min_horizontal_span_ratio: float = 0.15  # Fraction of page width for horizontal grid lines
    bom_min_vertical_span_ratio: float = 0.25  # Fraction of page height for vertical grid lines
    bom_block_overlap_tolerance: float = 15.0  # How close parallel segments must be to belong to same block
    bom_row_merge_tolerance: float = 1.2  # Merge nearly coincident row lines (PDF units)
    bom_row_band_padding: float = 1.5  # Padding when assigning text to row bands (PDF units)
    bom_gap_multiplier: float = 2.2  # Gap multiplier to detect block breaks
    bom_min_rows: int = 4  # Minimum horizontal lines in a block to treat as table


# ========================
# Line Detection
# ========================

def detect_table_lines(
    image: np.ndarray,
    config: TableExtractionConfig
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Detect horizontal and vertical lines in image using Hough Transform.

    Returns:
        (horizontal_lines, vertical_lines) as lists of (x1, y1, x2, y2)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find contours for lines
    h_lines = []
    v_lines = []

    # Horizontal lines
    contours_h, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_h:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > config.min_line_length:
            h_lines.append((x, y, x + w, y))

    # Vertical lines
    contours_v, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_v:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > config.min_line_length:
            v_lines.append((x, y, x, y + h))

    return h_lines, v_lines


def find_table_cells_from_lines(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    config: TableExtractionConfig
) -> List[Tuple[int, int, int, int]]:
    """
    Find table cells from detected horizontal and vertical lines.

    Returns:
        List of cell bounding boxes (x, y, w, h) in pixel coordinates
    """
    if not h_lines or not v_lines:
        return []

    # Sort lines
    h_lines = sorted(h_lines, key=lambda l: l[1])  # Sort by y coordinate
    v_lines = sorted(v_lines, key=lambda l: l[0])  # Sort by x coordinate

    cells = []

    # Create grid cells from line intersections
    for i in range(len(h_lines) - 1):
        y_top = h_lines[i][1]
        y_bottom = h_lines[i + 1][1]

        if y_bottom - y_top < config.min_cell_height:
            continue

        for j in range(len(v_lines) - 1):
            x_left = v_lines[j][0]
            x_right = v_lines[j + 1][0]

            if x_right - x_left < config.min_cell_width:
                continue

            # Add padding
            x_left += config.cell_padding
            y_top += config.cell_padding
            x_right -= config.cell_padding
            y_bottom -= config.cell_padding

            cells.append((x_left, y_top, x_right - x_left, y_bottom - y_top))

    return cells


# ========================
# Whitespace Detection
# ========================

def detect_columns_by_whitespace(
    image: np.ndarray,
    config: TableExtractionConfig
) -> List[int]:
    """
    Detect column boundaries by analyzing vertical whitespace.

    Returns:
        List of x-coordinates representing column boundaries
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create vertical projection (sum of white pixels per column)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    vertical_projection = np.sum(binary, axis=0)

    # Find valleys (whitespace) in projection
    height = image.shape[0]
    threshold = height * 255 * 0.9  # 90% white = likely column boundary

    boundaries = []
    in_whitespace = False
    whitespace_start = 0

    for x, val in enumerate(vertical_projection):
        if val > threshold and not in_whitespace:
            in_whitespace = True
            whitespace_start = x
        elif val <= threshold and in_whitespace:
            # End of whitespace region
            if x - whitespace_start >= config.whitespace_threshold:
                boundaries.append((whitespace_start + x) // 2)  # Mid-point
            in_whitespace = False

    return boundaries


def detect_rows_by_whitespace(
    image: np.ndarray,
    config: TableExtractionConfig
) -> List[int]:
    """
    Detect row boundaries by analyzing horizontal whitespace.

    Returns:
        List of y-coordinates representing row boundaries
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create horizontal projection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    horizontal_projection = np.sum(binary, axis=1)

    # Find valleys in projection
    width = image.shape[1]
    threshold = width * 255 * 0.9

    boundaries = []
    in_whitespace = False
    whitespace_start = 0

    for y, val in enumerate(horizontal_projection):
        if val > threshold and not in_whitespace:
            in_whitespace = True
            whitespace_start = y
        elif val <= threshold and in_whitespace:
            if y - whitespace_start >= 5:  # Min row gap
                boundaries.append((whitespace_start + y) // 2)
            in_whitespace = False

    return boundaries


# ========================
# OCR and Cell Extraction
# ========================

def extract_cell_text(
    image: np.ndarray,
    cell_bbox: Tuple[int, int, int, int],
    config: TableExtractionConfig
) -> Tuple[str, float]:
    """
    Extract text from a cell using OCR.

    Args:
        image: Full page image
        cell_bbox: (x, y, w, h) in pixel coordinates
        config: Extraction config

    Returns:
        (text, confidence)
    """
    x, y, w, h = cell_bbox

    # Extract cell region
    cell_img = image[y:y+h, x:x+w]

    if cell_img.size == 0:
        return "", 0.0

    # Preprocess cell
    if len(cell_img.shape) == 3:
        cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell_gray = cell_img.copy()

    # Enhance contrast
    cell_gray = cv2.equalizeHist(cell_gray)

    # OCR
    custom_config = f'--psm {config.ocr_psm} --oem 3'
    data = pytesseract.image_to_data(
        cell_gray,
        config=custom_config,
        output_type=pytesseract.Output.DICT
    )

    # Combine text with confidence
    texts = []
    confidences = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        try:
            conf = int(data['conf'][i])
        except (ValueError, KeyError):
            conf = 0

        # Include all text, even low confidence (for table extraction we need all cells)
        if text:
            texts.append(text)
            confidences.append(conf)

    if not texts:
        return "", 0.0

    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return combined_text, avg_confidence


# ========================
# Vector geometry helpers (grid reconstruction)
# ========================

def _iter_line_strings(geometry) -> Iterable[LineString]:
    """Yield LineString objects from a shapely geometry of arbitrary nesting."""
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            yield from _iter_line_strings(geom)
    elif hasattr(geometry, "geoms"):
        for geom in geometry.geoms:
            yield from _iter_line_strings(geom)


def _ranges_overlap(a0: float, a1: float, b0: float, b1: float, pad: float) -> bool:
    """Return True when two 1D ranges overlap within a tolerance."""
    return min(a1, b1) + pad >= max(a0, b0) - pad


def _dedup_sorted(values: Iterable[float], tol: float) -> List[float]:
    """Merge nearly identical sorted positions into a single representative."""
    dedup: List[float] = []
    for value in sorted(values):
        if not dedup:
            dedup.append(value)
            continue
        if abs(value - dedup[-1]) <= tol:
            dedup[-1] = (dedup[-1] + value) / 2.0
        else:
            dedup.append(value)
    return dedup


def _boundaries_from_centers(
    centers: List[float],
    left: float,
    right: float,
    count: int
) -> List[float]:
    """Infer column boundaries from column center positions."""
    if count <= 0:
        return [left, right]

    centers_sorted = sorted(centers)
    if not centers_sorted:
        step = (right - left) / float(count)
        return [left + step * i for i in range(count + 1)]

    # Fall back to uniform spacing when we have fewer centers than needed
    if len(centers_sorted) < count:
        step = (right - left) / float(count)
        return [left + step * i for i in range(count + 1)]

    span = centers_sorted[-1] - centers_sorted[0]
    if span <= 0.01:
        step = (right - left) / float(count)
        return [left + step * i for i in range(count + 1)]

    initial = [centers_sorted[0] + span * (i + 0.5) / count for i in range(count)]
    cluster_centers = initial

    for _ in range(10):
        clusters: List[List[float]] = [[] for _ in range(count)]
        for value in centers_sorted:
            idx = min(range(count), key=lambda i: abs(value - cluster_centers[i]))
            clusters[idx].append(value)
        new_centers = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                new_centers.append(sum(cluster) / len(cluster))
            else:
                new_centers.append(cluster_centers[idx])
        if max(abs(a - b) for a, b in zip(cluster_centers, new_centers)) < 0.25:
            cluster_centers = new_centers
            break
        cluster_centers = new_centers

    cluster_centers.sort()
    boundaries = [left]
    for i in range(len(cluster_centers) - 1):
        boundaries.append((cluster_centers[i] + cluster_centers[i + 1]) / 2.0)
    boundaries.append(right)
    return boundaries


# ========================
# Table Classification
# ========================

def classify_table(
    headers: List[str],
    config: TableExtractionConfig
) -> str:
    """
    Classify table type based on headers.

    Returns:
        "bom", "symbols", "line_types", or "generic"
    """
    headers_upper = " ".join(headers).upper()

    # Check for BOM keywords
    bom_score = sum(1 for kw in config.bom_keywords if kw in headers_upper)
    if bom_score >= 2:
        return "bom"

    # Check for symbol/legend keywords
    symbol_score = sum(1 for kw in config.symbol_keywords if kw in headers_upper)
    if symbol_score >= 1:
        return "symbols"

    # Check for line types
    if "LINE" in headers_upper and ("TYPE" in headers_upper or "STYLE" in headers_upper):
        return "line_types"

    return "generic"


# ========================
# Main Extraction Class
# ========================

class TableExtractor:
    """
    Extract structured tables from engineering drawings.

    Supports:
    - Bill of Materials (BOM) / Parts Lists
    - Symbol/Legend tables
    - Line type tables
    - Generic tabular data
    """

    def __init__(self, config: Optional[TableExtractionConfig] = None):
        self.config = config or TableExtractionConfig()
        self.tables: List[Table] = []

    def _extract_bom_from_vectors(
        self,
        page_vectors,
        page_index: int,
        table_bbox: Optional[Tuple[float, float, float, float]]
    ) -> Optional[Table]:
        """Reconstruct a BOM table using vector geometry and text positions."""

        config = self.config

        if not config.enable_bom_grid_reconstruction or not page_vectors:
            return None

        orientation_tol = config.bom_line_orientation_tol
        min_h_span = page_vectors.width * config.bom_min_horizontal_span_ratio
        min_v_span = page_vectors.height * config.bom_min_vertical_span_ratio

        horizontal_segments: List[Dict[str, float]] = []
        vertical_segments: List[Dict[str, float]] = []

        bbox = table_bbox

        for geom in getattr(page_vectors, "geoms", []):
            try:
                if getattr(geom.kind, "name", "") != "STROKE":
                    continue
            except AttributeError:
                continue

            try:
                shape = shapely_wkb.loads(geom.wkb)
            except Exception:
                continue

            if shape.is_empty:
                continue

            for line in _iter_line_strings(shape):
                minx, miny, maxx, maxy = line.bounds

                if bbox:
                    if maxx < bbox[0] - orientation_tol or minx > bbox[2] + orientation_tol:
                        continue
                    if maxy < bbox[1] - orientation_tol or miny > bbox[3] + orientation_tol:
                        continue

                dx = abs(maxx - minx)
                dy = abs(maxy - miny)
                length = line.length

                if length < min(min_h_span, min_v_span):
                    continue

                if dy <= orientation_tol and dx >= min_h_span:
                    horizontal_segments.append({
                        "x0": minx,
                        "x1": maxx,
                        "y": (miny + maxy) / 2.0,
                        "length": length,
                    })
                elif dx <= orientation_tol and dy >= min_v_span:
                    vertical_segments.append({
                        "x": (minx + maxx) / 2.0,
                        "y0": min(miny, maxy),
                        "y1": max(miny, maxy),
                        "length": length,
                    })

        if len(horizontal_segments) < config.bom_min_rows:
            return None

        horizontal_segments.sort(key=lambda seg: (seg["y"], seg["x0"]))
        blocks: List[Dict[str, Any]] = []

        for seg in horizontal_segments:
            assigned = None
            for block in blocks:
                if _ranges_overlap(seg["x0"], seg["x1"], block["x0"], block["x1"], config.bom_block_overlap_tolerance):
                    block["segments"].append(seg)
                    block["x0"] = min(block["x0"], seg["x0"])
                    block["x1"] = max(block["x1"], seg["x1"])
                    block["ys"].append(seg["y"])
                    assigned = block
                    break
            if assigned is None:
                blocks.append({
                    "segments": [seg],
                    "x0": seg["x0"],
                    "x1": seg["x1"],
                    "ys": [seg["y"]],
                })

        valid_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            row_levels = _dedup_sorted(block["ys"], config.bom_row_merge_tolerance)
            if len(row_levels) < config.bom_min_rows:
                continue
            block["row_levels"] = sorted(row_levels)
            block["y0"] = block["row_levels"][0]
            block["y1"] = block["row_levels"][-1]
            valid_blocks.append(block)

        if not valid_blocks:
            return None

        text_entries = []
        for text_run in getattr(page_vectors, "texts", []):
            text_value = (getattr(text_run, "text", "") or "").strip()
            if not text_value:
                continue
            x0, y0, x1, y1 = text_run.bbox
            text_entries.append({
                "text": text_value,
                "bbox": (x0, y0, x1, y1),
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
            })

        if not text_entries:
            return None

        expected_cols = len(config.bom_expected_headers)
        valid_blocks.sort(key=lambda b: b["x0"])

        block_results: List[Dict[str, Any]] = []

        for block_index, block in enumerate(valid_blocks):
            row_levels = block.get("row_levels", [])
            if len(row_levels) < 2:
                continue

            row_bands = [
                (row_levels[i], row_levels[i + 1])
                for i in range(len(row_levels) - 1)
            ]

            block_texts = [
                entry for entry in text_entries
                if block["x0"] - config.bom_block_overlap_tolerance <= entry["cx"] <= block["x1"] + config.bom_block_overlap_tolerance
                and block["y0"] - config.bom_block_overlap_tolerance <= entry["cy"] <= block["y1"] + config.bom_block_overlap_tolerance
            ]

            if not block_texts:
                continue

            candidate_verticals = []
            for seg in vertical_segments:
                if seg["x"] < block["x0"] - config.bom_block_overlap_tolerance or seg["x"] > block["x1"] + config.bom_block_overlap_tolerance:
                    continue
                overlap = min(seg["y1"], block["y1"]) - max(seg["y0"], block["y0"])
                if overlap <= 0:
                    continue
                if overlap >= (block["y1"] - block["y0"]) * 0.6:
                    candidate_verticals.append(seg["x"])

            boundaries = _dedup_sorted(candidate_verticals, 0.8)
            if not boundaries or boundaries[0] > block["x0"] + 0.8:
                boundaries.insert(0, block["x0"])
            if boundaries[-1] < block["x1"] - 0.8:
                boundaries.append(block["x1"])
            boundaries = _dedup_sorted(boundaries, 0.8)

            if len(boundaries) - 1 < expected_cols:
                centers = [entry["cx"] for entry in block_texts]
                boundaries = _boundaries_from_centers(centers, block["x0"], block["x1"], expected_cols)
            elif len(boundaries) - 1 > expected_cols:
                while len(boundaries) - 1 > expected_cols and len(boundaries) > 2:
                    diffs = [
                        (boundaries[i + 1] - boundaries[i], i)
                        for i in range(len(boundaries) - 1)
                    ]
                    diffs.sort(key=lambda item: item[0])
                    _, idx_min = diffs[0]
                    merged = (boundaries[idx_min] + boundaries[idx_min + 1]) / 2.0
                    boundaries[idx_min] = merged
                    del boundaries[idx_min + 1]

            column_centers = [
                (boundaries[i] + boundaries[i + 1]) / 2.0
                for i in range(len(boundaries) - 1)
            ]

            if len(column_centers) != expected_cols:
                continue

            rows_draft: List[Dict[str, Any]] = []

            for top, bottom in row_bands:
                if abs(bottom - top) < 0.5:
                    continue

                row_texts = [
                    entry for entry in block_texts
                    if top - config.bom_row_band_padding <= entry["cy"] <= bottom + config.bom_row_band_padding
                ]

                if not row_texts:
                    continue

                buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for entry in row_texts:
                    col_idx = min(
                        range(len(column_centers)),
                        key=lambda i: abs(entry["cx"] - column_centers[i])
                    )
                    buckets[col_idx].append(entry)

                cells = []
                row_text_upper_parts: List[str] = []

                for col_idx in range(expected_cols):
                    entries = buckets.get(col_idx, [])
                    if entries:
                        entries.sort(key=lambda e: e["cx"])
                        text_value = " ".join(e["text"] for e in entries).strip()
                        x0 = min(e["bbox"][0] for e in entries)
                        y0 = min(e["bbox"][1] for e in entries)
                        x1 = max(e["bbox"][2] for e in entries)
                        y1 = max(e["bbox"][3] for e in entries)
                    else:
                        text_value = ""
                        x0 = boundaries[col_idx]
                        x1 = boundaries[col_idx + 1]
                        y0 = top
                        y1 = bottom

                    row_text_upper_parts.append(text_value.upper())
                    cells.append({
                        "col": col_idx,
                        "text": text_value,
                        "bbox": (x0, y0, x1, y1),
                    })

                row_text_upper = " ".join(part for part in row_text_upper_parts if part)
                rows_draft.append({
                    "block": block_index,
                    "cells": cells,
                    "cy": (top + bottom) / 2.0,
                    "text_upper": row_text_upper,
                })

            if rows_draft:
                block_results.append({
                    "rows": rows_draft,
                    "bbox": (block["x0"], block["y0"], block["x1"], block["y1"]),
                })

        combined_rows: List[Dict[str, Any]] = []
        block_bboxes: List[Tuple[float, float, float, float]] = []

        for result in block_results:
            block_bboxes.append(result["bbox"])
            combined_rows.extend(result["rows"])

        if not combined_rows:
            return None

        combined_rows.sort(key=lambda row: (row["block"], row["cy"]))

        header_keywords = [kw.upper() for kw in config.bom_expected_headers]
        header_threshold = max(3, len(header_keywords) // 2)
        header_idx = None

        for idx, row in enumerate(combined_rows):
            score = sum(1 for kw in header_keywords if kw in row["text_upper"])
            if score >= header_threshold:
                header_idx = idx
                break

        if header_idx is None:
            return None

        data_rows_info = [
            row for idx, row in enumerate(combined_rows)
            if row["text_upper"] and sum(1 for kw in header_keywords if kw in row["text_upper"]) < header_threshold
        ]

        if not data_rows_info:
            return None

        data_rows_info.sort(key=lambda row: (row["block"], row["cy"]))

        if block_bboxes:
            global_x0 = min(b[0] for b in block_bboxes)
            global_y0 = min(b[1] for b in block_bboxes)
            global_x1 = max(b[2] for b in block_bboxes)
            global_y1 = max(b[3] for b in block_bboxes)
        else:
            global_x0, global_y0, global_x1, global_y1 = 0.0, 0.0, page_vectors.width, page_vectors.height

        headers = list(config.bom_expected_headers)
        table_rows: List[TableRow] = []

        for row_idx, row in enumerate(data_rows_info):
            cell_objs: List[TableCell] = []
            for cell in row["cells"]:
                cell_objs.append(TableCell(
                    row=row_idx,
                    col=cell["col"],
                    text=cell["text"],
                    bbox=cell["bbox"],
                    confidence=1.0,
                ))

            if cell_objs:
                row_bbox = (
                    min(c.bbox[0] for c in cell_objs),
                    min(c.bbox[1] for c in cell_objs),
                    max(c.bbox[2] for c in cell_objs),
                    max(c.bbox[3] for c in cell_objs),
                )
            else:
                row_bbox = (global_x0, row["cy"], global_x1, row["cy"])

            table_rows.append(TableRow(row_index=row_idx, cells=cell_objs, bbox=row_bbox))

        if not table_rows:
            return None

        table = Table(
            table_id=f"table_{page_index}_grid",
            table_type="bom",
            page=page_vectors.page_number,
            bbox=(global_x0, global_y0, global_x1, global_y1),
            headers=headers,
            rows=table_rows,
            metadata={
                "num_rows": len(table_rows),
                "num_cols": len(headers),
                "detection_method": "grid",
                "blocks": len(block_results),
            }
        )

        return table

    def detect_table_regions(
        self,
        pdf_path: str,
        page_index: int
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect regions likely containing tables.

        Returns:
            List of bounding boxes in PDF coordinates
        """
        # Render page
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        zoom = self.config.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect lines
        if self.config.enable_line_detection:
            h_lines, v_lines = detect_table_lines(gray, self.config)

            if h_lines and v_lines:
                # Find bounding box of table
                all_x = [x for line in h_lines for x in [line[0], line[2]]]
                all_x.extend([x for line in v_lines for x in [line[0], line[2]]])
                all_y = [y for line in h_lines for y in [line[1], line[3]]]
                all_y.extend([y for line in v_lines for y in [line[1], line[3]]])

                if all_x and all_y:
                    x0, x1 = min(all_x), max(all_x)
                    y0, y1 = min(all_y), max(all_y)

                    # Convert to PDF coords
                    return [(x0/zoom, y0/zoom, x1/zoom, y1/zoom)]

        # Text-based BOM detection for engineering drawings
        # Look for "PARTS LIST" text and find table region above it
        doc = fitz.open(pdf_path)
        page = doc[page_index]

        # Search for BOM indicators
        bom_keywords = ["PARTS LIST", "BILL OF MATERIALS", "BOM", "MATERIAL LIST"]

        for keyword in bom_keywords:
            text_instances = page.search_for(keyword)

            if text_instances:
                # Found BOM title - estimate table region
                # Typically, BOM tables are in the bottom-right corner of engineering drawings

                for title_bbox in text_instances:
                    # Get page dimensions
                    page_width = page.rect.width
                    page_height = page.rect.height

                    # BOM tables typically extend:
                    # - From right edge inward (usually 40-60% of page width)
                    # - From bottom upward to about 20-60% of page height

                    title_y = title_bbox.y1  # Bottom of title text

                    # Estimate table region:
                    # - Right edge to left (assume table is ~50% of page width)
                    # - From title upward (assume table is up to 50% of page height)

                    x0 = page_width * 0.05  # 5% margin from left
                    y0 = max(0, title_y - (page_height * 0.6))  # Up to 60% of page height
                    x1 = page_width * 0.95  # 95% of page width
                    y1 = title_y + 20  # Include title area

                    doc.close()
                    return [(x0, y0, x1, y1)]

        doc.close()
        return []

    def extract_table(
        self,
        pdf_path: str,
        page_index: int,
        table_bbox: Optional[Tuple[float, float, float, float]] = None,
        page_vectors: Optional[Any] = None,
    ) -> Optional[Table]:
        """
        Extract a single table from a page.

        Args:
            pdf_path: Path to PDF
            page_index: 0-based page index
            table_bbox: Optional bounding box in PDF coords. If None, auto-detect.

        Returns:
            Table object or None if extraction failed
        """
        # Try grid reconstruction using vector geometries first
        if page_vectors is not None:
            table_from_vectors = self._extract_bom_from_vectors(page_vectors, page_index, table_bbox)
            if table_from_vectors:
                return table_from_vectors

        # Render page for legacy detection / OCR fallback
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        zoom = self.config.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        doc.close()

        # If bbox provided, crop to region
        if table_bbox:
            x0, y0, x1, y1 = table_bbox
            px0, py0 = int(x0 * zoom), int(y0 * zoom)
            px1, py1 = int(x1 * zoom), int(y1 * zoom)
            image = image[py0:py1, px0:px1]
            offset_x, offset_y = px0, py0
        else:
            offset_x, offset_y = 0, 0
            table_bbox = (0, 0, pix.w / zoom, pix.h / zoom)

        # Detect table structure
        h_lines, v_lines = detect_table_lines(image, self.config)

        cells = []
        if h_lines and v_lines:
            # Line-based detection
            cells = find_table_cells_from_lines(h_lines, v_lines, self.config)
        else:
            # Fallback: whitespace-based detection
            col_boundaries = detect_columns_by_whitespace(image, self.config)
            row_boundaries = detect_rows_by_whitespace(image, self.config)

            if col_boundaries and row_boundaries:
                # Create cells from boundaries
                for i in range(len(row_boundaries) - 1):
                    for j in range(len(col_boundaries) - 1):
                        x = col_boundaries[j]
                        y = row_boundaries[i]
                        w = col_boundaries[j + 1] - x
                        h = row_boundaries[i + 1] - y
                        cells.append((x, y, w, h))

        if not cells:
            return None

        # Extract text from cells
        table_cells: List[TableCell] = []

        # Organize cells into grid
        cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))  # Sort by y, then x

        # Determine rows (group by similar y-coordinate)
        rows_dict: Dict[int, List[Tuple]] = {}
        current_row = 0
        last_y = -1

        for cell_bbox in cells_sorted:
            x, y, w, h = cell_bbox

            # New row if y changed significantly
            if last_y >= 0 and abs(y - last_y) > h * 0.3:
                current_row += 1

            if current_row not in rows_dict:
                rows_dict[current_row] = []

            rows_dict[current_row].append(cell_bbox)
            last_y = y

        # Extract text from each cell (including empty cells)
        for row_idx, row_cells in rows_dict.items():
            # Sort cells in row by x-coordinate
            row_cells = sorted(row_cells, key=lambda c: c[0])

            for col_idx, cell_bbox in enumerate(row_cells):
                x, y, w, h = cell_bbox
                text, conf = extract_cell_text(image, cell_bbox, self.config)

                # Convert back to PDF coordinates
                pdf_x0 = (offset_x + x) / zoom
                pdf_y0 = (offset_y + y) / zoom
                pdf_x1 = (offset_x + x + w) / zoom
                pdf_y1 = (offset_y + y + h) / zoom

                # Always create cell even if empty - preserves grid structure
                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=text,  # Will be empty string if no text found
                    bbox=(pdf_x0, pdf_y0, pdf_x1, pdf_y1),
                    confidence=conf
                )
                table_cells.append(cell)

        if not table_cells:
            return None

        # Organize cells into rows
        rows_final: List[TableRow] = []
        rows_by_idx: Dict[int, List[TableCell]] = {}

        for cell in table_cells:
            if cell.row not in rows_by_idx:
                rows_by_idx[cell.row] = []
            rows_by_idx[cell.row].append(cell)

        # Detect header location for engineering drawings
        # Engineering BOMs often have headers at BOTTOM with "PARTS LIST" title below headers
        header_row_idx = None
        headers = []
        is_bottom_up_bom = False

        if rows_by_idx:
            min_row = min(rows_by_idx.keys())
            max_row = max(rows_by_idx.keys())

            # Check if this looks like an engineering drawing BOM (bottom-up format)
            # Look for:
            # 1. "PARTS LIST" or "BOM" text near the bottom
            # 2. Standard BOM column headers (ITEM, QTY, PART NUMBER, DESCRIPTION) near bottom
            # 3. Data rows above the headers

            # Check last few rows for BOM indicators
            last_row_text = " ".join(cell.text.upper() for cell in rows_by_idx.get(max_row, []))
            second_last_row = max_row - 1 if max_row > 0 else max_row
            second_last_text = " ".join(cell.text.upper() for cell in rows_by_idx.get(second_last_row, [])) if second_last_row in rows_by_idx else ""

            # Scoring for bottom-up BOM detection
            bom_title_keywords = ["PARTS LIST", "BILL OF MATERIALS", "BOM", "MATERIAL LIST"]
            bom_header_keywords = ["ITEM", "QTY", "QUANTITY", "PART NUMBER", "DESCRIPTION", "MATERIAL", "ZONE", "FIND"]

            # Check if last row is a title row (centered, contains "PARTS LIST")
            last_row_has_title = any(kw in last_row_text for kw in bom_title_keywords)

            # Check if second-to-last row has BOM headers
            header_score_second_last = sum(1 for kw in bom_header_keywords if kw in second_last_text)

            # If we find "PARTS LIST" at bottom AND headers above it, it's a bottom-up BOM
            if last_row_has_title and header_score_second_last >= 3 and second_last_row in rows_by_idx:
                # Bottom-up engineering drawing BOM
                is_bottom_up_bom = True
                header_row_idx = second_last_row
                header_cells = sorted(rows_by_idx[second_last_row], key=lambda c: c.col)
                headers = [cell.text for cell in header_cells]

            # If no title row, check if last row itself has strong header keywords
            elif not last_row_has_title:
                header_score_last = sum(1 for kw in bom_header_keywords if kw in last_row_text)
                header_score_first = sum(1 for kw in bom_header_keywords if kw in " ".join(cell.text.upper() for cell in rows_by_idx.get(min_row, [])))

                if header_score_last >= 3 and header_score_last > header_score_first:
                    # Bottom-up BOM without separate title row
                    is_bottom_up_bom = True
                    header_row_idx = max_row
                    header_cells = sorted(rows_by_idx[max_row], key=lambda c: c.col)
                    headers = [cell.text for cell in header_cells]

            # Default: headers at top (standard table format)
            if header_row_idx is None and min_row in rows_by_idx:
                header_row_idx = min_row
                header_cells = sorted(rows_by_idx[min_row], key=lambda c: c.col)
                headers = [cell.text for cell in header_cells]

        # Build rows (skip header row and title row for bottom-up BOMs)
        rows_to_skip = {header_row_idx} if header_row_idx is not None else set()

        # If bottom-up BOM, also skip the last row if it's a title row
        if is_bottom_up_bom and rows_by_idx:
            max_row = max(rows_by_idx.keys())
            last_row_text = " ".join(cell.text.upper() for cell in rows_by_idx.get(max_row, []))
            if any(kw in last_row_text for kw in ["PARTS LIST", "BILL OF MATERIALS", "BOM"]):
                rows_to_skip.add(max_row)

        for row_idx in sorted(rows_by_idx.keys()):
            if row_idx in rows_to_skip:
                continue

            cells = sorted(rows_by_idx[row_idx], key=lambda c: c.col)

            if cells:
                # Calculate row bbox
                x0 = min(c.bbox[0] for c in cells)
                y0 = min(c.bbox[1] for c in cells)
                x1 = max(c.bbox[2] for c in cells)
                y1 = max(c.bbox[3] for c in cells)

                row = TableRow(
                    row_index=row_idx,
                    cells=cells,
                    bbox=(x0, y0, x1, y1)
                )
                rows_final.append(row)

        # Classify table
        table_type = classify_table(headers, self.config)

        # Create table object
        table = Table(
            table_id=f"table_{page_index}_{len(self.tables)}",
            table_type=table_type,
            page=page_index + 1,
            bbox=table_bbox,
            headers=headers,
            rows=rows_final,
            metadata={
                "num_rows": len(rows_final),
                "num_cols": len(headers),
                "detection_method": "lines" if h_lines and v_lines else "whitespace"
            }
        )

        return table

    def extract_all_tables(
        self,
        pdf_path: str,
        page_indices: Optional[List[int]] = None,
        vector_map: Optional[Any] = None,
    ) -> List[Table]:
        """
        Extract all tables from PDF.

        Args:
            pdf_path: Path to PDF
            page_indices: Optional list of 0-based page indices. If None, scan all pages.

        Returns:
            List of extracted tables
        """
        doc = fitz.open(pdf_path)

        if page_indices is None:
            page_indices = list(range(doc.page_count))

        doc.close()

        all_tables = []

        vector_lookup = {}
        if vector_map is not None:
            for pg in getattr(vector_map, "pages", []):
                try:
                    vector_lookup[pg.page_number - 1] = pg
                except AttributeError:
                    continue

        for page_idx in page_indices:
            # Detect table regions
            regions = self.detect_table_regions(pdf_path, page_idx)

            if regions:
                # Extract tables from detected regions
                for region in regions:
                    page_vectors = vector_lookup.get(page_idx)
                    table = self.extract_table(pdf_path, page_idx, region, page_vectors=page_vectors)
                    if table:
                        table.table_id = f"table_{page_idx}_{len(all_tables)}"
                        all_tables.append(table)
            else:
                # Try extracting from full page
                page_vectors = vector_lookup.get(page_idx)
                table = self.extract_table(pdf_path, page_idx, page_vectors=page_vectors)
                if table:
                    table.table_id = f"table_{page_idx}_{len(all_tables)}"
                    all_tables.append(table)

        self.tables = all_tables
        return all_tables

    def get_bom_tables(self) -> List[Table]:
        """Get all BOM/parts list tables."""
        return [t for t in self.tables if t.table_type == "bom"]

    def get_symbol_tables(self) -> List[Table]:
        """Get all symbol/legend tables."""
        return [t for t in self.tables if t.table_type == "symbols"]

    def save_tables_json(self, output_path: str):
        """Save all tables to JSON file."""
        data = [table.to_dict() for table in self.tables]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_tables_csv(self, output_dir: str):
        """Save each table as a separate CSV file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for table in self.tables:
            try:
                df = table.to_dataframe()
                csv_path = output_path / f"{table.table_id}.csv"
                df.to_csv(csv_path, index=False)
            except ImportError:
                print(f"Warning: pandas not installed, skipping CSV export for {table.table_id}")


__all__ = [
    "TableCell",
    "TableRow",
    "Table",
    "TableExtractionConfig",
    "TableExtractor",
]
