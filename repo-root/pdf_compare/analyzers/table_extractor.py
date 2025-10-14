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
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import re
import json

import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from rapidfuzz import fuzz


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
        """Convert table to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        # Build data dict with proper padding for missing cells
        data = {}
        num_cols = len(self.headers)

        for i, header in enumerate(self.headers):
            data[header] = []

        for row in self.rows:
            # Create a dict to map column index to cell text
            row_data = {}
            for cell in row.cells:
                if cell.col < num_cols:
                    row_data[cell.col] = cell.text

            # Add cells in order, padding missing columns with empty strings
            for i in range(num_cols):
                cell_text = row_data.get(i, "")
                data[self.headers[i]].append(cell_text)

        return pd.DataFrame(data)


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

        # TODO: Add text-based table detection (look for header keywords)

        return []

    def extract_table(
        self,
        pdf_path: str,
        page_index: int,
        table_bbox: Optional[Tuple[float, float, float, float]] = None
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
        # Render page
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

        # Extract headers (first row)
        headers = []
        if 0 in rows_by_idx:
            header_cells = sorted(rows_by_idx[0], key=lambda c: c.col)
            headers = [cell.text for cell in header_cells]

        # Build rows (skip header row)
        for row_idx in sorted(rows_by_idx.keys()):
            if row_idx == 0:  # Skip header
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
        page_indices: Optional[List[int]] = None
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

        for page_idx in page_indices:
            # Detect table regions
            regions = self.detect_table_regions(pdf_path, page_idx)

            if regions:
                # Extract tables from detected regions
                for region in regions:
                    table = self.extract_table(pdf_path, page_idx, region)
                    if table:
                        all_tables.append(table)
            else:
                # Try extracting from full page
                table = self.extract_table(pdf_path, page_idx)
                if table:
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
