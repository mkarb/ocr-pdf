"""
Legend extraction and symbol vocabulary builder for engineering drawings.

Extracts symbols, component IDs, and annotations from legend/title blocks
to build a validation dictionary for OCR post-processing.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import json
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np


@dataclass
class LegendEntry:
    """A single legend entry mapping symbol to description."""
    symbol_id: str  # e.g., "V-101", "FT-201"
    description: str  # e.g., "Gate Valve", "Flow Transmitter"
    category: str  # e.g., "valve", "instrument", "component"
    page: int  # Page number where found (1-based)
    bbox: Optional[Tuple[float, float, float, float]] = None


class LegendExtractor:
    """
    Extract and manage legend/symbol definitions from engineering drawings.

    Supports:
    - Automatic legend page detection
    - Symbol ID extraction (valves, instruments, components)
    - Title block parsing
    - Symbol library building
    """

    def __init__(self):
        self.entries: List[LegendEntry] = []
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for common symbol types."""
        return {
            "valve": re.compile(r'\b([VG]-\d{2,4}[A-Z]?)\b'),
            "instrument": re.compile(r'\b([A-Z]{2,3}-\d{2,4}[A-Z]?)\b'),
            "equipment": re.compile(r'\b([A-Z]-\d{3,4})\b'),
            "drawing_ref": re.compile(r'\b(DWG-\d{4,6})\b'),
            "revision": re.compile(r'\bREV[:\s]*([A-Z0-9]+)\b', re.IGNORECASE),
        }

    def detect_legend_pages(self, pdf_path: str) -> List[int]:
        """
        Detect which pages likely contain legends.

        Heuristics:
        - Contains words like "LEGEND", "SYMBOL", "KEY"
        - First few pages (cover, legends typically at front)
        - High text density in structured format

        Returns:
            List of 0-based page indices
        """
        doc = fitz.open(pdf_path)
        legend_pages = []

        keywords = ["LEGEND", "SYMBOL", "KEY", "ABBREVIATION", "NOTES"]

        for page_idx in range(min(5, doc.page_count)):  # Check first 5 pages
            page = doc[page_idx]
            text = page.get_text().upper()

            # Check for legend keywords
            if any(kw in text for kw in keywords):
                legend_pages.append(page_idx)

        doc.close()
        return legend_pages

    def extract_from_page(
        self,
        pdf_path: str,
        page_index: int,
        dpi: int = 300
    ) -> List[LegendEntry]:
        """
        Extract legend entries from a specific page.

        Args:
            pdf_path: Path to PDF
            page_index: 0-based page index
            dpi: OCR resolution

        Returns:
            List of extracted legend entries
        """
        # Render page
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Get native text first (preferred)
        native_text = page.get_text()

        # OCR as fallback for image-based legends
        if len(native_text.strip()) < 100:
            ocr_text = pytesseract.image_to_string(image)
            full_text = ocr_text
        else:
            full_text = native_text

        doc.close()

        entries = []

        # Extract symbols by category
        for category, pattern in self._patterns.items():
            matches = pattern.findall(full_text)
            for match in matches:
                if isinstance(match, tuple):
                    symbol_id = match[0]
                else:
                    symbol_id = match

                # Try to extract description (text following the symbol)
                desc_pattern = re.compile(
                    rf'{re.escape(symbol_id)}[:\s-]+([A-Za-z\s]+?)(?:\n|$|[;\.])',
                    re.IGNORECASE
                )
                desc_match = desc_pattern.search(full_text)
                description = desc_match.group(1).strip() if desc_match else ""

                entry = LegendEntry(
                    symbol_id=symbol_id,
                    description=description,
                    category=category,
                    page=page_index + 1  # 1-based
                )
                entries.append(entry)

        return entries

    def extract_from_pdf(self, pdf_path: str) -> List[LegendEntry]:
        """
        Extract all legend entries from a PDF.

        Auto-detects legend pages and extracts symbols.

        Returns:
            List of all extracted entries
        """
        legend_pages = self.detect_legend_pages(pdf_path)

        if not legend_pages:
            # Fallback: extract from first page only
            legend_pages = [0]

        all_entries = []
        for page_idx in legend_pages:
            entries = self.extract_from_page(pdf_path, page_idx)
            all_entries.extend(entries)

        self.entries = all_entries
        return all_entries

    def build_symbol_library(self) -> Dict[str, List[str]]:
        """
        Build a symbol library dictionary from extracted entries.

        Returns:
            Dictionary mapping categories to symbol lists
        """
        library: Dict[str, List[str]] = {}

        for entry in self.entries:
            if entry.category not in library:
                library[entry.category] = []
            if entry.symbol_id not in library[entry.category]:
                library[entry.category].append(entry.symbol_id)

        return library

    def to_json(self, output_path: str):
        """Save legend entries to JSON file."""
        data = [
            {
                "symbol_id": e.symbol_id,
                "description": e.description,
                "category": e.category,
                "page": e.page
            }
            for e in self.entries
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def from_json(self, input_path: str):
        """Load legend entries from JSON file."""
        with open(input_path, "r") as f:
            data = json.load(f)

        self.entries = [
            LegendEntry(
                symbol_id=item["symbol_id"],
                description=item["description"],
                category=item["category"],
                page=item["page"]
            )
            for item in data
        ]

    def diff_legends(
        self,
        other: "LegendExtractor"
    ) -> Dict[str, List[LegendEntry]]:
        """
        Compare two legend extractors to find changes.

        Args:
            other: Another LegendExtractor (typically from new revision)

        Returns:
            {
                "added": [...],    # Entries in other but not in self
                "removed": [...],  # Entries in self but not in other
                "common": [...]    # Entries in both
            }
        """
        self_ids = {e.symbol_id for e in self.entries}
        other_ids = {e.symbol_id for e in other.entries}

        added_ids = other_ids - self_ids
        removed_ids = self_ids - other_ids
        common_ids = self_ids & other_ids

        return {
            "added": [e for e in other.entries if e.symbol_id in added_ids],
            "removed": [e for e in self.entries if e.symbol_id in removed_ids],
            "common": [e for e in self.entries if e.symbol_id in common_ids]
        }


def validate_ocr_against_legend(
    ocr_spans: List[Dict],
    legend_extractor: LegendExtractor,
    fuzzy_threshold: int = 85
) -> List[Dict]:
    """
    Validate OCR spans against extracted legend.

    Adds validation metadata to each span:
    - is_in_legend: bool
    - matched_entry: LegendEntry or None
    - confidence: int (0-100)

    Args:
        ocr_spans: List of OCR results from enhanced_ocr
        legend_extractor: LegendExtractor with loaded entries
        fuzzy_threshold: Fuzzy match threshold (0-100)

    Returns:
        OCR spans with added validation metadata
    """
    from rapidfuzz import fuzz, process

    # Build lookup
    legend_ids = [e.symbol_id for e in legend_extractor.entries]

    for span in ocr_spans:
        text = span["text"]

        # Exact match
        matching_entry = next((e for e in legend_extractor.entries if e.symbol_id == text), None)
        if matching_entry:
            span["legend_validation"] = {
                "is_in_legend": True,
                "matched_entry": matching_entry.symbol_id,
                "category": matching_entry.category,
                "confidence": 100,
                "description": matching_entry.description
            }
            continue

        # Fuzzy match
        if legend_ids:
            match = process.extractOne(text, legend_ids, scorer=fuzz.ratio)
            if match and match[1] >= fuzzy_threshold:
                matching_entry = next((e for e in legend_extractor.entries if e.symbol_id == match[0]), None)
                span["legend_validation"] = {
                    "is_in_legend": True,
                    "matched_entry": match[0],
                    "category": matching_entry.category if matching_entry else "unknown",
                    "confidence": match[1],
                    "description": matching_entry.description if matching_entry else ""
                }
                continue

        # No match
        span["legend_validation"] = {
            "is_in_legend": False,
            "matched_entry": None,
            "category": "unknown",
            "confidence": 0,
            "description": ""
        }

    return ocr_spans


__all__ = [
    "LegendEntry",
    "LegendExtractor",
    "validate_ocr_against_legend",
]
