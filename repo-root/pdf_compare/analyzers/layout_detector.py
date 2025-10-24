"""
Layout detection using only OpenCV and NumPy (no additional dependencies).

Detects tables, diagrams, and text regions in document images.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal
import cv2
import numpy as np


@dataclass
class LayoutRegion:
    """A detected region in the page layout."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    region_type: Literal["table", "diagram", "text", "mixed"]
    confidence: float  # 0-100
    metadata: dict


class LayoutDetector:
    """
    Detect and classify page regions using OpenCV.

    Uses line detection for tables, component analysis for diagrams,
    and morphological operations for text blocks.
    """

    def __init__(
        self,
        min_table_size: int = 100,
        min_diagram_size: int = 200,
        min_text_width: int = 100,
        line_length_threshold: int = 40,
        max_region_pct: float = 0.7,  # Maximum region size as percentage of page (0.7 = 70%)
        debug: bool = False
    ):
        """
        Initialize layout detector.

        Args:
            min_table_size: Minimum width/height for table detection
            min_diagram_size: Minimum width/height for diagram detection
            min_text_width: Minimum width for text block detection
            line_length_threshold: Minimum line length for table detection
            max_region_pct: Maximum region size as percentage of page dimensions
            debug: Enable debug output
        """
        self.min_table_size = min_table_size
        self.min_diagram_size = min_diagram_size
        self.min_text_width = min_text_width
        self.line_length_threshold = line_length_threshold
        self.max_region_pct = max_region_pct
        self.debug = debug

    def detect_tables(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect tables using horizontal and vertical line detection.

        Tables have grid structures with intersecting horizontal/vertical lines.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_length_threshold, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.line_length_threshold))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine lines to form table grid
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

        # Dilate to connect nearby lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_mask = cv2.dilate(table_mask, dilate_kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get image dimensions for maximum size filtering
        img_height, img_width = gray.shape
        max_width = int(img_width * self.max_region_pct)
        max_height = int(img_height * self.max_region_pct)

        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (both minimum and maximum)
            if w < self.min_table_size or h < self.min_table_size:
                continue

            # Skip regions that are too large (likely false positives encompassing the whole page)
            if w > max_width or h > max_height:
                if self.debug:
                    print(f"[LayoutDetector] Skipping oversized table region: {w}x{h} (max: {max_width}x{max_height})")
                continue

            # Validate grid structure using helper method
            roi = table_mask[y:y+h, x:x+w]
            roi_h_lines = horizontal_lines[y:y+h, x:x+w]
            roi_v_lines = vertical_lines[y:y+h, x:x+w]

            is_valid, metrics = self._validate_table_grid(roi, roi_h_lines, roi_v_lines, w, h)

            if not is_valid:
                continue  # Skip regions that don't have proper table grid structure

            # Calculate confidence based on grid quality
            grid_quality = min(metrics["estimated_cells"] / 20.0, 1.0)
            density_score = min(metrics["line_density"] * 500, 30)
            intersection_score = min(metrics["intersection_density"] * 10000, 20)
            confidence = min(95, 50 + density_score + intersection_score + grid_quality * 25)

            metrics["method"] = "line_detection"
            table_regions.append(LayoutRegion(
                bbox=(x, y, x+w, y+h),
                region_type="table",
                confidence=confidence,
                metadata=metrics
            ))

        if self.debug:
            print(f"[LayoutDetector] Detected {len(table_regions)} tables")

        return table_regions

    def detect_diagrams(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect diagrams using edge detection and component analysis.

        Diagrams have complex shapes with many edges but relatively little text.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate edges to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_height, img_width = gray.shape
        max_width = int(img_width * self.max_region_pct)
        max_height = int(img_height * self.max_region_pct)
        max_area = img_width * img_height * self.max_region_pct

        diagram_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if w < self.min_diagram_size or h < self.min_diagram_size:
                continue

            if w > max_width or h > max_height:
                if self.debug:
                    print("LayoutDetector skipping oversized diagram region %dx%d (max %dx%d)" % (w, h, max_width, max_height))
                continue

            area = cv2.contourArea(contour)
            if area <= 0 or area > max_area:
                if self.debug and area > max_area:
                    print("LayoutDetector skipping high-area diagram region %.0f > %.0f" % (area, max_area))
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # Diagrams have complex shapes (high perimeter/area ratio)
            complexity = perimeter / np.sqrt(area) if area > 0 else 0

            # Filter: diagrams are complex and irregular
            if complexity > 4 and area > 1000:
                # Calculate edge density in region
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / (w * h)

                confidence = min(90, 50 + complexity * 5)

                diagram_regions.append(LayoutRegion(
                    bbox=(x, y, x + w, y + h),
                    region_type="diagram",
                    confidence=confidence,
                    metadata={
                        "method": "edge_detection",
                        "complexity": float(complexity),
                        "edge_density": float(edge_density),
                        "area": float(area)
                    }
                ))

        if self.debug:
            print(f"[LayoutDetector] Detected {len(diagram_regions)} diagrams")

        return diagram_regions

    def detect_text_blocks(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect text blocks using morphological operations.

        Text has horizontal word patterns that can be connected.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal kernel to connect words in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_blocks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Text blocks are typically wide
            if w < self.min_text_width or h < 10:
                continue

            aspect_ratio = w / float(h)
            area = w * h
            min_area = self.min_text_width * 20

            # Accept either wide single-line text or taller paragraph blocks
            qualifies = aspect_ratio >= 1.3 or (area >= min_area and h >= 20)
            if qualifies:
                # Derive confidence from shape/area score
                area_score = min(area / max(min_area, 1), 3.0)
                shape_score = max(aspect_ratio, 1.3 + area_score)
                confidence = min(85, 55 + shape_score * 5)

                text_blocks.append(LayoutRegion(
                    bbox=(x, y, x+w, y+h),
                    region_type="text",
                    confidence=confidence,
                    metadata={
                        "method": "morphology",
                        "aspect_ratio": float(aspect_ratio),
                        "width": w,
                        "height": h,
                        "area": area
                    }
                ))

        if self.debug:
            print(f"[LayoutDetector] Detected {len(text_blocks)} text blocks")

        return text_blocks

    def analyze_page(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect all regions in the page and classify them.

        Returns regions sorted by reading order (top to bottom).
        """
        regions = []

        # 1. Detect tables (highest priority)
        table_regions = self.detect_tables(image)
        regions.extend(table_regions)

        # 2. Detect diagrams (exclude areas already marked as tables)
        diagram_regions = self.detect_diagrams(image)
        for diagram in diagram_regions:
            # Check if overlaps with table
            overlaps_table = any(
                self._iou(diagram.bbox, table.bbox) > 0.5
                for table in table_regions
            )
            if not overlaps_table:
                regions.append(diagram)

        # 3. Detect text blocks (exclude areas already classified)
        text_regions = self.detect_text_blocks(image)
        for text in text_regions:
            # Check if already classified as table/diagram
            overlaps_other = any(
                self._iou(text.bbox, r.bbox) > 0.3
                for r in regions
            )
            if not overlaps_other:
                regions.append(text)

        # Sort by reading order (top to bottom, left to right)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        if self.debug:
            print(f"[LayoutDetector] Total regions: {len(regions)}")
            for i, region in enumerate(regions):
                print(f"  {i+1}. {region.region_type} @ {region.bbox} (conf={region.confidence:.1f})")

        return regions

    def _validate_table_grid(
        self,
        roi: np.ndarray,
        h_lines: np.ndarray,
        v_lines: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[bool, dict]:
        """
        Validate if a region has proper table grid structure.

        Returns:
            (is_valid, metrics_dict)
        """
        # Count distinct horizontal and vertical lines via projection
        h_projection = np.sum(h_lines, axis=1) > 0
        v_projection = np.sum(v_lines, axis=0) > 0
        h_line_count = np.sum(np.diff(h_projection.astype(int)) != 0) // 2
        v_line_count = np.sum(np.diff(v_projection.astype(int)) != 0) // 2

        # Count grid intersections
        intersections = cv2.bitwise_and(h_lines, v_lines)
        intersection_count = np.sum(intersections > 0)
        intersection_density = intersection_count / (width * height)

        # Estimate cell count (grid cells formed by lines)
        estimated_cells = max(h_line_count - 1, 0) * max(v_line_count - 1, 0)

        # Line density
        line_density = np.sum(roi > 0) / (width * height)

        # Validation criteria
        min_cells = 4
        min_intersection_density = 0.001

        is_valid = (
            estimated_cells >= min_cells and
            intersection_density >= min_intersection_density
        )

        metrics = {
            "line_density": float(line_density),
            "h_line_count": int(h_line_count),
            "v_line_count": int(v_line_count),
            "estimated_cells": int(estimated_cells),
            "intersection_density": float(intersection_density),
            "intersection_count": int(intersection_count)
        }

        if self.debug and not is_valid:
            if estimated_cells < min_cells:
                print(f"[LayoutDetector] Rejected: {h_line_count}h × {v_line_count}v = {estimated_cells} cells < {min_cells}")
            elif intersection_density < min_intersection_density:
                print(f"[LayoutDetector] Rejected: intersection density {intersection_density:.4f} < {min_intersection_density}")

        return is_valid, metrics

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (xi_max - xi_min) * (yi_max - yi_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
