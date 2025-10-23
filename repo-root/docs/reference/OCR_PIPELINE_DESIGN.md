# OCR Pipeline Enhancement - Design Document

**Version**: 1.0
**Date**: 2025-10-21
**Status**: Proposal / Review Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Proposed Enhancements](#proposed-enhancements)
4. [Visual Debugging System](#visual-debugging-system)
5. [Implementation Phases](#implementation-phases)
6. [API Changes](#api-changes)
7. [Database Schema Changes](#database-schema-changes)
8. [Testing Strategy](#testing-strategy)
9. [Performance Considerations](#performance-considerations)

---

## Executive Summary

### Problems Identified

1. **Lost Confidence Values**: OCR engines return confidence scores (0-100), but they are **discarded** before storage
2. **No Layout Intelligence**: All pages processed identically, regardless of content type (table vs diagram vs text)
3. **No Visual Debugging**: Cannot visualize OCR pipeline stages to understand failures or tune parameters
4. **Inefficient Processing**: OCR runs on pure diagram pages where no text exists

### Proposed Solutions

| Enhancement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| **Confidence Storage** | High (improves diff accuracy) | Low (2-3 hours) | **P0 - Critical** |
| **Visual Debug Pipeline** | High (enables tuning) | Medium (1-2 days) | **P0 - Critical** |
| **Layout Classification** | High (50-70% speedup) | Medium (2-3 days) | **P1 - High** |
| **Adaptive OCR Strategy** | Medium (better accuracy) | Medium (2-3 days) | **P1 - High** |
| **VLM Layout Analyzer** | Medium (highest accuracy) | High (1 week) | **P2 - Medium** |

---

## Current State Analysis

### Data Flow (Current Implementation)

```
PDF Upload
    ↓
pdf_to_vectormap()
    ↓
_extract_text()
    ├─→ Native text extraction
    └─→ OCR (if < 20 native spans)
        ├─→ Tesseract/EasyOCR/Qwen-VL
        ├─→ Returns: {"text": str, "bbox": tuple, "conf": int}
        └─→ ❌ CONFIDENCE DISCARDED HERE
            ↓
TextRun(text, bbox, font, size)  ← No confidence field
    ↓
Database: text_rows table  ← No confidence column
```

### Files Involved

| File | Lines | Current Behavior | Needs Changes? |
|------|-------|------------------|----------------|
| `models.py` | 25-29 | `TextRun` dataclass (no confidence) | ✅ Yes |
| `db_models.py` | 67-87 | `TextRow` table schema | ✅ Yes |
| `pdf_extract.py` | 136-228 | OCR orchestration, discards confidence | ✅ Yes |
| `db_backend.py` | 220-232 | Saves TextRun to DB | ✅ Yes |
| `highres_ocr.py` | 305-346 | OCR engine wrapper, returns confidence | ✅ Minor |

---

## Proposed Enhancements

### Enhancement 1: Confidence Value Storage

#### Goal
Preserve OCR confidence scores throughout the pipeline and store in database.

#### Changes Required

**1.1 Update Data Model** (`pdf_compare/models.py`)

```python
# BEFORE (lines 25-29)
@dataclass(frozen=True)
class TextRun:
    text: str
    bbox: BBox
    font: Optional[str]
    size: Optional[float]

# AFTER
@dataclass(frozen=True)
class TextRun:
    text: str
    bbox: BBox
    font: Optional[str]
    size: Optional[float]
    confidence: Optional[int] = None  # NEW: 0-100 for OCR, None for native text
    source: Optional[str] = None       # NEW: "native", "ocr", "ocr-tesseract", "ocr-easyocr", "ocr-qwen"
```

**1.2 Update Database Schema** (`pdf_compare/db_models.py`)

```python
# MODIFY: class TextRow (lines 67-87)
class TextRow(Base):
    """Text content table with full-text search support."""
    __tablename__ = "text_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    bbox = Column(Text)  # JSON string
    font = Column(Text)
    size = Column(Float)
    source = Column(String, default="native")  # EXISTING
    confidence = Column(Integer, nullable=True)  # NEW: 0-100, NULL for native

    # Relationships
    document = relationship("Document", back_populates="text_rows")

    # Indexes
    __table_args__ = (
        Index("idx_text_rows_doc_page", "doc_id", "page_number"),
        Index("idx_text_rows_source", "source"),
        Index("idx_text_rows_confidence", "confidence"),  # NEW: for filtering low-confidence results
    )
```

**1.3 Database Migration SQL**

```sql
-- Run this migration on existing databases
ALTER TABLE text_rows ADD COLUMN confidence INTEGER;
CREATE INDEX idx_text_rows_confidence ON text_rows(confidence) WHERE confidence IS NOT NULL;

-- Optional: Backfill existing OCR results with estimated confidence
UPDATE text_rows SET confidence = 75 WHERE source = 'ocr' AND confidence IS NULL;
```

**1.4 Update OCR Extraction** (`pdf_compare/pdf_extract.py`, lines 213-220)

```python
# BEFORE
for ocr_text in ocr_results:
    runs.append({
        "text": ocr_text.get("text", ""),
        "bbox": ocr_text.get("bbox", (0, 0, 0, 0)),
        "font": None,
        "size": None,
        "source": "ocr"
    })

# AFTER
for ocr_text in ocr_results:
    runs.append({
        "text": ocr_text.get("text", ""),
        "bbox": ocr_text.get("bbox", (0, 0, 0, 0)),
        "font": None,
        "size": None,
        "source": "ocr",
        "confidence": ocr_text.get("conf") or ocr_text.get("confidence")  # NEW: preserve confidence
    })
```

**1.5 Update Storage Logic** (`pdf_compare/db_backend.py`, lines 220-232)

```python
# BEFORE
text_objs.append(TextRow(
    doc_id=vm.meta.doc_id,
    page_number=pg.page_number,
    text=t.text,
    bbox=bbox_json,
    font=t.font,
    size=t.size,
    source="native"
))

# AFTER
text_objs.append(TextRow(
    doc_id=vm.meta.doc_id,
    page_number=pg.page_number,
    text=t.text,
    bbox=bbox_json,
    font=t.font,
    size=t.size,
    source=t.source or ("native" if t.confidence is None else "ocr"),  # Auto-detect
    confidence=t.confidence  # NEW: store confidence (NULL for native)
))
```

---

### Enhancement 2: Visual Debugging System

#### Goal
Create a visual debugging pipeline that outputs images at each OCR processing stage with confidence overlays.

#### Architecture

```
PDF Page
    ↓
[Stage 1] Original Render (600 DPI)
    ↓ Save: debug/page_001_01_original.png
[Stage 2] Grayscale Conversion
    ↓ Save: debug/page_001_02_grayscale.png
[Stage 3] Preprocessing (bilateral filter, adaptive threshold)
    ↓ Save: debug/page_001_03_preprocessed.png
[Stage 4] OCR Detection (bounding boxes)
    ↓ Save: debug/page_001_04_detections.png
         ↳ Overlay: confidence color-coded (green=90-100, yellow=70-89, red=<70)
[Stage 5] Tile Visualization (if tiled OCR used)
    ↓ Save: debug/page_001_05_tiles.png
         ↳ Overlay: tile grid, skipped tiles (gray), processed tiles (blue)
[Stage 6] Final Results with Confidence
    ↓ Save: debug/page_001_06_final.png
         ↳ Overlay: text bboxes + confidence scores + text content
[Stage 7] Confidence Heatmap
    ↓ Save: debug/page_001_07_heatmap.png
         ↳ Heatmap: color-coded confidence across page
```

#### Implementation

**2.1 Create Debug Utilities Module** (`pdf_compare/debug/ocr_visualizer.py`)

```python
"""
Visual debugging utilities for OCR pipeline.
Outputs images at each processing stage with confidence overlays.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

BBox = Tuple[float, float, float, float]

@dataclass
class OCRDebugConfig:
    """Configuration for OCR debug output."""
    enabled: bool = False
    output_dir: Path = Path("./debug/ocr")
    save_original: bool = True
    save_grayscale: bool = True
    save_preprocessed: bool = True
    save_detections: bool = True
    save_tiles: bool = True
    save_final: bool = True
    save_heatmap: bool = True
    confidence_threshold_low: int = 70   # Red overlay
    confidence_threshold_high: int = 90  # Green overlay
    overlay_alpha: float = 0.3           # Transparency for overlays
    font_scale: float = 0.5              # Text size for labels

    def __post_init__(self):
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)


class OCRVisualizer:
    """Visualizes OCR pipeline stages for debugging and tuning."""

    def __init__(self, config: OCRDebugConfig):
        self.config = config
        self.stage_counter = 0

    def reset(self, page_number: int):
        """Reset counter for new page."""
        self.stage_counter = 0
        self.page_number = page_number

    def _get_output_path(self, stage_name: str, extension: str = ".png") -> Path:
        """Generate output filename."""
        self.stage_counter += 1
        filename = f"page_{self.page_number:03d}_{self.stage_counter:02d}_{stage_name}{extension}"
        return self.config.output_dir / filename

    def save_original(self, image: np.ndarray, dpi: int):
        """Stage 1: Save original rendered image."""
        if not self.config.enabled or not self.config.save_original:
            return

        output_path = self._get_output_path("original")

        # Add metadata text
        img_with_info = image.copy()
        text = f"Original Render | DPI: {dpi} | Size: {image.shape[1]}x{image.shape[0]}px"
        cv2.putText(img_with_info, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imwrite(str(output_path), img_with_info)
        print(f"[DEBUG] Saved: {output_path}")

    def save_grayscale(self, gray: np.ndarray):
        """Stage 2: Save grayscale conversion."""
        if not self.config.enabled or not self.config.save_grayscale:
            return

        output_path = self._get_output_path("grayscale")
        cv2.imwrite(str(output_path), gray)
        print(f"[DEBUG] Saved: {output_path}")

    def save_preprocessed(self, processed: np.ndarray, preprocessing_info: Dict):
        """Stage 3: Save preprocessed image with applied filters."""
        if not self.config.enabled or not self.config.save_preprocessed:
            return

        output_path = self._get_output_path("preprocessed")

        # Add preprocessing info as text overlay
        img_with_info = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        y_offset = 30
        for key, value in preprocessing_info.items():
            text = f"{key}: {value}"
            cv2.putText(img_with_info, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25

        cv2.imwrite(str(output_path), img_with_info)
        print(f"[DEBUG] Saved: {output_path}")

    def save_detections(self, image: np.ndarray, detections: List[Dict]):
        """
        Stage 4: Save OCR detections with confidence-colored bounding boxes.

        Args:
            image: Grayscale or color image
            detections: List of {"text": str, "bbox": (x0,y0,x1,y1), "conf": int}
        """
        if not self.config.enabled or not self.config.save_detections:
            return

        output_path = self._get_output_path("detections")

        # Convert to BGR for colored overlays
        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        # Sort by confidence for rendering order (low conf on top)
        sorted_detections = sorted(detections, key=lambda d: d.get("conf", 0))

        for det in sorted_detections:
            bbox = det["bbox"]
            conf = det.get("conf", 0)
            text = det.get("text", "")

            # Color coding based on confidence
            if conf >= self.config.confidence_threshold_high:
                color = (0, 255, 0)  # Green: high confidence
            elif conf >= self.config.confidence_threshold_low:
                color = (0, 165, 255)  # Orange: medium confidence
            else:
                color = (0, 0, 255)  # Red: low confidence

            # Draw bounding box
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(img_color, (x0, y0), (x1, y1), color, 2)

            # Draw confidence label
            label = f"{conf}%"
            label_bg_color = tuple(int(c * 0.7) for c in color)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                      self.config.font_scale, 1)
            cv2.rectangle(img_color, (x0, y0 - label_h - 5),
                         (x0 + label_w + 5, y0), label_bg_color, -1)
            cv2.putText(img_color, label, (x0 + 2, y0 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (255, 255, 255), 1)

        # Add legend
        legend_y = 30
        cv2.putText(img_color, f"Detections: {len(detections)}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += 30
        cv2.rectangle(img_color, (10, legend_y), (30, legend_y + 20), (0, 255, 0), -1)
        cv2.putText(img_color, f"High (>={self.config.confidence_threshold_high}%)",
                   (35, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(img_color, (10, legend_y), (30, legend_y + 20), (0, 165, 255), -1)
        cv2.putText(img_color, f"Medium (>={self.config.confidence_threshold_low}%)",
                   (35, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        cv2.rectangle(img_color, (10, legend_y), (30, legend_y + 20), (0, 0, 255), -1)
        cv2.putText(img_color, f"Low (<{self.config.confidence_threshold_low}%)",
                   (35, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(output_path), img_color)
        print(f"[DEBUG] Saved: {output_path}")

    def save_tiles(self, image: np.ndarray, tiles: List[Dict], processed_mask: np.ndarray):
        """
        Stage 5: Save tile grid visualization.

        Args:
            image: Full page image
            tiles: List of tile bounds with metadata
            processed_mask: Boolean mask of which tiles were processed
        """
        if not self.config.enabled or not self.config.save_tiles:
            return

        output_path = self._get_output_path("tiles")

        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        # Overlay semi-transparent grid
        overlay = img_color.copy()

        for i, tile in enumerate(tiles):
            x0, y0, x1, y1 = tile["px0"], tile["py0"], tile["px1"], tile["py1"]

            if processed_mask[i]:
                # Processed tile: blue
                color = (255, 100, 0)
                thickness = 2
            else:
                # Skipped tile: gray
                color = (128, 128, 128)
                thickness = 1

            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)

            # Draw tile ID
            tile_id = tile.get("tile_id", f"{i}")
            cv2.putText(overlay, tile_id, (x0 + 5, y0 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Blend overlay
        cv2.addWeighted(overlay, self.config.overlay_alpha, img_color,
                       1 - self.config.overlay_alpha, 0, img_color)

        # Add stats
        total_tiles = len(tiles)
        processed_tiles = int(processed_mask.sum())
        skipped_tiles = total_tiles - processed_tiles

        stats_text = [
            f"Total Tiles: {total_tiles}",
            f"Processed: {processed_tiles}",
            f"Skipped: {skipped_tiles}",
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(img_color, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        cv2.imwrite(str(output_path), img_color)
        print(f"[DEBUG] Saved: {output_path}")

    def save_final_with_text(self, image: np.ndarray, results: List[Dict]):
        """
        Stage 6: Save final results with text content and confidence.

        Args:
            image: Original page image
            results: List of {"text": str, "bbox": tuple, "conf": int}
        """
        if not self.config.enabled or not self.config.save_final:
            return

        output_path = self._get_output_path("final_with_text")

        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        for res in results:
            bbox = res["bbox"]
            conf = res.get("conf", 0)
            text = res.get("text", "")

            # Confidence-based color
            if conf >= self.config.confidence_threshold_high:
                color = (0, 255, 0)
            elif conf >= self.config.confidence_threshold_low:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            x0, y0, x1, y1 = [int(v) for v in bbox]

            # Draw bbox
            cv2.rectangle(img_color, (x0, y0), (x1, y1), color, 1)

            # Draw text content (truncated)
            text_display = text[:20] + "..." if len(text) > 20 else text
            label = f"{text_display} ({conf}%)"

            # Background for readability
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.4, 1)
            cv2.rectangle(img_color, (x0, y1), (x0 + label_w + 5, y1 + label_h + 5),
                         (0, 0, 0), -1)
            cv2.putText(img_color, label, (x0 + 2, y1 + label_h + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(output_path), img_color)
        print(f"[DEBUG] Saved: {output_path}")

    def save_confidence_heatmap(self, page_width: int, page_height: int, results: List[Dict]):
        """
        Stage 7: Generate confidence heatmap across the page.

        Args:
            page_width: Page width in pixels
            page_height: Page height in pixels
            results: List of OCR results with confidence
        """
        if not self.config.enabled or not self.config.save_heatmap:
            return

        output_path = self._get_output_path("confidence_heatmap")

        # Create heatmap grid (lower resolution for performance)
        grid_size = 50
        heatmap = np.zeros((page_height // grid_size + 1, page_width // grid_size + 1))
        counts = np.zeros_like(heatmap)

        # Populate heatmap
        for res in results:
            bbox = res["bbox"]
            conf = res.get("conf", 0)

            x0, y0, x1, y1 = [int(v) for v in bbox]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

            gx, gy = cx // grid_size, cy // grid_size
            if 0 <= gy < heatmap.shape[0] and 0 <= gx < heatmap.shape[1]:
                heatmap[gy, gx] += conf
                counts[gy, gx] += 1

        # Average confidence per grid cell
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(counts > 0, heatmap / counts, 0)

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(12, 16))
        im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_title(f'OCR Confidence Heatmap - Page {self.page_number}', fontsize=16)
        ax.set_xlabel('X Position (grid cells)', fontsize=12)
        ax.set_ylabel('Y Position (grid cells)', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Confidence (%)', fontsize=12)

        # Add statistics
        avg_conf = heatmap[counts > 0].mean() if (counts > 0).any() else 0
        stats_text = f'Avg Confidence: {avg_conf:.1f}%\nTotal Detections: {len(results)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[DEBUG] Saved: {output_path}")

    def generate_summary_report(self, ocr_results: List[Dict], processing_time: float):
        """Generate summary report with confidence statistics."""
        if not self.config.enabled:
            return

        output_path = self._get_output_path("summary", ".txt")

        # Calculate statistics
        confidences = [r.get("conf", 0) for r in ocr_results]

        if confidences:
            avg_conf = np.mean(confidences)
            median_conf = np.median(confidences)
            std_conf = np.std(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)

            high_conf = sum(1 for c in confidences if c >= self.config.confidence_threshold_high)
            medium_conf = sum(1 for c in confidences if self.config.confidence_threshold_low <= c < self.config.confidence_threshold_high)
            low_conf = sum(1 for c in confidences if c < self.config.confidence_threshold_low)
        else:
            avg_conf = median_conf = std_conf = min_conf = max_conf = 0
            high_conf = medium_conf = low_conf = 0

        # Generate report
        report = f"""OCR Processing Summary - Page {self.page_number}
{'=' * 60}

Processing Statistics:
  - Total Detections: {len(ocr_results)}
  - Processing Time: {processing_time:.2f}s
  - Detections/Second: {len(ocr_results) / processing_time if processing_time > 0 else 0:.1f}

Confidence Statistics:
  - Average: {avg_conf:.1f}%
  - Median: {median_conf:.1f}%
  - Std Dev: {std_conf:.1f}%
  - Min: {min_conf}%
  - Max: {max_conf}%

Confidence Distribution:
  - High (>= {self.config.confidence_threshold_high}%): {high_conf} ({high_conf/len(ocr_results)*100 if ocr_results else 0:.1f}%)
  - Medium ({self.config.confidence_threshold_low}-{self.config.confidence_threshold_high-1}%): {medium_conf} ({medium_conf/len(ocr_results)*100 if ocr_results else 0:.1f}%)
  - Low (< {self.config.confidence_threshold_low}%): {low_conf} ({low_conf/len(ocr_results)*100 if ocr_results else 0:.1f}%)

Low Confidence Texts (< {self.config.confidence_threshold_low}%):
"""

        # List low confidence detections
        low_conf_items = [r for r in ocr_results if r.get("conf", 0) < self.config.confidence_threshold_low]
        for item in low_conf_items[:20]:  # Limit to first 20
            report += f"  - [{item.get('conf', 0)}%] \"{item.get('text', '')[:50]}\"\n"

        if len(low_conf_items) > 20:
            report += f"  ... and {len(low_conf_items) - 20} more\n"

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[DEBUG] Saved: {output_path}")


# Convenience function for integration
def create_visualizer(enabled: bool = False, output_dir: str = "./debug/ocr") -> OCRVisualizer:
    """Create an OCRVisualizer with default settings."""
    config = OCRDebugConfig(enabled=enabled, output_dir=Path(output_dir))
    return OCRVisualizer(config)
```

**2.2 Integrate into OCR Pipeline** (`pdf_compare/analyzers/highres_ocr.py`)

```python
# Add at top of file (after imports)
from ..debug.ocr_visualizer import OCRVisualizer, OCRDebugConfig, create_visualizer

# Modify highres_ocr() function (line 305)
def highres_ocr(
    pdf_path: str,
    page_index: int,
    cfg: HighResOCRConfig,
    tiles_pdf: Optional[List[BBox]] = None,
    debug_visualizer: Optional[OCRVisualizer] = None,  # NEW parameter
) -> List[Dict]:
    """
    OCR one page at high DPI with optional visual debugging.
    """
    import time
    start_time = time.time()

    # Initialize debug visualizer if not provided
    if debug_visualizer is None:
        debug_visualizer = create_visualizer(enabled=False)

    debug_visualizer.reset(page_index + 1)

    # Render page
    gray, zoom = _render_page_gray(pdf_path, page_index, cfg.dpi)

    # STAGE 1: Save original
    debug_visualizer.save_original(gray, cfg.dpi)

    # STAGE 2: Grayscale (already done, save it)
    debug_visualizer.save_grayscale(gray)

    # Preprocessing
    preprocessing_info = {
        "engine": cfg.engine,
        "psm": cfg.psm,
        "min_conf": cfg.min_conf,
        "dpi": cfg.dpi
    }

    # For Tesseract, show preprocessing steps
    if cfg.engine == "tesseract":
        proc = cv2.bilateralFilter(gray, 9, 75, 75)
        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2,2), np.uint8)
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

        preprocessing_info.update({
            "bilateral_filter": "9x9, sigma=75",
            "adaptive_threshold": "Gaussian, block=11",
            "morphology": "Close, 2x2 kernel"
        })

        # STAGE 3: Save preprocessed
        debug_visualizer.save_preprocessed(proc, preprocessing_info)

    # Run OCR
    h, w = gray.shape[:2]
    results: List[Dict] = []

    if tiles_pdf:
        # ... existing tile OCR logic ...
        pass
    else:
        # Whole page OCR
        spans = _ocr_tile(gray, cfg)
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            results.append({
                "text": s["text"],
                "bbox": (x0/zoom, y0/zoom, x1/zoom, y1/zoom),
                "conf": s.get("conf", 0)
            })

    # STAGE 4: Save detections with confidence overlay
    debug_visualizer.save_detections(gray, [
        {"text": r["text"], "bbox": [v*zoom for v in r["bbox"]], "conf": r.get("conf", 0)}
        for r in results
    ])

    # STAGE 6: Save final results with text
    debug_visualizer.save_final_with_text(gray, [
        {"text": r["text"], "bbox": [v*zoom for v in r["bbox"]], "conf": r.get("conf", 0)}
        for r in results
    ])

    # STAGE 7: Confidence heatmap
    debug_visualizer.save_confidence_heatmap(w, h, [
        {"bbox": [v*zoom for v in r["bbox"]], "conf": r.get("conf", 0)}
        for r in results
    ])

    # Generate summary report
    processing_time = time.time() - start_time
    debug_visualizer.generate_summary_report(results, processing_time)

    return results
```

**2.3 Add Streamlit UI Controls** (`ui/streamlit_app.py`)

```python
# Add to sidebar configuration section (around line 114)

st.sidebar.subheader("OCR Debug Mode")

enable_ocr_debug = st.sidebar.checkbox(
    "Enable OCR Visual Debugging",
    value=False,
    help="Save visual output at each OCR processing stage for tuning"
)

if enable_ocr_debug:
    debug_output_dir = st.sidebar.text_input(
        "Debug Output Directory",
        value="./debug/ocr",
        help="Directory to save debug images"
    )

    conf_threshold_low = st.sidebar.slider(
        "Low Confidence Threshold",
        min_value=0, max_value=100, value=70,
        help="Text below this confidence will be marked red"
    )

    conf_threshold_high = st.sidebar.slider(
        "High Confidence Threshold",
        min_value=0, max_value=100, value=90,
        help="Text above this confidence will be marked green"
    )

    # Store in session state
    st.session_state["ocr_debug_config"] = {
        "enabled": enable_ocr_debug,
        "output_dir": debug_output_dir,
        "confidence_threshold_low": conf_threshold_low,
        "confidence_threshold_high": conf_threshold_high,
    }
else:
    st.session_state["ocr_debug_config"] = {"enabled": False}

# Add debug output viewer
if enable_ocr_debug:
    with st.expander("🔍 OCR Debug Output Viewer"):
        debug_dir = Path(debug_output_dir)
        if debug_dir.exists():
            pages = sorted([p.name for p in debug_dir.glob("page_*_*_*.png")])
            if pages:
                selected_page = st.selectbox("Select debug image:", pages)
                img_path = debug_dir / selected_page
                st.image(str(img_path), caption=selected_page, use_column_width=True)

                # Show summary report if exists
                summary_path = img_path.parent / selected_page.replace(".png", ".txt")
                if summary_path.exists():
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        st.text(f.read())
            else:
                st.info("No debug images yet. Process a document to generate output.")
        else:
            st.warning(f"Debug directory does not exist: {debug_output_dir}")
```

---

### Enhancement 3: Layout Classification Pipeline

#### Goal
Automatically classify page types (table, diagram, text, technical drawing) to optimize OCR strategy.

#### Implementation

**3.1 Create Layout Classifier** (`pdf_compare/analyzers/layout_classifier.py`)

See full implementation in next section...

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Priority**: P0 - Critical
**Effort**: 3-4 days
**Deliverables**:
1. ✅ Confidence value storage (database migration)
2. ✅ Visual debugging system (OCRVisualizer class)
3. ✅ Streamlit UI integration
4. ✅ Documentation

**Tasks**:
- [ ] Update `models.py`, `db_models.py`, `pdf_extract.py`, `db_backend.py`
- [ ] Create `pdf_compare/debug/` directory
- [ ] Implement `OCRVisualizer` class
- [ ] Add Streamlit debug controls
- [ ] Test on sample PDFs (table, diagram, text)
- [ ] Write tuning guide

**Success Criteria**:
- Confidence values stored in database
- Debug images generated at each stage
- Streamlit UI shows debug output
- Average confidence visible in search results

---

### Phase 2: Layout Intelligence (Week 2)

**Priority**: P1 - High
**Effort**: 4-5 days
**Deliverables**:
1. ✅ Rule-based layout classifier
2. ✅ Adaptive OCR strategy
3. ✅ Performance benchmarks

**Tasks**:
- [ ] Implement `LayoutClassifier` (rule-based)
- [ ] Add page type detection
- [ ] Integrate adaptive OCR routing
- [ ] Benchmark on 100-page corpus
- [ ] Optimize for speed

**Success Criteria**:
- 80%+ accuracy on table/diagram/text classification
- 30-50% speedup on diagram-heavy documents
- OCR skipped on pure diagram pages

---

### Phase 3: VLM Enhancement (Week 3-4)

**Priority**: P2 - Medium
**Effort**: 1 week
**Deliverables**:
1. ✅ VLM-based layout analyzer
2. ✅ Batch inference optimization
3. ✅ Layout caching

**Tasks**:
- [ ] Implement `VLMLayoutAnalyzer`
- [ ] Integrate with vLLM service
- [ ] Add batch processing (8 pages)
- [ ] Cache layout classifications
- [ ] A/B test vs rule-based

**Success Criteria**:
- 95%+ accuracy on layout classification
- <500ms per page (with batching)
- Graceful fallback to rule-based

---

## API Changes

### New Functions

```python
# pdf_compare/debug/ocr_visualizer.py
create_visualizer(enabled: bool, output_dir: str) -> OCRVisualizer

class OCRVisualizer:
    def save_original(image, dpi)
    def save_grayscale(gray)
    def save_preprocessed(processed, info)
    def save_detections(image, detections)
    def save_tiles(image, tiles, mask)
    def save_final_with_text(image, results)
    def save_confidence_heatmap(width, height, results)
    def generate_summary_report(results, time)

# pdf_compare/analyzers/highres_ocr.py
highres_ocr(..., debug_visualizer: Optional[OCRVisualizer] = None)

# pdf_compare/analyzers/layout_classifier.py
classify_page_layout(page: PageVectors) -> LayoutClassification
```

### Modified Function Signatures

```python
# BEFORE
def highres_ocr(pdf_path, page_index, cfg, tiles_pdf=None) -> List[Dict]

# AFTER
def highres_ocr(pdf_path, page_index, cfg, tiles_pdf=None,
                debug_visualizer=None) -> List[Dict]
```

---

## Database Schema Changes

### Migration Script

```sql
-- File: migrations/001_add_confidence.sql

-- Add confidence column to text_rows
ALTER TABLE text_rows ADD COLUMN confidence INTEGER;

-- Add index for filtering by confidence
CREATE INDEX idx_text_rows_confidence ON text_rows(confidence)
WHERE confidence IS NOT NULL;

-- Optional: Add layout_type to pages table (for Phase 2)
ALTER TABLE pages ADD COLUMN layout_type VARCHAR(50);
ALTER TABLE pages ADD COLUMN layout_confidence FLOAT;

-- Add index for layout filtering
CREATE INDEX idx_pages_layout_type ON pages(layout_type);
```

### Rollback Script

```sql
-- File: migrations/001_add_confidence_rollback.sql

DROP INDEX IF EXISTS idx_text_rows_confidence;
ALTER TABLE text_rows DROP COLUMN confidence;

DROP INDEX IF EXISTS idx_pages_layout_type;
ALTER TABLE pages DROP COLUMN layout_confidence;
ALTER TABLE pages DROP COLUMN layout_type;
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_ocr_visualizer.py

def test_visualizer_creates_output_files():
    """Test that OCRVisualizer creates expected output files."""
    config = OCRDebugConfig(enabled=True, output_dir=Path("./test_output"))
    viz = OCRVisualizer(config)
    viz.reset(page_number=1)

    # Create dummy data
    image = np.zeros((1000, 1000), dtype=np.uint8)
    detections = [
        {"text": "Test", "bbox": (10, 10, 100, 50), "conf": 95},
        {"text": "Low", "bbox": (10, 60, 100, 100), "conf": 45},
    ]

    # Run visualization
    viz.save_detections(image, detections)

    # Assert files exist
    assert (config.output_dir / "page_001_01_detections.png").exists()

    # Cleanup
    shutil.rmtree(config.output_dir)

def test_confidence_storage_roundtrip():
    """Test that confidence values survive database roundtrip."""
    # Create TextRun with confidence
    text_run = TextRun(text="Test", bbox=(0, 0, 100, 50),
                       font=None, size=None, confidence=85)

    # Store in database
    vm = VectorMap(...)
    db.upsert_vectormap(vm)

    # Retrieve from database
    retrieved = db.get_document_text_with_coords(doc_id, page=1)

    # Assert confidence preserved
    assert retrieved[0]["confidence"] == 85
```

### Integration Tests

```python
# tests/test_ocr_pipeline_debug.py

def test_full_pipeline_with_debug():
    """Test complete OCR pipeline with debug output enabled."""
    config = OCRDebugConfig(enabled=True, output_dir=Path("./test_debug"))
    visualizer = OCRVisualizer(config)

    # Run OCR on test PDF
    results = highres_ocr("tests/fixtures/sample.pdf", 0,
                          HighResOCRConfig(dpi=400),
                          debug_visualizer=visualizer)

    # Assert all debug stages created files
    assert (config.output_dir / "page_001_01_original.png").exists()
    assert (config.output_dir / "page_001_02_grayscale.png").exists()
    assert (config.output_dir / "page_001_03_preprocessed.png").exists()
    assert (config.output_dir / "page_001_04_detections.png").exists()
    assert (config.output_dir / "page_001_06_final_with_text.png").exists()
    assert (config.output_dir / "page_001_07_confidence_heatmap.png").exists()
    assert (config.output_dir / "page_001_08_summary.txt").exists()
```

---

## Performance Considerations

### Debug Mode Overhead

| Operation | Normal Mode | Debug Mode | Overhead |
|-----------|-------------|------------|----------|
| OCR Processing | 2.5s/page | 2.5s/page | 0% |
| Image Saving | 0s | 0.3s | +12% |
| Heatmap Generation | 0s | 0.1s | +4% |
| **Total** | **2.5s/page** | **2.9s/page** | **+16%** |

**Recommendation**: Enable debug mode only for tuning sessions, not production ingestion.

### Storage Requirements

- Original image (600 DPI): ~5-10 MB
- Grayscale: ~2-4 MB
- Preprocessed: ~2-4 MB
- Detections overlay: ~5-10 MB
- Heatmap: ~1 MB
- **Total per page**: ~15-30 MB

**For 100-page document**: ~1.5-3 GB debug output

**Recommendation**: Implement automatic cleanup (keep last 10 pages, or clean after N days).

---

## Tuning Guide

### How to Use Debug Output

#### Step 1: Enable Debug Mode
1. Open Streamlit UI
2. Sidebar → "Enable OCR Visual Debugging"
3. Set confidence thresholds (default: 70/90)
4. Ingest a test document

#### Step 2: Review Debug Images

**Check Original Render** (`page_XXX_01_original.png`):
- Is DPI sufficient? (text should be crisp, not blurry)
- If blurry → Increase DPI (e.g., 400 → 600)

**Check Preprocessing** (`page_XXX_03_preprocessed.png`):
- Is text clearly separated from background?
- If too much noise → Adjust bilateral filter parameters
- If text merged → Reduce morphology kernel size

**Check Detections** (`page_XXX_04_detections.png`):
- Are bboxes accurate?
- Many red boxes (low confidence)?
  - Check preprocessing settings
  - Try different OCR engine (Tesseract vs EasyOCR)
  - Increase DPI

**Check Confidence Heatmap** (`page_XXX_07_heatmap.png`):
- Are there "cold zones" (blue/red areas)?
  - These indicate problematic regions
  - Zoom into those areas in original PDF
  - Adjust preprocessing for that content type

#### Step 3: Read Summary Report

```
Confidence Distribution:
  - High (>= 90%): 150 (75%)      ← GOOD: Most text high confidence
  - Medium (70-89%): 30 (15%)     ← OK: Review these manually
  - Low (< 70%): 20 (10%)         ← BAD: Check low confidence list

Low Confidence Texts:
  - [45%] "l0.5"                  ← Likely "10.5" misread
  - [52%] "V-l0l"                 ← Likely "V-101" misread
```

#### Step 4: Adjust Settings

**If low confidence on thin text**:
- Increase DPI: 400 → 600
- Enable dual-PSM mode (Tesseract)
- Try EasyOCR (better on small text)

**If low confidence on tables**:
- Use table-focused OCR strategy
- Increase preprocessing threshold
- Enable grid detection

**If low confidence on diagrams**:
- Enable callout detection
- Use sparse OCR mode
- Consider Qwen-VL for technical drawings

---

## Next Steps

### Review & Approval

**Please review the following sections**:

1. **Enhancement 1 (Confidence Storage)**:
   - [ ] Approve data model changes
   - [ ] Approve database schema
   - [ ] Approve migration strategy

2. **Enhancement 2 (Visual Debugging)**:
   - [ ] Approve debug output stages
   - [ ] Approve Streamlit UI integration
   - [ ] Approve storage requirements (15-30 MB/page)

3. **Implementation Priority**:
   - [ ] Confirm Phase 1 scope (confidence + debug)
   - [ ] Defer Phase 2 (layout classification)?
   - [ ] Defer Phase 3 (VLM integration)?

### Questions to Resolve

1. **Database Migration**: Should we backfill existing OCR results with estimated confidence (e.g., 75%)?
2. **Debug Storage**: Auto-cleanup strategy? (keep last N pages? delete after 7 days?)
3. **Confidence Thresholds**: Are 70/90 good defaults, or should we tune per use case?
4. **OCR Engine Priority**: Should we default to EasyOCR (better accuracy) or Tesseract (faster)?

---

## Appendix

### File Checklist

**New Files**:
- [ ] `pdf_compare/debug/__init__.py`
- [ ] `pdf_compare/debug/ocr_visualizer.py`
- [ ] `migrations/001_add_confidence.sql`
- [ ] `migrations/001_add_confidence_rollback.sql`
- [ ] `tests/test_ocr_visualizer.py`
- [ ] `tests/test_confidence_storage.py`

**Modified Files**:
- [ ] `pdf_compare/models.py` (add confidence to TextRun)
- [ ] `pdf_compare/db_models.py` (add confidence column)
- [ ] `pdf_compare/pdf_extract.py` (preserve confidence)
- [ ] `pdf_compare/db_backend.py` (store confidence)
- [ ] `pdf_compare/analyzers/highres_ocr.py` (integrate visualizer)
- [ ] `ui/streamlit_app.py` (add debug controls)

**Documentation**:
- [ ] `docs/OCR_TUNING_GUIDE.md`
- [ ] `docs/VISUAL_DEBUG_GUIDE.md`
- [ ] `README.md` (add debug mode section)

---

**End of Design Document**

*Ready for review and implementation. Please provide feedback on scope, priority, and any concerns.*
