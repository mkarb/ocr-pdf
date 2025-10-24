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

# Import shared visualization utilities
from .visualization_utils import (
    draw_bbox_with_label,
    draw_region_borders,
    draw_tile_grid,
    draw_tile_grid_with_regions,
)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

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
        self.page_number = 1

        # Create output directory if it doesn't exist
        if self.config.enabled:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, page_number: int):
        """Reset counter for new page."""
        self.stage_counter = 0
        self.page_number = page_number

    def _get_output_path(self, stage_name: str, extension: str = ".png") -> Path:
        """Generate output filename."""
        self.stage_counter += 1
        filename = f"page_{self.page_number:03d}_{self.stage_counter:02d}_{stage_name}{extension}"
        return self.config.output_dir / filename

    def _write_image(self, output_path: Path, image: np.ndarray, description: str):
        """Write an image to disk with consistent logging."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), image)
            if success:
                print(f"[OCR DEBUG] Saved {description}: {output_path}", flush=True)
            else:
                print(f"[OCR DEBUG] Failed to save {description}: {output_path}", flush=True)
        except Exception as exc:
            print(f"[OCR DEBUG] Error saving {description}: {output_path} ({exc})", flush=True)

    def save_original(self, image: np.ndarray, dpi: int):
        """Stage 1: Save original rendered image."""
        if not self.config.enabled or not self.config.save_original:
            return

        output_path = self._get_output_path("original")

        # Convert grayscale to BGR for colored text overlay
        if len(image.shape) == 2:
            img_with_info = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_with_info = image.copy()

        # Add metadata text
        text = f"Original Render | DPI: {dpi} | Size: {image.shape[1]}x{image.shape[0]}px"
        cv2.putText(img_with_info, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self._write_image(output_path, img_with_info, "original render")

    def save_grayscale(self, gray: np.ndarray):
        """Stage 2: Save grayscale conversion."""
        if not self.config.enabled or not self.config.save_grayscale:
            return

        output_path = self._get_output_path("grayscale")
        self._write_image(output_path, gray, "grayscale render")

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

        self._write_image(output_path, img_with_info, "preprocessed render")

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

            # Color coding based on confidence
            if conf >= self.config.confidence_threshold_high:
                color = (0, 255, 0)  # Green: high confidence
            elif conf >= self.config.confidence_threshold_low:
                color = (0, 165, 255)  # Orange: medium confidence
            else:
                color = (0, 0, 255)  # Red: low confidence

            draw_bbox_with_label(
                img_color,
                bbox,
                f"{conf}%",
                color,
                thickness=2,
                font_scale=self.config.font_scale,
                draw_filled_label=True,
            )

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

        self._write_image(output_path, img_color, "detections overlay")

    def save_tiles(self, image: np.ndarray, tiles: List[Dict], processed_mask: np.ndarray):
        """
        Stage 5: Save tile grid visualization with borders.

        Args:
            image: Full page image
            tiles: List of tile bounds with metadata (can have either 'bbox' or 'px0/py0/px1/py1')
            processed_mask: Boolean mask of which tiles were processed
        """
        if not self.config.enabled or not self.config.save_tiles:
            return

        output_path = self._get_output_path("tiles")

        # Convert tile data to format expected by visualization utility
        tile_vis_data = []
        for i, tile in enumerate(tiles):
            # Handle both bbox format and px0/py0/px1/py1 format
            if "bbox" in tile:
                bbox = tile["bbox"]
            else:
                bbox = (tile["px0"], tile["py0"], tile["px1"], tile["py1"])

            tile_vis_data.append({
                "bbox": bbox,
                "tile_id": tile.get("tile_id", f"{i}"),
                "has_content": bool(processed_mask[i])
            })

        # Use shared visualization utility
        img_with_tiles = draw_tile_grid(
            image, tile_vis_data,
            show_labels=True,
            show_processed_only=False
        )

        # Add stats overlay
        total_tiles = len(tiles)
        processed_tiles = int(processed_mask.sum())
        skipped_tiles = total_tiles - processed_tiles

        stats_text = [
            f"Total Tiles: {total_tiles}",
            f"Processed: {processed_tiles} (green)",
            f"Skipped: {skipped_tiles} (gray)",
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(img_with_tiles, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        self._write_image(output_path, img_with_tiles, "tile grid with borders")

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

        self._write_image(output_path, img_color, "final text overlay")

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

        if not HAVE_MATPLOTLIB:
            print("[OCR DEBUG] Skipping heatmap: matplotlib not available")
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

        print(f"[OCR DEBUG] Saved confidence heatmap: {output_path}")

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
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[OCR DEBUG] Saved: {output_path}", flush=True)
        except Exception as e:
            print(f"[OCR DEBUG] Error saving {output_path}: {e}", flush=True)

    def save_layout_regions(self, image: np.ndarray, regions: List[Dict]):
        """
        Save detected layout regions (tables, diagrams, text blocks).

        Args:
            image: Grayscale or color image
            regions: List of LayoutRegion objects or dicts with keys:
                     {"bbox": (x1,y1,x2,y2), "region_type": str, "confidence": float, "metadata": dict}
        """
        if not self.config.enabled:
            return

        output_path = self._get_output_path("layout_regions")

        # Use shared visualization utility
        img_with_regions = draw_region_borders(
            image, regions,
            show_labels=True,
            show_legend=True
        )

        self._write_image(output_path, img_with_regions, "layout regions")


# Convenience function for integration
def create_visualizer(enabled: bool = False, output_dir: str = "./debug/ocr",
                      conf_low: int = 70, conf_high: int = 90) -> OCRVisualizer:
    """Create an OCRVisualizer with specified settings."""
    config = OCRDebugConfig(
        enabled=enabled,
        output_dir=Path(output_dir),
        confidence_threshold_low=conf_low,
        confidence_threshold_high=conf_high
    )
    return OCRVisualizer(config)
