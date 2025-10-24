"""
Shared visualization utilities for OCR debugging.

Centralizes border/box drawing logic to avoid code duplication.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2


# Color palette for different region types
REGION_COLORS = {
    "table": (0, 255, 0),      # Green
    "diagram": (255, 0, 0),    # Blue
    "text": (0, 165, 255),     # Orange
    "mixed": (255, 255, 0),    # Cyan
    "tile": (255, 0, 255),     # Magenta
    "processed": (0, 255, 0),  # Green
    "skipped": (128, 128, 128) # Gray
}


def draw_bbox_with_label(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.7,
    draw_filled_label: bool = True
) -> np.ndarray:
    """
    Draw a bounding box with label on an image.

    Args:
        image: Image to draw on (will be modified in-place)
        bbox: (x1, y1, x2, y2) coordinates
        label: Text label to display
        color: RGB color tuple
        thickness: Line thickness
        font_scale: Font size
        draw_filled_label: If True, draw filled background for label

    Returns:
        Modified image (same as input)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Calculate label size
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Ensure label fits within image bounds
    label_y1 = max(y1 - label_h - baseline - 5, 0)
    label_y2 = label_y1 + label_h + baseline + 5
    label_x2 = min(x1 + label_w + 5, image.shape[1])

    if draw_filled_label:
        # Draw filled background for label
        cv2.rectangle(
            image,
            (x1, label_y1),
            (label_x2, label_y2),
            color,
            -1  # Filled
        )

        # Draw label text in white
        cv2.putText(
            image, label, (x1 + 2, label_y2 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
    else:
        # Draw label text in color directly
        cv2.putText(
            image, label, (x1 + 2, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )

    return image


def draw_tile_grid(
    image: np.ndarray,
    tiles: List[dict],
    show_labels: bool = True,
    show_processed_only: bool = False
) -> np.ndarray:
    """
    Draw tile borders on an image.

    Args:
        image: Image to draw on (BGR or grayscale)
        tiles: List of tile dicts with keys:
               - bbox: (x1, y1, x2, y2) in pixels
               - tile_id: Optional tile identifier
               - has_content: Optional flag (True if processed, False if skipped)
        show_labels: Show tile IDs as labels
        show_processed_only: Only draw tiles that were processed

    Returns:
        Image with tile grid overlay
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for tile in tiles:
        # Check if tile was processed
        has_content = tile.get("has_content", True)
        if show_processed_only and not has_content:
            continue

        bbox = tile.get("bbox")
        if bbox is None:
            continue

        # Choose color based on processing status
        color = REGION_COLORS["processed"] if has_content else REGION_COLORS["skipped"]

        # Draw border
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Optionally draw label
        if show_labels:
            tile_id = tile.get("tile_id", "?")
            status = "✓" if has_content else "✗"
            label = f"Tile {tile_id} {status}"

            draw_bbox_with_label(
                vis, bbox, label, color,
                thickness=1, font_scale=0.5, draw_filled_label=True
            )

    return vis


def draw_region_borders(
    image: np.ndarray,
    regions: List[dict],
    show_labels: bool = True,
    show_legend: bool = True
) -> np.ndarray:
    """
    Draw layout region borders (tables, diagrams, text).

    Args:
        image: Image to draw on (BGR or grayscale)
        regions: List of region dicts/objects with:
                 - bbox: (x1, y1, x2, y2)
                 - region_type: "table", "diagram", "text", "mixed"
                 - confidence: Optional confidence score (0-100)
        show_labels: Show region type and confidence
        show_legend: Draw color legend

    Returns:
        Image with region borders
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw each region
    for region in regions:
        # Handle both dict and object formats
        if hasattr(region, 'bbox'):
            bbox = region.bbox
            region_type = region.region_type
            confidence = getattr(region, 'confidence', None)
        else:
            bbox = region.get("bbox")
            region_type = region.get("region_type", "mixed")
            confidence = region.get("confidence")

        if bbox is None:
            continue

        # Get color for region type
        color = REGION_COLORS.get(region_type, (128, 128, 128))

        # Create label
        if confidence is not None:
            label = f"{region_type} {confidence:.0f}%"
        else:
            label = region_type

        # Draw border with label
        draw_bbox_with_label(
            vis, bbox, label, color,
            thickness=3, font_scale=0.8, draw_filled_label=True
        )

    # Draw legend
    if show_legend:
        legend_y = 30
        legend_x = 10
        legend_width = 350
        legend_height = 95

        # White background
        cv2.rectangle(
            vis,
            (legend_x - 5, 5),
            (legend_x + legend_width, legend_y + legend_height),
            (255, 255, 255),
            -1  # Filled
        )

        # Black border
        cv2.rectangle(
            vis,
            (legend_x - 5, 5),
            (legend_x + legend_width, legend_y + legend_height),
            (0, 0, 0),
            2
        )

        # Title
        cv2.putText(
            vis, "Layout Regions:", (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        # Legend entries
        legend_y += 25
        cv2.putText(
            vis, "Green = Table | Blue = Diagram", (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        legend_y += 20
        cv2.putText(
            vis, "Orange = Text | Cyan = Mixed", (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        legend_y += 20
        cv2.putText(
            vis, f"Total Regions: {len(regions)}", (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    return vis


def draw_tile_grid_with_regions(
    image: np.ndarray,
    tiles: List[dict],
    regions: List[dict],
    show_tile_labels: bool = False,
    show_region_labels: bool = True
) -> np.ndarray:
    """
    Draw both tile grid and layout regions on the same image.

    Useful for showing how tiles overlap with detected regions.

    Args:
        image: Image to draw on
        tiles: List of tile dicts
        regions: List of region dicts
        show_tile_labels: Show tile IDs
        show_region_labels: Show region types

    Returns:
        Image with both tile and region overlays
    """
    # First draw regions (thicker lines, more prominent)
    vis = draw_region_borders(image, regions, show_labels=show_region_labels, show_legend=True)

    # Then draw tiles (thinner lines, less prominent)
    for tile in tiles:
        bbox = tile.get("bbox")
        if bbox is None:
            continue

        has_content = tile.get("has_content", True)
        color = REGION_COLORS["tile"]  # Magenta for tiles

        # Draw thin border
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Optionally show tile ID in corner
        if show_tile_labels:
            tile_id = tile.get("tile_id", "?")
            status = "✓" if has_content else "✗"
            label = f"T{tile_id}{status}"

            # Small label in top-left corner
            cv2.putText(
                vis, label, (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    return vis
