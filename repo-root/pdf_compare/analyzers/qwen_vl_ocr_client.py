"""
Qwen2-VL OCR Client - Uses vLLM Microservice

Lightweight client that connects to the vLLM microservice for high-accuracy OCR.
This runs in the Streamlit UI container and sends images to the vLLM service.

Architecture:
    This Code (UI Container) → HTTP Request → vLLM Service (GPU Container)

Usage:
    from pdf_compare.analyzers import get_qwen_vl_ocr_client

    # Get client
    ocr = get_qwen_vl_ocr_client()

    # Check if service available
    if ocr.is_available():
        # Use vLLM OCR (high accuracy)
        results = ocr.extract_text_from_image(image)
    else:
        # Fallback to EasyOCR/Tesseract
        results = easyocr_fallback(image)
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Import vLLM client
try:
    from ..vllm_client import get_vllm_client, VLLMServiceUnavailable
    HAVE_CLIENT = True
except ImportError:
    HAVE_CLIENT = False
    logger.warning("vLLM client not available")


class QwenVLOCRClient:
    """
    Client for Qwen2-VL OCR via vLLM microservice.

    This is a LIGHTWEIGHT wrapper - no models loaded here!
    All inference happens in the separate vLLM service container.
    """

    def __init__(self, vllm_service_url: Optional[str] = None):
        """
        Initialize OCR client.

        Args:
            vllm_service_url: URL of vLLM service (default: from env)
        """
        if not HAVE_CLIENT:
            raise ImportError("vLLM client not available")

        self.client = get_vllm_client(base_url=vllm_service_url)
        logger.info(f"Qwen2-VL OCR client initialized: {self.client.base_url}")

    def is_available(self) -> bool:
        """Check if vLLM service is available and ready."""
        return self.client.is_available()

    def extract_text_from_image(
        self,
        image: np.ndarray,
        focus_technical: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from image using vLLM service.

        Args:
            image: Input image as numpy array (grayscale or RGB)
            focus_technical: Optimize for engineering documents

        Returns:
            List of dicts with keys:
                - text: Extracted text
                - bbox: (x0, y0, x1, y1) in pixel coordinates
                - confidence: Confidence score (0.0-1.0)
                - source: "qwen-vl"

        Raises:
            VLLMServiceUnavailable: If service not available
        """
        image_h, image_w = image.shape[:2]

        # Call vLLM service
        results = self.client.ocr_image_to_pdf_coords(
            image=image,
            image_width=image_w,
            image_height=image_h,
            focus_technical=focus_technical,
            min_confidence=0.5
        )

        # Add source tag
        for item in results:
            item["source"] = "qwen-vl"

        logger.info(f"Extracted {len(results)} text items via vLLM service")
        return results

    def extract_text_from_tile(
        self,
        tile_image: np.ndarray,
        tile_bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from tile and map to PDF coordinates.

        Compatible with existing tiled_ocr() infrastructure.

        Args:
            tile_image: Tile image
            tile_bbox: Tile position in PDF coords (x0, y0, x1, y1)
            page_width: Page width in PDF points
            page_height: Page height in PDF points

        Returns:
            OCR results with bboxes in PDF coordinates
        """
        # Extract from tile
        tile_results = self.extract_text_from_image(
            tile_image,
            focus_technical=True
        )

        # Map tile-local coords to PDF coords
        tile_x0, tile_y0, tile_x1, tile_y1 = tile_bbox
        tile_h, tile_w = tile_image.shape[:2]

        mapped_results = []
        for item in tile_results:
            tx0, ty0, tx1, ty1 = item["bbox"]

            # Map to PDF coordinates
            pdf_x0 = tile_x0 + (tx0 / tile_w) * (tile_x1 - tile_x0)
            pdf_y0 = tile_y0 + (ty0 / tile_h) * (tile_y1 - tile_y0)
            pdf_x1 = tile_x0 + (tx1 / tile_w) * (tile_x1 - tile_x0)
            pdf_y1 = tile_y0 + (ty1 / tile_h) * (tile_y1 - tile_y0)

            mapped_results.append({
                "text": item["text"],
                "bbox": (pdf_x0, pdf_y0, pdf_x1, pdf_y1),
                "conf": int(item.get("confidence", 1.0) * 100),
                "source": "qwen-vl"
            })

        return mapped_results


# Global instance
_global_client: Optional[QwenVLOCRClient] = None


def get_qwen_vl_ocr_client(
    vllm_service_url: Optional[str] = None
) -> QwenVLOCRClient:
    """
    Get global Qwen2-VL OCR client (singleton).

    Args:
        vllm_service_url: Optional service URL

    Returns:
        QwenVLOCRClient instance

    Raises:
        ImportError: If client not available
    """
    global _global_client

    if _global_client is None:
        _global_client = QwenVLOCRClient(vllm_service_url)

    return _global_client


def reset_qwen_vl_ocr_client():
    """Reset global client (for testing)."""
    global _global_client
    _global_client = None


__all__ = [
    "QwenVLOCRClient",
    "get_qwen_vl_ocr_client",
    "reset_qwen_vl_ocr_client",
    "HAVE_CLIENT",
]
