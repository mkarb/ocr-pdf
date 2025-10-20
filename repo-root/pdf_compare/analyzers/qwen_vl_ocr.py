"""
Qwen2-VL based OCR for scanned engineering documents.

This module provides high-accuracy OCR using Vision-Language Models (VLMs)
specifically optimized for:
- Scanned engineering drawings with no vector data
- Technical text (part numbers, dimensions, annotations)
- Large format documents (A0, A1, A2)
- Degraded or low-quality scans

Usage:
    # Initialize OCR engine
    ocr = QwenVLOCR(model_name="Qwen/Qwen2-VL-7B-Instruct")

    # Extract text from image
    results = ocr.extract_text_from_image(image_array)

    # Extract from specific region
    results = ocr.extract_text_from_region(
        image_array,
        region_bbox=(100, 200, 500, 600)
    )
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import vLLM (required for Qwen2-VL)
try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False
    logger.warning("vLLM not available. Install with: pip install vllm")


@dataclass
class VLOCRConfig:
    """Configuration for Vision-Language OCR."""

    # Model settings
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    tensor_parallel_size: int = 1  # 1 for single GPU, 2 for both AMD GPUs
    gpu_memory_utilization: float = 0.85
    dtype: str = "float16"  # float16 faster on AMD ROCm

    # Inference settings
    temperature: float = 0.1  # Low temperature for deterministic OCR
    max_tokens: int = 2048
    top_p: float = 0.95

    # OCR settings
    min_confidence: float = 0.5  # Minimum confidence to include result
    return_bboxes: bool = True  # Return bounding boxes with text
    focus_technical_text: bool = True  # Optimize for engineering docs


class QwenVLOCR:
    """
    Vision-Language Model OCR engine using Qwen2-VL.

    Provides high-accuracy text extraction from scanned documents using
    state-of-the-art vision-language models optimized for AMD ROCm.
    """

    def __init__(self, config: Optional[VLOCRConfig] = None):
        """
        Initialize Qwen2-VL OCR engine.

        Args:
            config: OCR configuration. If None, uses defaults.

        Raises:
            ImportError: If vLLM is not installed
            RuntimeError: If model fails to load
        """
        if not HAVE_VLLM:
            raise ImportError(
                "vLLM is required for Qwen2-VL OCR. "
                "Install with: pip install vllm"
            )

        self.config = config or VLOCRConfig()

        logger.info(f"Initializing Qwen2-VL OCR: {self.config.model_name}")
        logger.info(f"  Tensor parallel: {self.config.tensor_parallel_size}")
        logger.info(f"  GPU memory: {self.config.gpu_memory_utilization}")

        try:
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype=self.config.dtype,
                trust_remote_code=True,
                max_model_len=8192,  # Qwen2-VL supports long contexts
            )

            # Prepare sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )

            logger.info("Qwen2-VL OCR initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qwen2-VL: {e}")
            raise RuntimeError(f"Qwen2-VL initialization failed: {e}")

    def _build_ocr_prompt(self, focus_technical: bool = True) -> str:
        """
        Build OCR prompt optimized for engineering documents.

        Args:
            focus_technical: If True, emphasize technical text extraction

        Returns:
            Formatted prompt string
        """
        if focus_technical:
            prompt = """Extract all text from this engineering document image.

Focus on:
1. Part numbers, item numbers, and reference designators
2. Dimensions, measurements, and technical specifications
3. Labels, callouts, and annotations
4. Table content (headers and data)
5. Title block information

For each text element, provide:
- The exact text content (preserve capitalization and spacing)
- Bounding box as percentage of image dimensions [x0, y0, x1, y1] where (0,0) is top-left
- Confidence score (0.0 to 1.0)

Return JSON format:
{
  "texts": [
    {
      "text": "exact text content",
      "bbox": [x0_pct, y0_pct, x1_pct, y1_pct],
      "confidence": 0.95
    }
  ]
}

Important:
- Be precise with technical text (part numbers must be exact)
- Include all visible text, even if partially obscured
- Preserve formatting (newlines, spaces)
- Return valid JSON only, no additional commentary"""

        else:
            prompt = """Extract all visible text from this image.

For each text element, provide the text content, bounding box, and confidence.

Return JSON format:
{
  "texts": [
    {"text": "...", "bbox": [x0, y0, x1, y1], "confidence": 0.95}
  ]
}

Bounding boxes are percentages of image dimensions (0-100).
Return valid JSON only."""

        return prompt

    def extract_text_from_image(
        self,
        image: np.ndarray,
        focus_technical: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from full image using VLM.

        Args:
            image: Input image as numpy array (grayscale or RGB)
            focus_technical: If True, optimizes for technical documents

        Returns:
            List of dicts with keys: 'text', 'bbox', 'confidence'
            bbox is (x0, y0, x1, y1) in pixel coordinates
        """
        # Convert to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image).convert("RGB")
        else:  # Already RGB
            pil_image = Image.fromarray(image)

        # Build prompt
        prompt = self._build_ocr_prompt(focus_technical=focus_technical)

        # Run VLM inference
        logger.debug(f"Running Qwen2-VL OCR on image: {image.shape}")

        try:
            # vLLM multimodal inference
            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": pil_image}}],
                self.sampling_params
            )

            response_text = outputs[0].outputs[0].text
            logger.debug(f"VLM response: {response_text[:200]}...")

            # Parse JSON response
            result = json.loads(response_text)

            # Convert percentage bboxes to pixel coordinates
            image_h, image_w = image.shape[:2]
            processed_results = []

            for item in result.get("texts", []):
                confidence = item.get("confidence", 1.0)

                # Filter by confidence
                if confidence < self.config.min_confidence:
                    continue

                # Convert percentage bbox to pixels
                bbox_pct = item.get("bbox", [0, 0, 100, 100])
                bbox_px = (
                    int(bbox_pct[0] * image_w / 100),
                    int(bbox_pct[1] * image_h / 100),
                    int(bbox_pct[2] * image_w / 100),
                    int(bbox_pct[3] * image_h / 100),
                )

                processed_results.append({
                    "text": item.get("text", ""),
                    "bbox": bbox_px,
                    "confidence": confidence,
                    "source": "qwen-vl"
                })

            logger.info(f"Extracted {len(processed_results)} text items (VLM OCR)")
            return processed_results

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VLM JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return []

        except Exception as e:
            logger.error(f"VLM OCR failed: {e}")
            return []

    def extract_text_from_tile(
        self,
        tile_image: np.ndarray,
        tile_bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a tile and map coordinates to full page.

        Compatible with existing tiled_ocr() infrastructure.

        Args:
            tile_image: Tile image as numpy array
            tile_bbox: Tile position in PDF coordinates (x0, y0, x1, y1)
            page_width: Full page width in PDF points
            page_height: Full page height in PDF points

        Returns:
            List of OCR results with bboxes in PDF coordinates
        """
        # Extract text from tile
        tile_results = self.extract_text_from_image(
            tile_image,
            focus_technical=self.config.focus_technical_text
        )

        # Map tile-local pixel coords to PDF coords
        tile_x0, tile_y0, tile_x1, tile_y1 = tile_bbox
        tile_h, tile_w = tile_image.shape[:2]

        mapped_results = []
        for item in tile_results:
            # Get bbox in tile-local pixel coordinates
            tx0, ty0, tx1, ty1 = item["bbox"]

            # Map to PDF coordinates
            # Tile pixel → PDF coordinate mapping
            pdf_x0 = tile_x0 + (tx0 / tile_w) * (tile_x1 - tile_x0)
            pdf_y0 = tile_y0 + (ty0 / tile_h) * (tile_y1 - tile_y0)
            pdf_x1 = tile_x0 + (tx1 / tile_w) * (tile_x1 - tile_x0)
            pdf_y1 = tile_y0 + (ty1 / tile_h) * (tile_y1 - tile_y0)

            mapped_results.append({
                "text": item["text"],
                "bbox": (pdf_x0, pdf_y0, pdf_x1, pdf_y1),
                "conf": int(item["confidence"] * 100),  # Match existing format
                "source": "qwen-vl"
            })

        return mapped_results

    def extract_text_from_region(
        self,
        page_image: np.ndarray,
        region_bbox: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Extract text from specific region of image.

        Args:
            page_image: Full page image
            region_bbox: Region to OCR (x0, y0, x1, y1) in pixels

        Returns:
            List of OCR results with bboxes in full page coordinates
        """
        x0, y0, x1, y1 = region_bbox

        # Crop region
        region_img = page_image[y0:y1, x0:x1]

        # Extract text from region
        region_results = self.extract_text_from_image(region_img)

        # Map region-local coords to full page coords
        mapped_results = []
        for item in region_results:
            rx0, ry0, rx1, ry1 = item["bbox"]

            mapped_results.append({
                "text": item["text"],
                "bbox": (x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1),
                "confidence": item["confidence"],
                "source": "qwen-vl"
            })

        return mapped_results


# Singleton instance (lazy-initialized)
_global_qwen_vl_ocr: Optional[QwenVLOCR] = None


def get_qwen_vl_ocr(config: Optional[VLOCRConfig] = None) -> QwenVLOCR:
    """
    Get global Qwen2-VL OCR instance (singleton pattern).

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        Global Qwen2-VL OCR engine

    Raises:
        ImportError: If vLLM not available
    """
    global _global_qwen_vl_ocr

    if _global_qwen_vl_ocr is None:
        _global_qwen_vl_ocr = QwenVLOCR(config)

    return _global_qwen_vl_ocr


def reset_qwen_vl_ocr():
    """Reset global OCR instance (useful for testing)."""
    global _global_qwen_vl_ocr
    _global_qwen_vl_ocr = None


__all__ = [
    "QwenVLOCR",
    "VLOCRConfig",
    "get_qwen_vl_ocr",
    "reset_qwen_vl_ocr",
    "HAVE_VLLM",
]
