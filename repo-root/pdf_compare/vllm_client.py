"""
vLLM Service Client

Client library for Streamlit UI to communicate with the vLLM microservice.
Handles all HTTP requests and provides a simple Python interface.

Usage in Streamlit UI or OCR code:
    from pdf_compare.vllm_client import get_vllm_client

    # Get client (auto-connects to service)
    client = get_vllm_client()

    # Check if service is available
    if client.is_available():
        # Use vLLM for high-quality OCR
        result = client.ocr_image(image_data)
    else:
        # Fallback to EasyOCR/Tesseract
        result = fallback_ocr(image_data)
"""

import os
import logging
import base64
import requests
from typing import List, Dict, Any, Optional
from io import BytesIO

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VLLMClientError(Exception):
    """Base exception for vLLM client errors."""
    pass


class VLLMServiceUnavailable(VLLMClientError):
    """Raised when vLLM service is not available."""
    pass


class VLLMClient:
    """
    Client for vLLM microservice.

    Provides high-level interface for:
    - Text-based LLM queries (layout analysis, reasoning)
    - Vision-based OCR (scanned documents)
    - Batch processing

    Falls back gracefully when service is unavailable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize vLLM client.

        Args:
            base_url: vLLM service URL (default: from env VLLM_HOST)
            timeout: Request timeout in seconds
            verify_ssl: Verify SSL certificates
        """
        self.base_url = (
            base_url
            or os.getenv("VLLM_HOST")
            or os.getenv("VLLM_SERVICE_URL")
            or "http://localhost:8000"
        ).rstrip("/")

        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._available = None  # Cache availability status

        logger.info(f"vLLM client initialized: {self.base_url}")

    def is_available(self, force_check: bool = False) -> bool:
        """
        Check if vLLM service is available.

        Args:
            force_check: Force health check even if cached

        Returns:
            True if service is healthy and ready
        """
        if self._available is not None and not force_check:
            return self._available

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5,
                verify=self.verify_ssl
            )

            if response.status_code == 200:
                health = response.json()
                is_healthy = (
                    health.get("status") == "healthy" and
                    health.get("text_model_loaded", False)
                )
                self._available = is_healthy
                logger.info(f"vLLM service health check: {health}")
                return is_healthy

        except requests.RequestException as e:
            logger.warning(f"vLLM service unavailable: {e}")
            self._available = False

        return False

    def query(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """
        Single text query to LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            VLLMServiceUnavailable: If service is not available
            VLLMClientError: On other errors
        """
        if not self.is_available():
            raise VLLMServiceUnavailable(
                f"vLLM service not available at {self.base_url}"
            )

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/query",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            response.raise_for_status()
            return response.json()["text"]

        except requests.HTTPError as e:
            logger.error(f"vLLM query failed: {e}")
            raise VLLMClientError(f"HTTP {e.response.status_code}: {e.response.text}")

        except requests.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            raise VLLMClientError(str(e))

    def batch_query(
        self,
        prompts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> List[str]:
        """
        Batch process multiple prompts (faster than individual queries).

        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Max tokens per response

        Returns:
            List of generated texts (same order as prompts)

        Raises:
            VLLMServiceUnavailable: If service not available
        """
        if not self.is_available():
            raise VLLMServiceUnavailable("vLLM service not available")

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/batch",
                json={
                    "prompts": prompts,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=self.timeout * 2,  # Longer timeout for batch
                verify=self.verify_ssl
            )

            response.raise_for_status()
            return response.json()["results"]

        except requests.RequestException as e:
            logger.error(f"Batch query failed: {e}")
            raise VLLMClientError(str(e))

    def ocr_image(
        self,
        image: np.ndarray,
        focus_technical: bool = True,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract text from image using Vision-Language Model.

        THIS IS THE KEY METHOD FOR SCANNED DOCUMENTS!

        Args:
            image: Image as numpy array (grayscale or RGB)
            focus_technical: Optimize for engineering documents
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of dicts with keys:
                - text: Extracted text
                - bbox: [x0, y0, x1, y1] as percentages (0-100)
                - confidence: Confidence score (0.0-1.0)

        Raises:
            VLLMServiceUnavailable: If service not available
        """
        if not self.is_available():
            raise VLLMServiceUnavailable("vLLM service not available")

        try:
            # Convert numpy array to base64-encoded PNG
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image).convert("RGB")
            else:  # RGB
                pil_image = Image.fromarray(image)

            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Call OCR endpoint
            response = requests.post(
                f"{self.base_url}/api/v1/ocr",
                json={
                    "image_base64": image_base64,
                    "focus_technical": focus_technical,
                    "min_confidence": min_confidence
                },
                timeout=self.timeout * 3,  # OCR takes longer
                verify=self.verify_ssl
            )

            response.raise_for_status()
            return response.json()["texts"]

        except requests.RequestException as e:
            logger.error(f"OCR request failed: {e}")
            raise VLLMClientError(str(e))

    def ocr_image_to_pdf_coords(
        self,
        image: np.ndarray,
        image_width: int,
        image_height: int,
        focus_technical: bool = True,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract text and convert bbox percentages to pixel coordinates.

        Args:
            image: Image as numpy array
            image_width: Image width in pixels
            image_height: Image height in pixels
            focus_technical: Optimize for engineering docs
            min_confidence: Minimum confidence

        Returns:
            List of dicts with bbox in pixel coordinates
        """
        results = self.ocr_image(image, focus_technical, min_confidence)

        # Convert percentage bboxes to pixel coordinates
        for item in results:
            bbox_pct = item["bbox"]
            item["bbox"] = (
                int(bbox_pct[0] * image_width / 100),
                int(bbox_pct[1] * image_height / 100),
                int(bbox_pct[2] * image_width / 100),
                int(bbox_pct[3] * image_height / 100),
            )

        return results


# ============================================================================
# Global Client Instance
# ============================================================================

_global_client: Optional[VLLMClient] = None


def get_vllm_client(
    base_url: Optional[str] = None,
    force_new: bool = False
) -> VLLMClient:
    """
    Get global vLLM client instance (singleton).

    Args:
        base_url: Optional custom URL (uses env if not provided)
        force_new: Create new client instead of using cached

    Returns:
        VLLMClient instance
    """
    global _global_client

    if _global_client is None or force_new:
        _global_client = VLLMClient(base_url=base_url)

    return _global_client


def reset_vllm_client():
    """Reset global client (useful for testing)."""
    global _global_client
    _global_client = None


__all__ = [
    "VLLMClient",
    "VLLMClientError",
    "VLLMServiceUnavailable",
    "get_vllm_client",
    "reset_vllm_client",
]
