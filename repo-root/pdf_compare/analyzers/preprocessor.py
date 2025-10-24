"""
Image preprocessing policies for OCR engines.

Centralizes all preprocessing logic in one place with named policies,
eliminating code duplication across different OCR engines.

Usage:
    from .preprocessor import Preprocessor, PreprocessPolicy

    # Apply preprocessing
    processed = Preprocessor.apply(gray_image, PreprocessPolicy.TESSERACT_DRAWINGS)

    # Or with upscaling
    processed, scale = Preprocessor.apply_with_upscaling(
        gray_image,
        policy=PreprocessPolicy.TESSERACT_DRAWINGS,
        upscale_factor=1.5,
        enable_sharpening=True
    )
"""
from __future__ import annotations
from enum import Enum
from typing import Tuple, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PreprocessPolicy(Enum):
    """Named preprocessing policies for different OCR engines and use cases."""

    NONE = "none"
    """No preprocessing - pass image as-is."""

    TESSERACT_DRAWINGS = "tesseract_drawings"
    """
    Heavy preprocessing optimized for Tesseract on engineering drawings.

    Pipeline:
    1. Bilateral filter (denoise while preserving edges)
    2. Shade removal (normalize uneven backgrounds/scanning artifacts)
    3. Adaptive threshold (handles varying lighting/contrast)
    4. Vertical line cleanup (removes table ruling lines that confuse OCR)
    5. Morphological close (reconnects characters broken by previous steps)

    Best for: Technical drawings, diagrams, tables with ruling lines
    """

    EASYOCR_LIGHT = "easyocr_light"
    """
    Light denoising for EasyOCR.

    EasyOCR's neural network handles most preprocessing internally,
    so we only apply minimal bilateral filtering to reduce noise.

    Best for: General text, photos, natural images
    """

    QWEN_MINIMAL = "qwen_minimal"
    """
    Minimal preprocessing for Qwen-VL (vision-language model).

    Vision models are trained on diverse images and handle
    preprocessing internally. Pass image as-is.

    Best for: Complex layouts, mixed content
    """


class Preprocessor:
    """Centralized image preprocessing with named policies."""

    @staticmethod
    def apply(
        gray: np.ndarray,
        policy: PreprocessPolicy,
        debug: bool = False
    ) -> np.ndarray:
        """
        Apply preprocessing policy to grayscale image.

        Args:
            gray: Grayscale image (H×W numpy array)
            policy: Preprocessing policy to apply
            debug: If True, log preprocessing steps

        Returns:
            Preprocessed grayscale image

        Raises:
            ValueError: If policy is unknown
        """
        if debug:
            logger.debug(f"Applying preprocessing policy: {policy.value}")

        if policy == PreprocessPolicy.NONE:
            return gray
        elif policy == PreprocessPolicy.TESSERACT_DRAWINGS:
            return Preprocessor._tesseract_drawings(gray, debug=debug)
        elif policy == PreprocessPolicy.EASYOCR_LIGHT:
            return Preprocessor._easyocr_light(gray, debug=debug)
        elif policy == PreprocessPolicy.QWEN_MINIMAL:
            return Preprocessor._qwen_minimal(gray, debug=debug)
        else:
            raise ValueError(f"Unknown preprocessing policy: {policy}")

    @staticmethod
    def apply_with_upscaling(
        gray: np.ndarray,
        policy: PreprocessPolicy,
        upscale_factor: float = 1.0,
        enable_sharpening: bool = False,
        debug: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Apply preprocessing with optional upscaling for small text.

        Args:
            gray: Grayscale image
            policy: Preprocessing policy
            upscale_factor: Upscaling multiplier (1.0 = no upscaling, 1.5-2.0 recommended)
            enable_sharpening: Apply unsharp mask after upscaling
            debug: Log preprocessing steps

        Returns:
            (processed_image, actual_scale_factor)
        """
        # Upscale first if requested
        if upscale_factor > 1.0:
            gray, actual_scale = Preprocessor._upscale_image(
                gray,
                upscale_factor,
                enable_sharpening,
                debug=debug
            )
        else:
            actual_scale = 1.0

        # Apply preprocessing policy
        processed = Preprocessor.apply(gray, policy, debug=debug)

        return processed, actual_scale

    @staticmethod
    def _upscale_image(
        gray: np.ndarray,
        upscale_factor: float,
        enable_sharpening: bool,
        debug: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Upscale image for better OCR on small text.

        Args:
            gray: Input grayscale image
            upscale_factor: Target upscaling factor
            enable_sharpening: Apply unsharp mask
            debug: Log steps

        Returns:
            (upscaled_image, actual_scale_factor)
        """
        if debug:
            logger.debug(f"Upscaling image by {upscale_factor}x")

        height, width = gray.shape
        new_width = int(width * upscale_factor)
        new_height = int(height * upscale_factor)

        # Use cubic interpolation for best quality
        upscaled = cv2.resize(
            gray,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        # Optional sharpening to enhance edges after upscaling
        if enable_sharpening:
            if debug:
                logger.debug("Applying unsharp mask for sharpening")

            # Gaussian blur for unsharp mask
            blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)
            upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

        return upscaled, upscale_factor

    @staticmethod
    def _tesseract_drawings(gray: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Heavy preprocessing for engineering drawings (Tesseract).

        This pipeline is specifically designed for technical drawings with:
        - Varying background shading from scanning
        - Table ruling lines that confuse Tesseract
        - Small text that needs enhancement
        - Dense grids and diagrams

        Args:
            gray: Grayscale image
            debug: Log processing steps

        Returns:
            Binary (black & white) processed image
        """
        if debug:
            logger.debug("Step 1: Bilateral filter (denoise, preserve edges)")

        # 1. Denoise with bilateral filter (preserves edges better than Gaussian)
        base = cv2.bilateralFilter(gray, 9, 75, 75)

        if debug:
            logger.debug("Step 2: Shade removal (normalize background)")

        # 2. Remove soft background shading that appears in scanned documents
        # This is critical for engineering drawings scanned from paper
        background = cv2.medianBlur(base, 21)
        shade_removed = cv2.absdiff(base, background)
        shade_removed = cv2.normalize(shade_removed, None, 0, 255, cv2.NORM_MINMAX)

        if debug:
            logger.debug("Step 3: Adaptive thresholding")

        # 3. Adaptive thresholding with dynamic window size
        # Larger windows for large images to handle varying lighting
        min_dim = min(shade_removed.shape[:2])
        if min_dim > 600:
            block_size = 41
        elif min_dim > 400:
            block_size = 31
        elif min_dim > 200:
            block_size = 21
        else:
            block_size = 11

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        proc = cv2.adaptiveThreshold(
            shade_removed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2,
        )

        if debug:
            logger.debug("Step 4: Vertical line cleanup")

        # 4. Remove vertical ruling lines that confuse Tesseract's confidence scores
        # These appear in tables and dimension lines on technical drawings
        vertical_kernel_len = max(3, int(proc.shape[0] * 0.05))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
        vertical_lines = cv2.morphologyEx(proc, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        proc = cv2.subtract(proc, vertical_lines)

        if debug:
            logger.debug("Step 5: Morphological close (reconnect characters)")

        # 5. Morphological close to reconnect characters broken by previous steps
        kernel = np.ones((2, 2), np.uint8)
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

        return proc

    @staticmethod
    def _easyocr_light(gray: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Light denoising for EasyOCR.

        EasyOCR's neural network is trained on diverse images and handles
        most preprocessing internally. Heavy preprocessing can actually
        hurt accuracy. We only apply minimal bilateral filtering.

        Args:
            gray: Grayscale image
            debug: Log processing steps

        Returns:
            Lightly denoised grayscale image
        """
        if debug:
            logger.debug("Applying light bilateral filter for EasyOCR")

        # Light bilateral filter to reduce noise without losing detail
        return cv2.bilateralFilter(gray, 5, 50, 50)

    @staticmethod
    def _qwen_minimal(gray: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Minimal preprocessing for Qwen-VL.

        Vision-language models like Qwen2-VL are trained on raw images
        and handle all preprocessing internally. Pass the image as-is.

        Args:
            gray: Grayscale image
            debug: Log processing steps

        Returns:
            Unmodified grayscale image
        """
        if debug:
            logger.debug("No preprocessing for Qwen-VL (vision model handles it)")

        return gray

    @staticmethod
    def get_policy_for_engine(engine: str) -> PreprocessPolicy:
        """
        Get recommended preprocessing policy for an OCR engine.

        Args:
            engine: OCR engine name ("tesseract", "easyocr", "qwen-vl")

        Returns:
            Recommended preprocessing policy
        """
        engine_lower = engine.lower()

        if engine_lower == "tesseract":
            return PreprocessPolicy.TESSERACT_DRAWINGS
        elif engine_lower == "easyocr":
            return PreprocessPolicy.EASYOCR_LIGHT
        elif engine_lower == "qwen-vl":
            return PreprocessPolicy.QWEN_MINIMAL
        else:
            logger.warning(f"Unknown engine '{engine}', using TESSERACT_DRAWINGS policy")
            return PreprocessPolicy.TESSERACT_DRAWINGS
