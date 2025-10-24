"""
Unit tests for Preprocessor class.

Run with: pytest test_preprocessor.py -v
"""
import numpy as np
import cv2
import pytest
from pdf_compare.analyzers.preprocessor import Preprocessor, PreprocessPolicy


class TestPreprocessor:
    """Test cases for Preprocessor class."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample grayscale image for testing."""
        # Create 100x100 grayscale image with some noise
        img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
        # Add some text-like features
        cv2.rectangle(img, (10, 10), (90, 30), 0, -1)  # Dark rectangle
        cv2.putText(img, "TEST", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        return img

    def test_none_policy(self, sample_image):
        """Test that NONE policy returns image unchanged."""
        result = Preprocessor.apply(sample_image, PreprocessPolicy.NONE)
        assert np.array_equal(result, sample_image)

    def test_tesseract_drawings_policy(self, sample_image):
        """Test Tesseract drawings preprocessing."""
        result = Preprocessor.apply(sample_image, PreprocessPolicy.TESSERACT_DRAWINGS)

        # Check that output is valid
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # Check that it's been binarized (only 0 and 255 values)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2  # Should be binary (0 and 255)

    def test_easyocr_light_policy(self, sample_image):
        """Test EasyOCR light preprocessing."""
        result = Preprocessor.apply(sample_image, PreprocessPolicy.EASYOCR_LIGHT)

        # Check that output is valid
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

        # Should still be grayscale (not binary)
        unique_values = np.unique(result)
        assert len(unique_values) > 2  # Should have multiple gray levels

    def test_qwen_minimal_policy(self, sample_image):
        """Test Qwen-VL minimal preprocessing."""
        result = Preprocessor.apply(sample_image, PreprocessPolicy.QWEN_MINIMAL)

        # Should be unchanged
        assert np.array_equal(result, sample_image)

    def test_upscaling_no_scale(self, sample_image):
        """Test upscaling with factor 1.0 (no scaling)."""
        result, scale = Preprocessor.apply_with_upscaling(
            sample_image,
            PreprocessPolicy.NONE,
            upscale_factor=1.0
        )

        assert scale == 1.0
        assert np.array_equal(result, sample_image)

    def test_upscaling_2x(self, sample_image):
        """Test 2x upscaling."""
        result, scale = Preprocessor.apply_with_upscaling(
            sample_image,
            PreprocessPolicy.NONE,
            upscale_factor=2.0
        )

        assert scale == 2.0
        assert result.shape == (200, 200)  # 2x the original size

    def test_upscaling_with_sharpening(self, sample_image):
        """Test upscaling with sharpening enabled."""
        result, scale = Preprocessor.apply_with_upscaling(
            sample_image,
            PreprocessPolicy.NONE,
            upscale_factor=1.5,
            enable_sharpening=True
        )

        assert scale == 1.5
        assert result.shape == (150, 150)  # 1.5x the original size

    def test_upscaling_with_preprocessing(self, sample_image):
        """Test upscaling combined with preprocessing policy."""
        result, scale = Preprocessor.apply_with_upscaling(
            sample_image,
            PreprocessPolicy.EASYOCR_LIGHT,
            upscale_factor=2.0
        )

        assert scale == 2.0
        assert result.shape == (200, 200)
        # Should have been preprocessed after upscaling
        assert not np.array_equal(result, cv2.resize(sample_image, (200, 200)))

    def test_get_policy_for_engine_tesseract(self):
        """Test policy recommendation for Tesseract."""
        policy = Preprocessor.get_policy_for_engine("tesseract")
        assert policy == PreprocessPolicy.TESSERACT_DRAWINGS

    def test_get_policy_for_engine_easyocr(self):
        """Test policy recommendation for EasyOCR."""
        policy = Preprocessor.get_policy_for_engine("easyocr")
        assert policy == PreprocessPolicy.EASYOCR_LIGHT

    def test_get_policy_for_engine_qwen(self):
        """Test policy recommendation for Qwen-VL."""
        policy = Preprocessor.get_policy_for_engine("qwen-vl")
        assert policy == PreprocessPolicy.QWEN_MINIMAL

    def test_get_policy_for_unknown_engine(self):
        """Test policy recommendation for unknown engine."""
        policy = Preprocessor.get_policy_for_engine("unknown")
        # Should default to Tesseract
        assert policy == PreprocessPolicy.TESSERACT_DRAWINGS

    def test_invalid_policy_raises_error(self, sample_image):
        """Test that invalid policy raises ValueError."""
        with pytest.raises(ValueError):
            # Create a fake invalid enum (this won't work directly, so test the logic)
            Preprocessor.apply(sample_image, "invalid_policy")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
