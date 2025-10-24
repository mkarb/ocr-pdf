"""
Standalone unit tests for Preprocessor class.

Run with: python test_preprocessor_standalone.py
"""
import sys
import os
import numpy as np
import cv2

# Add parent directory to path to import preprocessor directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pdf_compare', 'analyzers'))
from preprocessor import Preprocessor, PreprocessPolicy


def test_none_policy():
    """Test that NONE policy returns image unchanged."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result = Preprocessor.apply(img, PreprocessPolicy.NONE)
    assert np.array_equal(result, img), "NONE policy should return unchanged image"
    print("[PASS] test_none_policy")


def test_tesseract_drawings_policy():
    """Test Tesseract drawings preprocessing."""
    # Create sample grayscale image with some noise
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 30), 0, -1)  # Dark rectangle
    cv2.putText(img, "TEST", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    result = Preprocessor.apply(img, PreprocessPolicy.TESSERACT_DRAWINGS)

    # Check that output is valid
    assert result.shape == img.shape, "Output shape should match input"
    assert result.dtype == np.uint8, "Output should be uint8"

    # Check that it's been binarized (only 0 and 255 values)
    unique_values = np.unique(result)
    assert len(unique_values) <= 2, "Should be binary (0 and 255)"
    print("[PASS] test_tesseract_drawings_policy")


def test_easyocr_light_policy():
    """Test EasyOCR light preprocessing."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result = Preprocessor.apply(img, PreprocessPolicy.EASYOCR_LIGHT)

    # Check that output is valid
    assert result.shape == img.shape, "Output shape should match input"
    assert result.dtype == np.uint8, "Output should be uint8"

    # Should still be grayscale (not binary)
    unique_values = np.unique(result)
    assert len(unique_values) > 2, "Should have multiple gray levels"
    print("[PASS] test_easyocr_light_policy")


def test_qwen_minimal_policy():
    """Test Qwen-VL minimal preprocessing."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result = Preprocessor.apply(img, PreprocessPolicy.QWEN_MINIMAL)

    # Should be unchanged
    assert np.array_equal(result, img), "QWEN_MINIMAL should return unchanged image"
    print("[PASS] test_qwen_minimal_policy")


def test_upscaling_no_scale():
    """Test upscaling with factor 1.0 (no scaling)."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result, scale = Preprocessor.apply_with_upscaling(
        img,
        PreprocessPolicy.NONE,
        upscale_factor=1.0
    )

    assert scale == 1.0, "Scale should be 1.0"
    assert np.array_equal(result, img), "Image should be unchanged"
    print("[PASS] test_upscaling_no_scale")


def test_upscaling_2x():
    """Test 2x upscaling."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result, scale = Preprocessor.apply_with_upscaling(
        img,
        PreprocessPolicy.NONE,
        upscale_factor=2.0
    )

    assert scale == 2.0, "Scale should be 2.0"
    assert result.shape == (200, 200), "Should be 2x the original size"
    print("[PASS] test_upscaling_2x")


def test_upscaling_with_sharpening():
    """Test upscaling with sharpening enabled."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result, scale = Preprocessor.apply_with_upscaling(
        img,
        PreprocessPolicy.NONE,
        upscale_factor=1.5,
        enable_sharpening=True
    )

    assert scale == 1.5, "Scale should be 1.5"
    assert result.shape == (150, 150), "Should be 1.5x the original size"
    print("[PASS] test_upscaling_with_sharpening")


def test_upscaling_with_preprocessing():
    """Test upscaling combined with preprocessing policy."""
    img = np.random.randint(200, 256, (100, 100), dtype=np.uint8)
    result, scale = Preprocessor.apply_with_upscaling(
        img,
        PreprocessPolicy.EASYOCR_LIGHT,
        upscale_factor=2.0
    )

    assert scale == 2.0, "Scale should be 2.0"
    assert result.shape == (200, 200), "Should be 2x the original size"
    # Should have been preprocessed after upscaling
    simple_upscale = cv2.resize(img, (200, 200))
    assert not np.array_equal(result, simple_upscale), "Should be preprocessed, not just upscaled"
    print("[PASS] test_upscaling_with_preprocessing")


def test_get_policy_for_engine_tesseract():
    """Test policy recommendation for Tesseract."""
    policy = Preprocessor.get_policy_for_engine("tesseract")
    assert policy == PreprocessPolicy.TESSERACT_DRAWINGS
    print("[PASS] test_get_policy_for_engine_tesseract")


def test_get_policy_for_engine_easyocr():
    """Test policy recommendation for EasyOCR."""
    policy = Preprocessor.get_policy_for_engine("easyocr")
    assert policy == PreprocessPolicy.EASYOCR_LIGHT
    print("[PASS] test_get_policy_for_engine_easyocr")


def test_get_policy_for_engine_qwen():
    """Test policy recommendation for Qwen-VL."""
    policy = Preprocessor.get_policy_for_engine("qwen-vl")
    assert policy == PreprocessPolicy.QWEN_MINIMAL
    print("[PASS] test_get_policy_for_engine_qwen")


def test_get_policy_for_unknown_engine():
    """Test policy recommendation for unknown engine."""
    policy = Preprocessor.get_policy_for_engine("unknown")
    # Should default to Tesseract
    assert policy == PreprocessPolicy.TESSERACT_DRAWINGS
    print("[PASS] test_get_policy_for_unknown_engine")


if __name__ == "__main__":
    print("Running Preprocessor unit tests...\n")

    tests = [
        test_none_policy,
        test_tesseract_drawings_policy,
        test_easyocr_light_policy,
        test_qwen_minimal_policy,
        test_upscaling_no_scale,
        test_upscaling_2x,
        test_upscaling_with_sharpening,
        test_upscaling_with_preprocessing,
        test_get_policy_for_engine_tesseract,
        test_get_policy_for_engine_easyocr,
        test_get_policy_for_engine_qwen,
        test_get_policy_for_unknown_engine,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Tests run: {len(tests)}")
    print(f"Passed: {len(tests) - failed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print(f"\n[FAILURE] {failed} test(s) failed")
        sys.exit(1)
