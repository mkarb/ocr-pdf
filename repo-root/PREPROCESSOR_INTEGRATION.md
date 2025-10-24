# Preprocessor Integration Guide

## Overview

The `Preprocessor` class centralizes all image preprocessing logic in one place, eliminating code duplication across OCR engines.

## Before (Duplicated Code)

Each OCR engine function had its own preprocessing:

```python
# In _ocr_tile_tesseract()
base = cv2.bilateralFilter(gray, 9, 75, 75)
background = cv2.medianBlur(base, 21)
shade_removed = cv2.absdiff(base, background)
# ... 30 more lines of preprocessing

# In _ocr_tile_easyocr()
proc = cv2.bilateralFilter(gray, 5, 50, 50)  # Duplicated logic

# In _ocr_tile_qwen_vl()
# No preprocessing, but unclear why
```

## After (Centralized)

Single line with named policy:

```python
from .preprocessor import Preprocessor, PreprocessPolicy

# In _ocr_tile_tesseract()
proc = Preprocessor.apply(gray, PreprocessPolicy.TESSERACT_DRAWINGS)

# In _ocr_tile_easyocr()
proc = Preprocessor.apply(gray, PreprocessPolicy.EASYOCR_LIGHT)

# In _ocr_tile_qwen_vl()
proc = Preprocessor.apply(gray, PreprocessPolicy.QWEN_MINIMAL)
```

## Integration Steps

### Step 1: Update `_ocr_tile_tesseract()`

**Before:**
```python
def _ocr_tile_tesseract(gray: np.ndarray, cfg: HighResOCRConfig, upscale_factor: float = 1.0, enable_sharpening: bool = False) -> List[Dict]:
    # Optional upscaling for small text
    if upscale_factor > 1.0:
        gray, actual_scale = _preprocess_tile_upscale(gray, upscale_factor, enable_sharpening)
    else:
        actual_scale = 1.0

    # Enhanced preprocessing for engineering drawings (shaded tables, dense grids)
    # 1. Denoise with bilateral filter (preserves edges better than Gaussian)
    base = cv2.bilateralFilter(gray, 9, 75, 75)
    # ... 30 lines of preprocessing

    # Use image_to_data to get boxes + confidences
    ts_cfg = f"-l {cfg.lang} --psm {cfg.psm} --oem 3"
    data = pytesseract.image_to_data(proc, config=ts_cfg, output_type=pytesseract.Output.DICT)
```

**After:**
```python
from .preprocessor import Preprocessor, PreprocessPolicy

def _ocr_tile_tesseract(gray: np.ndarray, cfg: HighResOCRConfig, upscale_factor: float = 1.0, enable_sharpening: bool = False) -> List[Dict]:
    # Apply preprocessing with optional upscaling
    proc, actual_scale = Preprocessor.apply_with_upscaling(
        gray,
        policy=PreprocessPolicy.TESSERACT_DRAWINGS,
        upscale_factor=upscale_factor,
        enable_sharpening=enable_sharpening
    )

    # Use image_to_data to get boxes + confidences
    ts_cfg = f"-l {cfg.lang} --psm {cfg.psm} --oem 3"
    data = pytesseract.image_to_data(proc, config=ts_cfg, output_type=pytesseract.Output.DICT)
```

**Lines saved:** ~40 lines → ~6 lines ✅

---

### Step 2: Update `_ocr_tile_easyocr()`

**Before:**
```python
def _ocr_tile_easyocr(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    reader = _get_easyocr_reader(lang=cfg.lang, use_gpu=cfg.use_gpu)

    # EasyOCR works best with minimal preprocessing
    # Just apply light denoising to reduce noise
    proc = cv2.bilateralFilter(gray, 5, 50, 50)
```

**After:**
```python
from .preprocessor import Preprocessor, PreprocessPolicy

def _ocr_tile_easyocr(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    reader = _get_easyocr_reader(lang=cfg.lang, use_gpu=cfg.use_gpu)

    # Apply light preprocessing
    proc = Preprocessor.apply(gray, PreprocessPolicy.EASYOCR_LIGHT)
```

**Lines saved:** 4 lines → 2 lines ✅

---

### Step 3: Update `_ocr_tile_qwen_vl()`

**Before:**
```python
def _ocr_tile_qwen_vl_base(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    # No explicit preprocessing (unclear why)
    # Convert grayscale to PIL Image
    pil_img = Image.fromarray(gray)
```

**After:**
```python
from .preprocessor import Preprocessor, PreprocessPolicy

def _ocr_tile_qwen_vl_base(gray: np.ndarray, cfg: HighResOCRConfig) -> List[Dict]:
    # Apply minimal preprocessing (vision model handles the rest)
    proc = Preprocessor.apply(gray, PreprocessPolicy.QWEN_MINIMAL)

    # Convert to PIL Image
    pil_img = Image.fromarray(proc)
```

**Benefit:** Now it's explicit that Qwen-VL intentionally skips preprocessing ✅

---

### Step 4: Remove Old Helper Functions

Delete these functions (no longer needed):
- `_preprocess_tile_upscale()` (lines 267-302) - Now in Preprocessor

**Total lines removed:** ~35 lines ✅

---

## Benefits

### 1. **Maintainability**
- ✅ Change preprocessing in ONE place
- ✅ Easy to add new policies (e.g., `TESSERACT_PHOTOS`)
- ✅ Clear what each engine does

### 2. **Testability**
- ✅ Unit test preprocessing independent of OCR
- ✅ Test different policies on same image
- ✅ Regression testing when tuning algorithms

### 3. **Debuggability**
- ✅ Enable debug logging: `Preprocessor.apply(gray, policy, debug=True)`
- ✅ Save intermediate preprocessing steps
- ✅ Compare policies side-by-side

### 4. **Consistency**
- ✅ All engines use same preprocessing infrastructure
- ✅ No duplicated/divergent preprocessing code
- ✅ Easier to ensure preprocessing matches expectations

---

## Advanced Usage

### Custom Policy for Specific Use Case

```python
# In your code
from pdf_compare.analyzers.preprocessor import Preprocessor, PreprocessPolicy

# Use different policy for photos vs drawings
if is_photo:
    policy = PreprocessPolicy.EASYOCR_LIGHT
else:
    policy = PreprocessPolicy.TESSERACT_DRAWINGS

processed = Preprocessor.apply(image, policy)
```

### Debug Preprocessing

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

processed = Preprocessor.apply(image, PreprocessPolicy.TESSERACT_DRAWINGS, debug=True)

# Output:
# DEBUG:preprocessor:Applying preprocessing policy: tesseract_drawings
# DEBUG:preprocessor:Step 1: Bilateral filter (denoise, preserve edges)
# DEBUG:preprocessor:Step 2: Shade removal (normalize background)
# ...
```

### Programmatic Policy Selection

```python
# Automatically choose policy based on engine
from pdf_compare.analyzers.preprocessor import Preprocessor

engine = "tesseract"  # or "easyocr", "qwen-vl"
policy = Preprocessor.get_policy_for_engine(engine)
processed = Preprocessor.apply(image, policy)
```

---

## Testing

Run unit tests:
```bash
pytest test_preprocessor.py -v
```

Expected output:
```
test_preprocessor.py::TestPreprocessor::test_none_policy PASSED
test_preprocessor.py::TestPreprocessor::test_tesseract_drawings_policy PASSED
test_preprocessor.py::TestPreprocessor::test_easyocr_light_policy PASSED
test_preprocessor.py::TestPreprocessor::test_qwen_minimal_policy PASSED
test_preprocessor.py::TestPreprocessor::test_upscaling_2x PASSED
...
```

---

## Migration Checklist

- [ ] Create `preprocessor.py` file
- [ ] Run unit tests to verify Preprocessor works
- [ ] Update `_ocr_tile_tesseract()` to use Preprocessor
- [ ] Update `_ocr_tile_easyocr()` to use Preprocessor
- [ ] Update `_ocr_tile_qwen_vl()` to use Preprocessor
- [ ] Remove old `_preprocess_tile_upscale()` function
- [ ] Test end-to-end OCR with all three engines
- [ ] Verify debug visualization still works
- [ ] Update documentation

---

## Next Steps

After Preprocessor integration, consider:

1. **TileExecutor** - Centralize parallelism logic
2. **DebugSink** - Centralize debug output
3. **LayoutPolicy** - Centralize layout-aware processing
4. **Logging** - Replace print() with proper logging

This follows the Single Responsibility Principle and makes the codebase much more maintainable!
