# OCR Pipeline Refactoring - Status Overview

## Your Complete Refactoring Plan

### 1. **Unify "page OCR" entry points** ❌ NOT STARTED
**Goal:** Replace `highres_ocr()` and `tiled_ocr()` with single `OCRPage.run(...)`

**Current state:** Still have two separate entry points in [highres_ocr.py](pdf_compare/analyzers/highres_ocr.py):
- `tiled_ocr()` - Lines 1069-1555 (~486 lines)
- `highres_ocr()` - Lines 1558-1720 (~162 lines)

**What needs to happen:**
- Create `OCRPage` class with unified `run()` method
- Accept either `tiles_pdf: List[BBox]` or computed `TileGrid`
- Eliminate duplicate PDF rendering/mapping logic between the two functions

---

### 2. **Strategy pattern for engines** ❌ NOT STARTED
**Goal:** Create `BaseOCREngine` + engine implementations

**Current state:** Engine logic embedded in highres_ocr.py:
- `_ocr_tile_tesseract()` - Lines 368-463 (~95 lines)
- `_ocr_tile_easyocr()` - Lines 466-527 (~61 lines)
- `_ocr_tile_qwen_vl()` - Lines 323-365 (~42 lines)
- Engine-specific splitting in `_get_engine_max_dimension()` and `_ocr_tile_with_splitting()`

**What needs to happen:**
```python
class BaseOCREngine:
    def max_side(self) -> int:
        """Return max dimension this engine can handle."""
        raise NotImplementedError

    def infer(self, gray: np.ndarray, cfg: Config) -> List[Dict]:
        """OCR without splitting (guaranteed to be within max_side)."""
        raise NotImplementedError

    def run(self, gray: np.ndarray, cfg: Config) -> List[Dict]:
        """Handle splitting automatically using max_side()."""
        # Base class implements recursive splitting here
        pass

class TesseractEngine(BaseOCREngine):
    def max_side(self) -> int:
        return 29000

    def infer(self, gray: np.ndarray, cfg: Config) -> List[Dict]:
        # Current _ocr_tile_tesseract() logic (without splitting)
        pass

class EasyOCREngine(BaseOCREngine):
    def max_side(self) -> int:
        return 32000

    def infer(self, gray: np.ndarray, cfg: Config) -> List[Dict]:
        # Current _ocr_tile_easyocr() logic
        pass

class QwenVLEngine(BaseOCREngine):
    def max_side(self) -> int:
        return 4096

    def infer(self, gray: np.ndarray, cfg: Config) -> List[Dict]:
        # Current _ocr_tile_qwen_vl() logic
        pass
```

**Benefits:**
- Remove all per-engine splitting branches
- Centralize splitting logic in `BaseOCREngine.run()`
- Easy to add new engines without modifying core code

---

### 3. **One coordinate mapper** ❌ NOT STARTED
**Goal:** Single helper for all bbox mappings (tile-local px → PDF coords)

**Current state:** Coordinate mapping scattered throughout highres_ocr.py:
- Manual bbox adjustments in `_ocr_tile_with_splitting()` - Lines 579-591
- Tile coordinate mapping in `tiled_ocr()` - Lines 1435-1443
- Multiple places do `bbox = (x0 + tile_x0, y0 + tile_y0, x1 + tile_x0, y1 + tile_y0)`

**What needs to happen:**
```python
class CoordinateMapper:
    @staticmethod
    def tile_to_pdf(
        bbox: Tuple[int, int, int, int],
        tile_offset: Tuple[int, int],
        scale_factor: float = 1.0
    ) -> Tuple[int, int, int, int]:
        """Map tile-local coordinates to PDF coordinates."""
        x0, y0, x1, y1 = bbox
        tile_x0, tile_y0 = tile_offset
        return (
            int((x0 + tile_x0) / scale_factor),
            int((y0 + tile_y0) / scale_factor),
            int((x1 + tile_x0) / scale_factor),
            int((y1 + tile_y0) / scale_factor)
        )
```

---

### 4. **One renderer** ❌ NOT STARTED
**Goal:** `PDFRenderer` opens document once, renders page or clipped region

**Current state:** PDF rendering scattered:
- `tiled_ocr()` opens PDF with `fitz.open()` - Line 1129
- `highres_ocr()` opens PDF separately - Line 1562
- Rendering logic duplicated between functions
- Multiple `doc.load_page()` and `page.get_pixmap()` calls

**What needs to happen:**
```python
class PDFRenderer:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)

    def render_page(
        self,
        page_num: int,
        dpi: int = 300,
        clip_rect: Optional[fitz.Rect] = None
    ) -> Tuple[np.ndarray, float]:
        """Render page or region, return grayscale + actual zoom."""
        page = self.doc.load_page(page_num)
        zoom = dpi / 72.0

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

        # Convert to grayscale numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        gray = cv2.cvtColor(img.reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2GRAY)

        return gray, zoom

    def close(self):
        self.doc.close()
```

---

### 5. **Single preprocessing policy** ✓ **COMPLETE**
**Goal:** Centralized preprocessing with named policies

**Status:** ✓ Fully implemented and tested
- [preprocessor.py](pdf_compare/analyzers/preprocessor.py) - 334 lines
- 4 named policies: NONE, TESSERACT_DRAWINGS, EASYOCR_LIGHT, QWEN_MINIMAL
- 12/12 unit tests passing
- Integration guide available

**Next step:** Integrate into highres_ocr.py (migration checklist in PREPROCESSOR_STATUS.md)

---

### 6. **Parallelism & memory sizing** ⚠️ **PARTIALLY COMPLETE**
**Goal:** Put worker sizing and parallel execution in `TileExecutor`

**Current state:**
- ✓ Worker calculation exists: `calculate_optimal_workers()` - Lines 988-1066 (78 lines)
- ✓ Parallel execution implemented with ThreadPoolExecutor - Lines 1402-1426
- ❌ Not centralized in dedicated `TileExecutor` class
- ❌ Logic still embedded in `tiled_ocr()` function

**What needs to happen:**
```python
class TileExecutor:
    def __init__(self, engine: str, use_gpu: bool, ram_budget_mb: int = 8192):
        self.engine = engine
        self.use_gpu = use_gpu
        self.ram_budget_mb = ram_budget_mb

    def calculate_workers(
        self,
        tile_pixel_area: int,
        num_tiles: int,
        max_workers_limit: int = 8
    ) -> int:
        """Calculate optimal workers based on memory constraints."""
        # Current calculate_optimal_workers() logic
        pass

    def execute_parallel(
        self,
        tiles: List[Dict],
        worker_func: Callable,
        max_workers: Optional[int] = None
    ) -> List[Dict]:
        """Execute tiles in parallel with ThreadPoolExecutor."""
        if max_workers is None:
            max_workers = self.calculate_workers(...)

        # Current parallel execution logic
        pass
```

---

### 7. **Debug hooks in one place** ❌ NOT STARTED
**Goal:** `DebugSink` adapter around `OCRVisualizer`

**Current state:** Direct OCRVisualizer calls throughout highres_ocr.py:
- Line 1154: `visualizer = OCRVisualizer(...)`
- Line 1167: `visualizer.save_original(...)`
- Line 1188: `visualizer.save_layout(...)`
- Line 1211: `visualizer.save_preprocessed(...)`
- Line 1234: `visualizer.save_tiles(...)`
- Line 1472: `visualizer.save_results(...)`

**What needs to happen:**
```python
class DebugSink:
    def __init__(self, visualizer: Optional[OCRVisualizer] = None):
        self.visualizer = visualizer

    def save_stage(self, stage: str, image: np.ndarray, **kwargs):
        """Save debug output for a pipeline stage."""
        if self.visualizer is None:
            return

        if stage == "original":
            self.visualizer.save_original(image)
        elif stage == "layout":
            self.visualizer.save_layout(image, **kwargs)
        elif stage == "preprocessed":
            self.visualizer.save_preprocessed(image)
        elif stage == "tiles":
            self.visualizer.save_tiles(image, **kwargs)
        elif stage == "results":
            self.visualizer.save_results(image, **kwargs)
```

**Benefits:**
- Single point of contact with OCRVisualizer
- Easy to disable all debug output
- Can add logging or alternative outputs

---

### 8. **Layout awareness as a policy** ❌ NOT STARTED
**Goal:** Optional `LayoutPolicy` for DPI boost on tables

**Current state:** Layout detection logic scattered:
- `detect_regions()` in [layout_detector.py](pdf_compare/analyzers/layout_detector.py) - Lines 103-320
- DPI boost logic in `highres_ocr()` - Lines 1642-1664
- Overlap detection mixed with rendering

**What needs to happen:**
```python
class LayoutPolicy:
    def __init__(self, layout_detector: LayoutDetector):
        self.detector = layout_detector

    def should_boost_dpi(
        self,
        tile_bbox: Tuple[int, int, int, int],
        regions: List[Dict]
    ) -> bool:
        """Check if tile overlaps table region requiring DPI boost."""
        for region in regions:
            if region["type"] == "table":
                if self._bbox_overlaps(tile_bbox, region["bbox"]):
                    return True
        return False

    def get_boost_factor(self, region_type: str) -> float:
        """Get DPI boost multiplier for region type."""
        if region_type == "table":
            return 1.5  # 50% higher DPI for tables
        return 1.0

    @staticmethod
    def _bbox_overlaps(bbox1, bbox2) -> bool:
        """Check if two bboxes overlap."""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)
```

---

### 9. **Dedup stays as a service** ✓ **ALREADY DONE**
**Goal:** Keep `merge_and_deduplicate()` as standalone service

**Status:** ✓ Already well-implemented
- `merge_and_deduplicate()` in highres_ocr.py - Lines 158-319 (161 lines)
- Uses RapidFuzz with fallback to difflib
- Clean interface, reusable
- No changes needed

---

### 10. **Logging, not prints** ❌ NOT STARTED
**Goal:** Replace all `print(...)` with `logging` module

**Current state:** 50+ print statements in highres_ocr.py:
- Line 1081: `print(f"Opening PDF: {pdf_path} ...")`
- Line 1087: `print(f"Total pages in PDF: {total_pages}")`
- Line 1128: `print(f"[Page {page_num}] Starting tiled OCR...")`
- Lines 1405-1406: `print(f"OCR: Processing {len(tiles)} tiles in parallel...")`
- Many more throughout the file

**What needs to happen:**
```python
import logging

logger = logging.getLogger(__name__)

# Replace all prints:
print(f"Opening PDF: {pdf_path} ...", file=sys.stderr)
# With:
logger.info(f"Opening PDF: {pdf_path}")

print(f"[OCR] Tile {tile.tile_id} generated an exception: {exc}", file=sys.stderr)
# With:
logger.error(f"Tile {tile.tile_id} generated an exception: {exc}", exc_info=True)
```

**Benefits:**
- Consistent log levels (DEBUG, INFO, WARNING, ERROR)
- Easy to configure (file output, log rotation, filtering)
- No more `file=sys.stderr` repetition
- Production-ready logging

---

## Summary Status

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| 1. Unified entry points | ❌ Not started | ~648 lines | Need OCRPage.run() |
| 2. Strategy pattern for engines | ❌ Not started | ~198 lines | Need BaseOCREngine + 3 implementations |
| 3. One coordinate mapper | ❌ Not started | Scattered | Need CoordinateMapper class |
| 4. One renderer | ❌ Not started | Duplicated | Need PDFRenderer class |
| 5. ✓ Single preprocessing | ✓ **COMPLETE** | 334 lines | Ready to integrate |
| 6. Parallelism & memory | ⚠️ Partial | 78 lines | Need TileExecutor wrapper |
| 7. Debug hooks | ❌ Not started | 6 call sites | Need DebugSink adapter |
| 8. Layout policy | ❌ Not started | Mixed in | Need LayoutPolicy class |
| 9. ✓ Dedup service | ✓ Already done | 161 lines | No changes needed |
| 10. Logging migration | ❌ Not started | 50+ prints | Need logging.getLogger() |

**Progress: 1.5/10 components complete** (Preprocessor complete, parallelism partially done)

---

## Recommended Implementation Order

Based on dependencies and impact:

### Phase 1: Foundation (Low risk, high value)
1. ✓ **Preprocessor** - COMPLETE
2. **Integrate Preprocessor** - Use it in highres_ocr.py (~50 lines removed)
3. **Logging migration** - Replace prints with logging (~1 hour)
4. **DebugSink** - Wrap OCRVisualizer (~2 hours)

### Phase 2: Extract Components (Medium risk)
5. **CoordinateMapper** - Extract bbox mapping (~3 hours)
6. **PDFRenderer** - Centralize rendering (~4 hours)
7. **TileExecutor** - Wrap parallelism (~3 hours)
8. **LayoutPolicy** - Extract DPI boost logic (~4 hours)

### Phase 3: Restructure Core (High risk, high reward)
9. **Strategy pattern** - Create BaseOCREngine + implementations (~8 hours)
10. **Unified entry points** - Replace highres_ocr/tiled_ocr with OCRPage.run() (~12 hours)

**Total estimated effort:** ~40-50 hours for complete refactoring

---

## Current Working State

**What works now:**
- All existing OCR functionality (Tesseract, EasyOCR, Qwen-VL)
- Parallel tile processing with auto-calculated workers
- Unified recursive tile splitting
- Layout detection with table validation
- Debug visualization
- Deduplication with fuzzy matching

**Code quality issues:**
- Massive 1720-line highres_ocr.py file
- Duplicate entry points (highres_ocr + tiled_ocr)
- Engine logic not abstracted (hard to add new engines)
- Preprocessing duplicated across engines
- 50+ print statements instead of logging
- No clear separation of concerns

**Next immediate step:**
Based on Phase 1 recommendation: **Integrate Preprocessor into highres_ocr.py** following the checklist in PREPROCESSOR_STATUS.md. This is low-risk and will immediately save ~50 lines while validating the Preprocessor implementation.

---

**Date:** 2025-10-24
**Document:** Refactoring status overview
