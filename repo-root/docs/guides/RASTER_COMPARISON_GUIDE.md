# Raster Comparison Guide

## Problem: Over-Sensitive Change Detection

The original raster comparison was highlighting everything because:
1. Too sensitive threshold (18 instead of 30+)
2. Didn't skip empty white space
3. No adaptive thresholding based on document characteristics

## Solution: Improved Raster Comparison

### New Features

#### 1. Adaptive Thresholding
Automatically calculates optimal threshold based on image statistics:
```python
auto_threshold = mean_diff + 2 * std_diff  # Catches outliers, ignores noise
```

#### 2. White Space Detection
Skips cells that are mostly empty (white space):
- Detects content regions (non-white pixels)
- Skips cells with less than 10% content
- Can improve performance by 50-80% on diagram-heavy PDFs

#### 3. Better Sensitivity Controls
- `cell_change_ratio`: Increased from 3% to 5% (less sensitive)
- `threshold`: Manual control for pixel difference detection
- `method`: "adaptive" (recommended), "ssim", "hybrid", or "abs"

### Usage

#### Debug First
```python
from pdf_compare.raster_diff_debug import debug_raster_comparison

debug_raster_comparison("old.pdf", "new.pdf", page_index=0)
# Creates visualization showing why differences are detected
```

#### Use Improved Comparison
```python
from pdf_compare.raster_grid import raster_grid_changed_boxes

boxes, metrics = raster_grid_changed_boxes(
    "old.pdf",
    "new.pdf",
    page_index=0,
    method="adaptive",           # Auto-adjusts threshold
    skip_empty_cells=True,       # Skip white space
    cell_change_ratio=0.05,      # 5% change threshold
    return_metrics=True          # Get diagnostics
)

print(f"Found {len(boxes)} changed regions")
print(f"Skipped {metrics['cells_skipped_empty']} empty cells")
print(f"Efficiency gain: {metrics['efficiency_gain']}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | "adaptive" | Change detection method |
| `cell_change_ratio` | 0.05 | Minimum 5% pixels changed to flag cell |
| `skip_empty_cells` | True | Skip cells with mostly white space |
| `white_threshold` | 250 | Pixel value considered white (0-255) |
| `min_content_ratio` | 0.10 | Minimum 10% content to process cell |
| `threshold` | None | Manual threshold (None=auto) |
| `dpi` | 400 | Rendering resolution |
| `rows` | 12 | Grid rows |
| `cols` | 16 | Grid columns |

### Methods

#### adaptive (Recommended)
- Calculates optimal threshold automatically
- Uses mean + 2*std of differences
- Best for varying document types
- Combines pixel diff + edge detection

#### ssim
- Structural Similarity Index
- Requires scikit-image
- Good for subtle structural changes
- Slower but more accurate

#### hybrid
- Fixed threshold (30) + edge detection
- Good balance of speed and accuracy
- Works well for most documents

#### abs
- Simple absolute difference
- Fastest method
- Use when speed is critical

### Performance Optimization

#### White Space Skipping
For diagrams with 70% white space:
```python
# Before: Process all 192 cells (12x16)
# After: Process only 58 cells (skip 134 empty cells)
# Performance gain: ~70% faster
```

#### Grid Size Optimization
```python
# Fine-grained (slower, more precise)
rows=20, cols=30  # 600 cells

# Balanced (recommended)
rows=12, cols=16  # 192 cells

# Coarse (faster, less precise)
rows=6, cols=8    # 48 cells
```

### Troubleshooting

#### Everything is Highlighted

**Diagnosis**:
```bash
python pdf_compare/raster_diff_debug.py old.pdf new.pdf 0
```

**Common Causes**:
1. Comparing same PDF to itself (should be identical)
2. Threshold too sensitive
3. Alignment issue
4. Different PDF versions (compression artifacts)

**Solutions**:
```python
# Increase threshold
threshold=50  # vs default 30

# Increase cell change ratio
cell_change_ratio=0.10  # 10% instead of 5%

# Use SSIM method
method="ssim"
```

#### Nothing is Highlighted

**Causes**:
1. Threshold too high
2. Cell change ratio too high
3. All cells skipped as empty

**Solutions**:
```python
# Decrease threshold
threshold=20

# Decrease cell change ratio
cell_change_ratio=0.03  # 3% instead of 5%

# Disable white space skipping
skip_empty_cells=False
```

#### Slow Performance

**Solutions**:
```python
# Enable white space skipping
skip_empty_cells=True  # Can give 50-80% speedup

# Use coarser grid
rows=8, cols=10  # Fewer cells to process

# Lower DPI
dpi=300  # vs 400

# Use faster method
method="hybrid"  # vs "ssim"
```

### Metrics Interpretation

```python
boxes, metrics = raster_grid_changed_boxes(..., return_metrics=True)

print(metrics)
{
    'identical': False,
    'alignment': {
        'translation': 0.5,    # pixels shifted
        'rotation': 0.001      # degrees rotated
    },
    'diff_detection': {
        'method': 'adaptive',
        'threshold_used': 42,         # Auto-calculated
        'diff_percentage': 2.5,       # % of pixels different
        'final_change_percentage': 1.8  # After cleanup
    },
    'total_cells': 192,
    'cells_skipped_empty': 134,       # 70% were empty!
    'cells_processed': 58,
    'efficiency_gain': '69.8%',
    'boxes_found': 5
}
```

### Integration with CLI

```bash
# Use improved comparison in CLI
compare-pdf-revs compare-grid old.pdf new.pdf \
    --out-overlay diff.pdf \
    --grid-dpi 400 \
    --grid-rows 12 \
    --grid-cols 16 \
    --grid-ratio 0.05 \
    --method adaptive
```

### Best Practices

1. **Always debug first** on a sample page
2. **Start with defaults** then tune if needed
3. **Use adaptive method** for best results
4. **Enable white space skipping** for diagrams
5. **Lower DPI** if performance is an issue
6. **Check metrics** to understand what's happening

### Examples

#### Engineering Diagrams
```python
# Diagrams have lots of white space and fine details
boxes = raster_grid_changed_boxes(
    old_pdf, new_pdf, 0,
    method="adaptive",
    skip_empty_cells=True,    # Skip white space
    cell_change_ratio=0.05,   # 5% change
    dpi=400,                  # High resolution
    rows=16, cols=20          # Fine grid
)
```

#### Text Documents
```python
# Text documents need different settings
boxes = raster_grid_changed_boxes(
    old_pdf, new_pdf, 0,
    method="hybrid",
    skip_empty_cells=True,    # Skip margins
    cell_change_ratio=0.08,   # Higher threshold
    dpi=300,                  # Lower resolution OK
    rows=10, cols=8           # Coarser grid
)
```

#### Scanned Documents
```python
# Scanned docs have noise and alignment issues
boxes = raster_grid_changed_boxes(
    old_pdf, new_pdf, 0,
    method="ssim",            # Better for noise
    skip_empty_cells=False,   # Scan might have artifacts
    cell_change_ratio=0.10,   # Less sensitive
    threshold=50,             # Higher threshold
    dpi=300                   # Match scan resolution
)
```
