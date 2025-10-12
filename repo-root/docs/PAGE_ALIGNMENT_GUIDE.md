# Page Alignment Guide

## Overview

The Page Alignment feature enables intelligent comparison of PDF documents where pages have been inserted, deleted, or reordered between versions. Instead of simple 1:1 page number matching, the system uses **content-based fingerprinting** to align pages correctly.

## Problem Statement

### Without Page Alignment (Old Behavior)

```
Old Document: [A, B, C, D, E]  (5 pages)
New Document: [A, B, X, C, D, E, F]  (7 pages - X inserted after B, F added at end)

Simple comparison (page number matching):
  Page 1 vs 1: A vs A ✓ (correct match)
  Page 2 vs 2: B vs B ✓ (correct match)
  Page 3 vs 3: C vs X ✗ (WRONG - compares unrelated pages)
  Page 4 vs 4: D vs C ✗ (WRONG)
  Page 5 vs 5: E vs D ✗ (WRONG)
  Pages 6-7 in new document are ignored!

Result: Massive false positives showing C, D, E as "completely changed"
```

### With Page Alignment (New Behavior)

```
Content-based alignment:
  Old 1 → New 1: A vs A ✓ (matched, similarity: 100%)
  Old 2 → New 2: B vs B ✓ (matched, similarity: 100%)
  None  → New 3: X is INSERTED
  Old 3 → New 4: C vs C ✓ (matched, similarity: 100%)
  Old 4 → New 5: D vs D ✓ (matched, similarity: 100%)
  Old 5 → New 6: E vs E ✓ (matched, similarity: 100%)
  None  → New 7: F is INSERTED

Result: Accurate detection of 2 new pages, no false changes on existing pages
```

## How It Works

### 1. Page Fingerprinting

Each page is analyzed to create a content fingerprint:

- **Text hash**: SHA-256 of all text content (order-preserved)
- **Text sample**: First 500 characters for quick comparison
- **Geometry hash**: SHA-256 of all vector bounding boxes
- **Element counts**: Number of text runs and geometry elements

### 2. Similarity Scoring

Pages are compared using a weighted scoring system:

- **Text exact match**: 60% weight
  - Identical text hash = full score
  - Partial text overlap = proportional score
- **Geometry match**: 20% weight
  - Identical geometry hash = full score
  - Similar element count = proportional score
- **Text sample overlap**: 20% weight
  - Word-level set overlap

**Score range**: 0.0 (completely different) to 1.0 (identical)

### 3. Alignment Algorithms

#### Greedy Algorithm (Faster)

- For each old page, find the best matching new page
- Greedy selection (first match wins)
- **Time complexity**: O(n × m) where n, m are page counts
- **Best for**: Quick comparisons, documents with mostly sequential changes

#### Dynamic Programming Algorithm (Optimal)

- Uses Needleman-Wunsch sequence alignment
- Considers global optimum across all pages
- Handles gaps (insertions/deletions) optimally
- **Time complexity**: O(n × m) with backtracking
- **Best for**: Complex reorderings, maximum accuracy

## Usage

### Python API

#### Basic Usage

```python
import sqlite3
from pdf_compare.page_alignment import align_pages
from pdf_compare.compare import diff_documents_aligned

# Connect to database
conn = sqlite3.connect("vectormap.sqlite")

# Align pages between documents
alignments = align_pages(
    conn,
    old_id="doc_v1",
    new_id="doc_v2",
    method="dynamic",  # or "greedy"
    similarity_threshold=0.5  # 50% minimum similarity
)

# Print alignment results
for old_pg, new_pg, score in alignments:
    if old_pg and new_pg:
        print(f"Page {old_pg} → {new_pg} (similarity: {score:.2%})")
    elif old_pg:
        print(f"Page {old_pg} was DELETED")
    else:
        print(f"Page {new_pg} was INSERTED")
```

#### Aligned Comparison

```python
# Perform comparison using aligned pages
diffs, alignments = diff_documents_aligned(
    conn,
    old_id="doc_v1",
    new_id="doc_v2",
    alignment_method="dynamic",
    similarity_threshold=0.5
)

# Process diffs
for diff in diffs:
    page_num = diff["page"]
    alignment_info = diff.get("alignment", {})

    status = alignment_info.get("status")
    if status == "matched":
        old_pg = alignment_info["old_page"]
        new_pg = alignment_info["new_page"]
        sim = alignment_info["similarity"]
        print(f"Page {old_pg}→{new_pg} (sim: {sim:.2%})")
        print(f"  Added geometries: {len(diff['geometry']['added'])}")
        print(f"  Removed geometries: {len(diff['geometry']['removed'])}")
    elif status == "inserted":
        print(f"Page {page_num} INSERTED")
    elif status == "deleted":
        print(f"Page {page_num} DELETED")
```

#### Get Alignment Summary

```python
from pdf_compare.page_alignment import get_alignment_summary

summary = get_alignment_summary(alignments)
print(f"Total alignments: {summary['total_alignments']}")
print(f"Matched pages: {summary['matched_pages']}")
print(f"Deleted pages: {summary['deleted_pages']}")
print(f"Inserted pages: {summary['inserted_pages']}")
print(f"Average similarity: {summary['average_similarity']:.2%}")
```

### CLI Usage

```bash
# Compare with automatic page alignment
compare-pdf-revs compare old.pdf new.pdf --aligned --method dynamic

# Compare with custom similarity threshold
compare-pdf-revs compare old.pdf new.pdf --aligned --similarity-threshold 0.7

# Show alignment information only (no diff)
compare-pdf-revs align old.pdf new.pdf --method greedy
```

### Streamlit UI

The Streamlit UI includes an "Enable Page Alignment" checkbox in the comparison section:

1. Upload and ingest both PDF versions
2. Select old and new documents
3. ✓ Check "Enable Page Alignment"
4. Click "Compare"
5. View alignment visualization showing:
   - Matched pages with similarity scores
   - Inserted pages (green badge)
   - Deleted pages (red badge)
   - Page mapping diagram

## Configuration

### Similarity Threshold

Controls how similar pages must be to match:

- **0.3-0.4**: Very lenient - matches pages with major differences
- **0.5** (default): Balanced - good for most cases
- **0.6-0.7**: Strict - only matches very similar pages
- **0.8+**: Very strict - near-identical content required

**Recommendation**: Start with 0.5, increase if you get false matches.

### Algorithm Selection

#### Use Greedy When:
- Documents are large (50+ pages)
- Changes are mostly sequential (insertions at specific points)
- Speed is critical
- Content is highly distinctive (low chance of false matches)

#### Use Dynamic When:
- Documents have complex reorderings
- Accuracy is critical
- Pages have similar content (risk of false matches with greedy)
- Documents are moderate size (<50 pages)

## Advanced Use Cases

### Handling Page Splits

When one page in the old document becomes two pages in the new:

```python
# The alignment will show:
Old 5 → New 7 (similarity: 0.8)  # Partial match
None → New 8 (inserted)           # Rest of content

# Manual verification needed
```

### Handling Page Merges

When two pages merge into one:

```python
# The alignment will show:
Old 5 → New 7 (similarity: 0.6)
Old 6 → None (deleted)            # Actually merged, not deleted

# Manual verification recommended
```

### Custom Fingerprinting

For specialized documents, you can extend the fingerprinting:

```python
from pdf_compare.page_alignment import _compute_page_fingerprint

def custom_fingerprint(conn, doc_id, page):
    base = _compute_page_fingerprint(conn, doc_id, page)

    # Add custom metrics
    # Example: Page dimensions, image count, table detection
    # ...

    return base
```

## Performance

### Computational Cost

| Document Size | Greedy | Dynamic | Memory |
|--------------|--------|---------|--------|
| 10 pages | <1s | <1s | <10MB |
| 50 pages | 2-3s | 3-5s | ~50MB |
| 100 pages | 5-8s | 10-15s | ~100MB |
| 500 pages | 30-60s | 120-180s | ~500MB |

**Note**: Fingerprinting is the main bottleneck (database reads + hashing).

### Optimization Tips

1. **Cache fingerprints**: Store in database for repeated comparisons
2. **Parallel fingerprinting**: Use multiprocessing for large documents
3. **Incremental updates**: Only recompute fingerprints for changed pages
4. **Use greedy for large docs**: 3-5x faster with minor accuracy trade-off

## Limitations

### What Page Alignment CANNOT Handle:

1. **Significant content changes**: If >50% of page content changes, alignment may fail
2. **Blank pages**: Hard to distinguish between blank pages
3. **Identical repeated pages**: May align incorrectly (e.g., cover pages, templates)
4. **Rotated pages**: Current implementation doesn't detect rotation
5. **Language changes**: Translating document breaks text matching

### Workarounds:

- **Manual hints**: Provide page number hints for problematic pages
- **Hybrid approach**: Use alignment for most pages, manual for edge cases
- **Iterative refinement**: Start with low threshold, manually verify, adjust

## Troubleshooting

### Problem: Pages Not Matching

**Symptoms**: Low similarity scores for pages that should match

**Solutions**:
1. Lower similarity threshold to 0.3-0.4
2. Check if pages have different headers/footers
3. Verify both PDFs were ingested correctly
4. Try raster comparison for heavily reformatted pages

### Problem: False Matches

**Symptoms**: Pages with different content showing as matched

**Solutions**:
1. Increase similarity threshold to 0.6-0.7
2. Switch to dynamic algorithm
3. Add custom fingerprinting for domain-specific features
4. Manual review of low-confidence matches (<0.6 similarity)

### Problem: Slow Performance

**Symptoms**: Alignment takes >1 minute for <100 pages

**Solutions**:
1. Switch to greedy algorithm
2. Enable fingerprint caching
3. Check database indexes are created
4. Reduce text sample size in fingerprinting

## Examples

### Example 1: Technical Manual Revision

```
Old: 45 pages
New: 52 pages (added troubleshooting section, updated diagrams)

Alignment result:
- 40 matched pages (similarity: 0.95-1.0)
- 3 matched pages with updates (similarity: 0.7-0.8)
- 2 deleted pages (old specifications)
- 9 inserted pages (new troubleshooting section)

Processing time: 3.5 seconds (dynamic)
```

### Example 2: Legal Contract

```
Old: 28 pages
New: 31 pages (added clauses, reordered sections)

Alignment result:
- 25 matched pages (similarity: 0.98-1.0)
- 3 inserted pages (new clauses)
- Detected reordering: Old page 15 → New page 18

Processing time: 1.2 seconds (dynamic)
```

### Example 3: Specification Sheet

```
Old: 5 pages
New: 8 pages (added 3 product variants)

Alignment result:
- 5 matched pages (similarity: 0.85-1.0)
- 3 inserted pages (new variants)

Processing time: 0.3 seconds (greedy)
```

## API Reference

### `align_pages()`

```python
def align_pages(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    method: str = "dynamic",
    similarity_threshold: float = 0.5
) -> List[Tuple[Optional[int], Optional[int], float]]
```

**Returns**: List of `(old_page, new_page, similarity_score)` tuples

### `diff_documents_aligned()`

```python
def diff_documents_aligned(
    conn: sqlite3.Connection,
    old_id: str,
    new_id: str,
    alignment_method: str = "dynamic",
    similarity_threshold: float = 0.5
) -> Tuple[List[Dict], List[Tuple[Optional[int], Optional[int], float]]]
```

**Returns**: Tuple of `(diffs, alignments)`

### `get_alignment_summary()`

```python
def get_alignment_summary(
    alignments: List[Tuple[Optional[int], Optional[int], float]]
) -> dict
```

**Returns**: Summary dict with counts and statistics

## Best Practices

1. **Always review alignment first**: Check alignment results before assuming diffs are correct
2. **Use appropriate threshold**: Start with 0.5, adjust based on document type
3. **Validate edge cases**: Manually verify pages with low similarity scores
4. **Document assumptions**: Note which algorithm and threshold you used
5. **Combine with raster**: Use raster comparison for heavily reformatted pages
6. **Test on sample**: Try on a few pages first before full document

## Future Enhancements

Potential improvements for future versions:

- **GPU acceleration**: Faster fingerprinting with CUDA
- **Machine learning**: Learn optimal thresholds from user feedback
- **Rotation detection**: Handle rotated pages
- **Image fingerprinting**: Use perceptual hashing for diagram-heavy pages
- **Interactive alignment**: UI for manual page matching hints
- **Confidence intervals**: Provide uncertainty estimates
- **Batch processing**: Align multiple document pairs simultaneously

## Support

For issues or questions:
- Check troubleshooting section above
- Review alignment summary for insights
- Open GitHub issue with example PDFs (if possible)
- Include alignment output and similarity scores
