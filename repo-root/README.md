# PDF Compare - Vector Extraction & Revision Diff

Local PDF vector extraction, search, and revision diff tool with OCR support.

## Features

- **Vector Extraction**: Extract geometric shapes and text from PDFs using PyMuPDF
- **Text Search**: Full-text search using PostgreSQL full-text search (tsvector/GIN)
- **Revision Diff**: Compare PDF revisions with visual overlays showing changes
- **OCR Support**: High-resolution tiled OCR using EasyOCR (GPU, auto-detected) or Tesseract, for scanned/large drawings
- **Raster Comparison**: Grid-based and pixel-level comparison with alignment
- **Streamlit UI**: Interactive web interface for document management
- **CLI Tools**: Command-line interface for automation

## Installation

```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.11+
- Tesseract OCR (for OCR features)
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Linux: `apt-get install tesseract-ocr tesseract-ocr-eng`
  - macOS: `brew install tesseract`

## Usage

### Command Line

```bash
# Ingest a PDF
compare-pdf-revs ingest document.pdf

# Search text
compare-pdf-revs search-text "search term"

# Compare documents
compare-pdf-revs compare old_id new_id --out-overlay diff.pdf

# Compare PDF appearance directly, including scans (no database required)
compare-pdf-revs compare-grid old.pdf new.pdf --out-overlay visual_diff.pdf

# Compare block positions and sizes without comparing wording (no database required)
compare-pdf-revs compare-layout old.pdf new.pdf --out-overlay layout_diff.pdf

# OCR augmentation
compare-pdf-revs compare old_id new_id --with-ocr --ocr-mode sparse
```

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

In **Compare & Create Overlay**, choose **Visual (raster)** for scanned PDFs,
images, and drawings, **Vector and text** for extracted shapes and text, or
**Layout** for content block positions and sizes.
Visual comparison aligns scan shifts before detecting changed regions. Vector
comparison reports moved text and changes to shape coordinates.

Layout comparison detects moved, resized, added, and removed text, image, and
drawing blocks, plus page dimensions and rotation changes. It ignores wording
changes when block bounds remain the same; text reflow that changes those bounds
can count as a layout change. Scanned pages use visual regions without requiring
OCR. Scan noise can affect region boundaries. Position and size tolerances
default to 3 PDF points (72 points = 1 inch), adjustable in the UI or with
`--position-tolerance` and `--size-tolerance`. Layout overlays show additions in
green, removals in red, moves in orange, and resizing in purple.

All methods compare pages by page number and include added or removed trailing
pages in the overlay. They do not automatically match reordered or inserted
pages. The CLI uses the revised PDF as the overlay base unless `--base-pdf` is
provided; the base must be one of the two input PDFs. Visual comparison defaults
to detecting any changed pixels above its pixel threshold so small revisions on
large sheets are retained. Increase **Minimum changed area per grid cell (%)**
in the UI or `--grid-ratio` in the CLI to filter scan noise. Higher values can
also filter small revisions. CLI grid density uses `--grid-rows` and `--grid-cols`.

## Verification

```bash
python -m pytest -q
```

The suite covers geometry and text matching, visual and layout comparison, rotated pages,
and generated PDF overlays. To also exercise PostgreSQL ingestion, CLI export,
and Streamlit comparison/download state, set `PDF_COMPARE_TEST_DATABASE_URL` to
a disposable PostgreSQL database and run the same command. These integration
tests remove their own test documents afterward.

## Docker

```bash
docker-compose up
```

## Documentation

See additional documentation:
- [Docker Quick Start](DOCKER_QUICKSTART.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Server Mode Comparison](SERVER_MODE_COMPARISON.md)
- [Server Mode README](SERVER_MODE_README.md)
