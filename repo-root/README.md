# PDF Compare - Vector Extraction & Revision Diff

Local PDF vector extraction, search, and revision diff tool with OCR support.

## Features

- **Vector Extraction**: Extract geometric shapes and text from PDFs using PyMuPDF
- **Text Search**: Full-text search using SQLite FTS5
- **Revision Diff**: Compare PDF revisions with visual overlays showing changes
- **OCR Support**: High-resolution OCR using Tesseract for scanned documents
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

# OCR augmentation
compare-pdf-revs compare old_id new_id --with-ocr --ocr-mode sparse
```

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

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
