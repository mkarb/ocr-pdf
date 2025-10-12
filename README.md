# PDF Compare - Engineering Diagram Analysis & Comparison

Intelligent PDF comparison and analysis tool for engineering diagrams with AI-powered symbol recognition.

## Features

### Core Capabilities
- **Vector Extraction**: Fast parallel extraction of lines, curves, fills, and text from PDFs
- **Raster Comparison**: Grid-based pixel comparison with adaptive thresholding and white space skipping
- **Database Storage**: PostgreSQL or SQLite backend with spatial indexing
- **Streamlit UI**: Interactive web interface with real-time progress tracking
- **Multi-core Processing**: Configurable parallel processing (up to 16 workers)

### AI-Powered Analysis (RAG Integration)
- **Symbol Recognition**: Identify symbols regardless of size using LLM understanding
- **Legend Extraction**: Automatically extract symbol definitions from diagrams
- **Natural Language Queries**: Ask questions about PDFs in plain English
- **Intelligent Comparison**: Understand semantic changes, not just geometric differences

## Quick Start

### 1. Install Dependencies

```powershell
cd repo-root
pip install -r requirements.txt
```

### 2. Install Ollama (for AI features)

Follow the detailed guide: [INSTALL_OLLAMA_WINDOWS.md](repo-root/docs/setup/INSTALL_OLLAMA_WINDOWS.md)

Quick version:
```powershell
# Download and install from https://ollama.com/download/windows
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Test the System

```powershell
# Test RAG/AI features
cd repo-root
python test_rag.py your_diagram.pdf

# Run Streamlit UI
streamlit run ui/streamlit_app.py
```

## Usage Examples

### Basic PDF Comparison (UI)
```powershell
streamlit run ui/streamlit_app.py
```
Then upload two PDF versions and configure comparison settings.

### AI-Powered Analysis (Python)
```python
from pdf_compare.rag_simple import chat_with_pdf

# Chat with a PDF
chat = chat_with_pdf("diagram.pdf")
print(chat.ask("What symbols are in the legend?"))

# Extract symbol legend
legend = chat.extract_symbol_legend()

# Find symbol instances
print(chat.find_symbol_instances("Main Control Valve"))
```

### Compare Two PDFs with AI
```python
from pdf_compare.rag_simple import compare_pdfs

comparator = compare_pdfs("old_diagram.pdf", "new_diagram.pdf")
print(comparator.compare_symbols())
```

### Interactive CLI Chat
```powershell
python repo-root/pdf_compare/rag_simple.py diagram.pdf
```

## Architecture

```
ocr-pdf/
├── repo-root/
│   ├── pdf_compare/           # Core extraction and comparison
│   │   ├── pdf_extract_server.py    # Multi-core PDF extraction
│   │   ├── raster_grid_improved.py  # Optimized raster comparison
│   │   ├── rag_simple.py            # AI/RAG integration
│   │   └── db_backend.py            # Database abstraction
│   ├── ui/                    # Streamlit web interface
│   ├── test_rag.py           # Test suite for AI features
│   └── requirements.txt       # Python dependencies
```

## Configuration

### Environment Variables
```bash
# CPU/Performance
CPU_LIMIT=15                    # Max worker processes
PDF_MIN_SEGMENT_LEN=0.50       # Min line segment length
PDF_BEZIER_SAMPLES=24          # Bezier curve sampling

# Database
DATABASE_URL=postgresql://user:pass@host:5432/pdfcompare
# or
DATABASE_URL=sqlite:///./data/comparisons.db

# Ollama
OLLAMA_HOST=http://localhost:11434
```

### Streamlit UI Configuration
- **Worker Processes**: Adjustable slider (1-16 cores)
- **Debug Mode**: Shows extraction metrics and progress
- **DPI Settings**: 300-600 for raster comparison
- **Grid Size**: 5-50 pixels for change detection

## Documentation

📚 **[Complete Documentation Index](repo-root/docs/INDEX.md)**

### Quick Links

**Getting Started:**
- **[Complete Setup Guide](repo-root/docs/setup/COMPLETE_SETUP_GUIDE.md)** - End-to-end setup for local and Docker
- [Install Ollama (Windows)](repo-root/docs/setup/INSTALL_OLLAMA_WINDOWS.md) - Step-by-step Ollama installation

**Deployment:**
- [Docker Deployment Guide](repo-root/docs/deployment/DOCKER_DEPLOYMENT.md) - Complete Docker deployment
- [Docker Build Verification](repo-root/docs/deployment/DOCKER_BUILD_VERIFICATION.md) - Build testing checklist

**Features:**
- [RAG Quick Start](repo-root/docs/guides/QUICK_START_RAG.md) - AI features in 3 lines of code
- [Raster Comparison Guide](repo-root/docs/guides/RASTER_COMPARISON_GUIDE.md) - Improved pixel comparison
- [Streamlit Features](repo-root/docs/guides/STREAMLIT_FEATURES.md) - Web UI guide

**Reference:**
- [Quick Reference](repo-root/docs/reference/QUICK_REFERENCE.md) - Commands and troubleshooting
- [Implementation Summary](repo-root/docs/reference/IMPLEMENTATION_SUMMARY.md) - Technical details
- [Database Comparison](repo-root/docs/reference/DATABASE_COMPARISON.md) - SQLite vs PostgreSQL

## Performance

### Extraction Speed
- **Single-core**: ~0.5 pages/sec (typical)
- **Multi-core** (15 workers): ~5-7 pages/sec on 16-core system
- **Raster comparison**: 50-80% faster with white space skipping

### Resource Requirements
- **RAM**: 2-4 GB per worker process
- **Storage**: ~2 GB for Ollama models (llama3.2 + nomic-embed-text)
- **CPU**: Benefits from 4+ cores

## Docker Deployment

```powershell
# Build and run with PostgreSQL
docker-compose -f docker-compose-postgres.yml up

# Access UI at http://localhost:8501
```

## Troubleshooting

### Ollama Not Found
```powershell
# Restart PowerShell after installation
ollama --version

# Check if service is running
Get-Process ollama
```

### RAG Tests Failing
```powershell
# Verify models are pulled
ollama list

# Should show:
# llama3.2:latest
# nomic-embed-text:latest
```

### Performance Issues
- Increase worker count in UI slider
- Check CPU usage (should be near 100% during extraction)
- Verify SSD for temporary files (not HDD)

### Raster Comparison Highlighting Everything
- Increase sensitivity threshold (5% → 10%)
- Enable white space skipping
- Check DPI consistency between PDFs

## System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.11+
- **RAM**: 8 GB minimum, 16 GB recommended
- **CPU**: 4+ cores recommended
- **Storage**: 10 GB free (includes models)

## License

See LICENSE file for details.

## Support

- **Issues**: Report bugs or feature requests on GitHub
- **Ollama**: https://github.com/ollama/ollama
- **LangChain**: https://python.langchain.com/