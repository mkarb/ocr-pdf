# Complete Setup Guide - PDF Compare with AI

End-to-end setup guide covering both local development and Docker deployment.

## Table of Contents

1. [Overview](#overview)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Testing & Verification](#testing--verification)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## Overview

PDF Compare is an intelligent PDF analysis and comparison tool with AI-powered features:

- **Vector Extraction**: Fast parallel extraction of lines, curves, fills, and text
- **Raster Comparison**: Grid-based pixel comparison with white space optimization
- **AI/RAG Features**: Symbol recognition, natural language queries, intelligent comparison
- **Database Storage**: PostgreSQL or SQLite with spatial indexing
- **Web UI**: Interactive Streamlit interface with real-time progress

### System Requirements

**Minimum:**
- 8 GB RAM
- 4 CPU cores
- 20 GB free disk space
- Windows 10/11, Linux, or macOS
- Python 3.11+ (local) OR Docker (containerized)

**Recommended:**
- 16 GB RAM
- 8+ CPU cores
- 50 GB free disk space (SSD)
- NVIDIA GPU (optional, for faster AI inference)

---

## Local Development Setup

### Step 1: Install Python Dependencies

```powershell
# Navigate to project
cd repo-root

# Install dependencies
pip install -r requirements.txt
```

**Troubleshooting:**

```powershell
# If scikit-image fails on Windows:
pip install --only-binary scikit-image scikit-image

# If build tools missing:
# Install Visual Studio Build Tools from:
# https://visualstudio.microsoft.com/downloads/
```

### Step 2: Install System Dependencies

**Windows:**

1. **Tesseract OCR**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Run installer
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

**Linux:**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libgeos-dev
```

**macOS:**

```bash
brew install tesseract geos
```

### Step 3: Install Ollama (for AI features)

**Follow detailed guide:** [INSTALL_OLLAMA_WINDOWS.md](INSTALL_OLLAMA_WINDOWS.md)

**Quick version:**

```powershell
# Download and install from https://ollama.com/download/windows

# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Verify
ollama list
```

### Step 4: Verify Installation

```powershell
# Run setup check
python setup_check.py
```

**Expected output:**

```
==================================================================
  PDF COMPARE - SETUP VERIFICATION
==================================================================

[OK]    Python Version            Python 3.11.x
[OK]    PyMuPDF (PDF processing)  Installed
[OK]    Shapely (geometry)        Installed
[OK]    NumPy (numerical)         Installed
[OK]    Streamlit (UI)            Installed
[OK]    OpenCV (raster)           Installed
[OK]    Tesseract (OCR)           Installed
[OK]    SQLAlchemy (database)     Installed
[OK]    LangChain                 Installed
[OK]    ChromaDB                  Installed
[OK]    Ollama Installation       ollama version x.x.x
[OK]    LLM Model (llama3.2)      Available
[OK]    Embed Model               Available

✓ All checks passed! System is ready to use.
```

### Step 5: Test with Sample PDF

```powershell
# Test RAG features
python test_rag.py your_diagram.pdf

# Launch UI
streamlit run ui/streamlit_app.py
```

---

## Docker Deployment

### Step 1: Install Docker

**Windows:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop/
- Install and restart
- Verify: `docker --version`

**Linux:**

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop/
- Install and open

### Step 2: Choose Deployment Type

Three options:

1. **Full Stack** (Recommended): All-in-one with PostgreSQL + Ollama + UI
2. **External Ollama**: UI + PostgreSQL, connect to host Ollama
3. **Standalone**: Single container, SQLite database

### Step 3a: Full Stack Deployment

**Automated (Windows):**

```powershell
.\docker-build-test.ps1 -DeploymentType full
```

**Manual:**

```bash
# Build and start
docker-compose -f docker-compose-full.yml up -d

# Wait for models to download (first run only, ~5-10 minutes)
docker-compose -f docker-compose-full.yml logs -f ollama-init

# Verify services
docker-compose -f docker-compose-full.yml ps

# Run tests
docker exec pdf-compare-ui bash /app/docker_test.sh
```

**Services started:**
- Streamlit UI: http://localhost:8501
- Ollama API: http://localhost:11434
- PostgreSQL: localhost:5432

### Step 3b: External Ollama Deployment

**Prerequisites:**
- Ollama installed on host
- Models pulled: `ollama pull llama3.2 && ollama pull nomic-embed-text`

```bash
# Start Ollama on host
ollama serve

# Start UI + PostgreSQL
docker-compose -f docker-compose-postgres.yml up -d

# Test connection
docker exec pdf-compare-ui curl http://host.docker.internal:11434/api/version
```

### Step 3c: Standalone Deployment

```bash
# Build with embedded Ollama
docker build -f Dockerfile.with-ollama -t pdf-compare-full .

# Run
docker run -d \
  --name pdf-compare-full \
  -p 8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  pdf-compare-full

# Wait for startup
docker logs -f pdf-compare-full
```

### Step 4: Verify Deployment

**See detailed guide:** [DOCKER_BUILD_VERIFICATION.md](DOCKER_BUILD_VERIFICATION.md)

**Quick check:**

```bash
# All services healthy
docker-compose ps

# Models downloaded
docker exec pdf-compare-ollama ollama list

# Run integration tests
docker exec pdf-compare-ui bash /app/docker_test.sh
```

**Expected: 20/20 tests pass**

---

## Testing & Verification

### Local Testing

```powershell
# 1. Setup check
python setup_check.py

# 2. RAG test with PDF
python test_rag.py your_diagram.pdf

# 3. Interactive chat
python pdf_compare/rag_simple.py your_diagram.pdf

# 4. Launch UI
streamlit run ui/streamlit_app.py
```

### Docker Testing

```bash
# 1. Integration tests
docker exec pdf-compare-ui bash /app/docker_test.sh

# 2. RAG test
docker cp your_diagram.pdf pdf-compare-ui:/tmp/test.pdf
docker exec pdf-compare-ui python test_rag.py /tmp/test.pdf

# 3. Interactive chat
docker exec -it pdf-compare-ui python pdf_compare/rag_simple.py /tmp/test.pdf

# 4. Access UI
# http://localhost:8501
```

### Performance Testing

```python
# Test extraction speed (local or Docker)
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
import time

start = time.time()
vm = pdf_to_vectormap_server("diagram.pdf", workers=15)
elapsed = time.time() - start

print(f"Pages: {len(vm.pages)}")
print(f"Time: {elapsed:.2f}s")
print(f"Speed: {len(vm.pages)/elapsed:.2f} pages/sec")
```

**Expected:** 5-7 pages/sec with 15 workers on 16-core CPU

---

## Usage Examples

### 1. Chat with PDF

```python
from pdf_compare.rag_simple import chat_with_pdf

# Load PDF
chat = chat_with_pdf("diagram.pdf")

# Ask questions
print(chat.ask("What symbols are in the legend?"))
print(chat.ask("What is on page 3?"))

# Extract legend
legend = chat.extract_symbol_legend()
print(legend)
```

### 2. Compare Two PDFs

```python
from pdf_compare.rag_simple import compare_pdfs

# Load both
comparator = compare_pdfs("old.pdf", "new.pdf")

# Get comparison
diff = comparator.compare_symbols()
print(diff)
```

### 3. Extract PDF Vectors

```python
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server

# Extract with 15 workers
vm = pdf_to_vectormap_server("diagram.pdf", workers=15)

# Access data
for page in vm.pages:
    print(f"Page {page.page_number}:")
    print(f"  Geometries: {len(page.geoms)}")
    print(f"  Text runs: {len(page.texts)}")
```

### 4. Store in Database

```python
from pdf_compare.db_backend import DatabaseBackend
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
import os

# Connect to database
db = DatabaseBackend(os.environ.get(
    "DATABASE_URL",
    "sqlite:///./data/comparisons.db"
))

# Extract and store
vm = pdf_to_vectormap_server("diagram.pdf", workers=15)
db.store_vectormap(vm)

# Retrieve later
vm2 = db.get_vectormap(vm.meta.doc_id)
```

### 5. Streamlit UI

```bash
# Start UI
streamlit run ui/streamlit_app.py

# Access at http://localhost:8501
```

**Features:**
- Upload PDFs
- Configure worker count (1-16)
- View extraction progress
- Run comparisons
- Export results

---

## Troubleshooting

### Local Issues

**"ollama: command not found"**

```powershell
# Restart PowerShell
# Or check installation:
Get-Command ollama
Test-Path "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
```

**"No module named 'langchain'"**

```powershell
pip install -r requirements.txt
```

**Slow extraction (using only 1 core)**

```python
# Check worker count in UI slider
# Or specify explicitly:
vm = pdf_to_vectormap_server("diagram.pdf", workers=15)
```

### Docker Issues

**Container exits immediately**

```bash
docker logs pdf-compare-ui
# Check for errors in startup

# Common causes:
# - Missing files (requirements.txt, pyproject.toml)
# - Port conflict (8501 already in use)
# - Out of memory
```

**Ollama not responding**

```bash
# Check Ollama is running
docker exec pdf-compare-ollama ps aux | grep ollama

# Check models
docker exec pdf-compare-ollama ollama list

# Restart Ollama
docker restart pdf-compare-ollama
```

**Database connection failed**

```bash
# Check PostgreSQL
docker exec pdf-compare-postgres pg_isready

# Test connection
docker exec pdf-compare-ui python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.environ['DATABASE_URL'])
engine.connect()
print('OK')
"
```

**Out of memory**

```yaml
# Edit docker-compose-full.yml
services:
  pdf-compare-ui:
    mem_limit: 16g  # Increase from 8g
    environment:
      CPU_LIMIT: 8  # Reduce workers from 15
```

### Performance Issues

**Slow LLM responses**

```bash
# Use smaller model
ollama pull llama3.2:1b

# Or in Docker:
docker exec pdf-compare-ollama ollama pull llama3.2:1b

# Edit rag_simple.py to use llama3.2:1b by default
```

**Raster comparison highlighting everything**

- Increase sensitivity (5% → 10%)
- Enable white space skipping (default)
- Check DPI consistency between PDFs

---

## Architecture

```
pdf-compare/
├── pdf_compare/              # Core library
│   ├── pdf_extract_server.py # Multi-core extraction
│   ├── raster_grid_improved.py # Optimized raster diff
│   ├── rag_simple.py         # AI/RAG integration
│   └── db_backend.py         # Database abstraction
├── ui/                       # Streamlit web interface
├── test_rag.py              # RAG test suite
├── setup_check.py           # Setup verification
└── requirements.txt         # Python dependencies

Docker:
├── Dockerfile               # Standard build (external Ollama)
├── Dockerfile.with-ollama   # Self-contained build
├── docker-compose-full.yml  # Complete stack
├── docker-compose-postgres.yml  # External Ollama
└── docker_test.sh           # Integration tests
```

---

## Documentation

### Setup Guides
- **[INSTALL_OLLAMA_WINDOWS.md](INSTALL_OLLAMA_WINDOWS.md)** - Ollama installation
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Docker setup
- **[DOCKER_BUILD_VERIFICATION.md](DOCKER_BUILD_VERIFICATION.md)** - Build testing

### Feature Guides
- **[QUICK_START_RAG.md](QUICK_START_RAG.md)** - RAG quick start
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference
- **[RASTER_COMPARISON_GUIDE.md](RASTER_COMPARISON_GUIDE.md)** - Raster diff

### Technical References
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete features
- **[DATABASE_COMPARISON.md](DATABASE_COMPARISON.md)** - SQLite vs PostgreSQL

---

## Next Steps

### After Local Setup

1. **Test with your PDFs**: `python test_rag.py your_diagram.pdf`
2. **Launch UI**: `streamlit run ui/streamlit_app.py`
3. **Read RAG guide**: See [QUICK_START_RAG.md](QUICK_START_RAG.md)

### After Docker Setup

1. **Access UI**: http://localhost:8501
2. **Run tests**: `docker exec pdf-compare-ui bash /app/docker_test.sh`
3. **Test with PDF**: See [DOCKER_BUILD_VERIFICATION.md](DOCKER_BUILD_VERIFICATION.md)

### For Production

1. **Change passwords**: Edit docker-compose-full.yml
2. **Enable TLS**: Configure reverse proxy (nginx/traefik)
3. **Set up monitoring**: Add Prometheus/Grafana
4. **Configure backups**: See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

---

## Support

- **Local issues**: Run `python setup_check.py`
- **Docker issues**: Run `docker exec pdf-compare-ui bash /app/docker_test.sh`
- **Check logs**: `docker-compose logs -f` or `streamlit run ui/streamlit_app.py`
- **Review docs**: See documentation links above

---

## Quick Reference

### Local Development

```powershell
# Setup
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text

# Verify
python setup_check.py

# Test
python test_rag.py diagram.pdf

# Run UI
streamlit run ui/streamlit_app.py
```

### Docker Deployment

```powershell
# Build and test (Windows)
.\docker-build-test.ps1 -DeploymentType full

# Or manually
docker-compose -f docker-compose-full.yml up -d
docker exec pdf-compare-ui bash /app/docker_test.sh

# Access
# http://localhost:8501
```

### Common Commands

```bash
# Setup check
python setup_check.py

# Test RAG
python test_rag.py your.pdf

# Interactive chat
python pdf_compare/rag_simple.py your.pdf

# Launch UI
streamlit run ui/streamlit_app.py

# Docker logs
docker-compose logs -f

# Docker test
docker exec pdf-compare-ui bash /app/docker_test.sh
```
