# Dockerfile for PDF Compare Server
# Multi-stage build to separate dependencies from application code
# This allows for faster rebuilds when only code changes

# ==============================================================================
# Stage 1: Base image with system dependencies
# This layer changes VERY rarely (only when system packages change)
# ==============================================================================
FROM python:3.11-slim AS base

# Install system dependencies for PyMuPDF, Tesseract OCR, OpenCV, Shapely, and RAG features
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgeos-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    sqlite3 \
    libsqlite3-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==============================================================================
# Stage 2: Dependencies layer
# This layer changes rarely (only when requirements.txt changes)
# ==============================================================================
FROM base AS dependencies

# Copy only requirements files for optimal caching
COPY requirements.txt pyproject.toml README.md ./

# Install Python dependencies in stages to avoid timeouts
# Stage 1: Core dependencies (faster)
RUN pip install --no-cache-dir --timeout=1000 \
    "pymupdf>=1.24.8,<1.28" \
    "shapely>=2.0.2,<2.2" \
    "rtree>=1.2.0,<1.5" \
    "numpy>=1.26,<3" \
    "pillow>=10.3,<12" \
    "typer>=0.12,<1.0" \
    "streamlit>=1.32,<1.51"

# Stage 2: Image processing and OCR
RUN pip install --no-cache-dir --timeout=1000 \
    "opencv-python>=4.8,<5.0" \
    "pytesseract>=0.3.10,<0.4" \
    "scikit-image>=0.21.0,<0.26" \
    "pikepdf>=8.0.0,<10.0"

# Stage 3: Database dependencies
RUN pip install --no-cache-dir --timeout=1000 \
    "sqlalchemy>=2.0.0,<2.1" \
    "alembic>=1.13.0,<1.14" \
    "psycopg2-binary>=2.9.0,<3.0" \
    "asyncpg>=0.29.0,<0.30" \
    "geoalchemy2>=0.14.0,<0.15" \
    "psutil>=5.9.0,<7.0"

# Stage 4: RAG/LLM dependencies (heaviest, install last)
RUN pip install --no-cache-dir --timeout=1000 \
    "langchain>=0.1.0,<0.3" \
    "langchain-community>=0.0.10,<0.3" \
    "langchain-ollama>=0.1.0,<0.2" \
    "pypdf>=3.17.0,<5.0" \
    "tiktoken>=0.5.0,<0.8"

# Stage 5: Vector store and embeddings (very heavy due to PyTorch)
RUN pip install --no-cache-dir --timeout=1000 \
    "chromadb>=0.4.0,<0.6" \
    "sentence-transformers>=2.2.0,<3.0" \
    "faiss-cpu>=1.7.4,<1.9"

# ==============================================================================
# Stage 3: Application layer
# This layer changes FREQUENTLY (every time code changes)
# ==============================================================================
FROM dependencies AS app

# Copy application code
# This is the layer that gets rebuilt when you update your code
COPY pdf_compare/ ./pdf_compare/
COPY ui/ ./ui/
COPY test_rag.py setup_check.py ./

# Create directories for data persistence
RUN mkdir -p /app/data/uploads /app/data/outputs /app/data/db

# Environment variables for server configuration
ENV CPU_LIMIT=4 \
    PDF_MIN_SEGMENT_LEN=0.50 \
    PDF_MIN_FILL_AREA=0.50 \
    PDF_BEZIER_SAMPLES=24 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OLLAMA_HOST=http://host.docker.internal:11434

# Expose ports
# 8000: API server (future)
# 8501: Streamlit UI
EXPOSE 8000 8501

# Health check - verify Streamlit is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command runs Streamlit UI
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]