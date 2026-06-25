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

# Install Python dependencies from the single source of truth (requirements.txt).
# Kept in its own build stage so the deps layer is cached independently of code.
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# ==============================================================================
# Stage 3: Application layer
# This layer changes FREQUENTLY (every time code changes)
# ==============================================================================
FROM dependencies AS app

# Copy application code
# This is the layer that gets rebuilt when you update your code
COPY pdf_compare/ ./pdf_compare/
COPY ui/ ./ui/
COPY setup_check.py ./

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
