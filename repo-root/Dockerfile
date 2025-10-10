# Dockerfile for PDF Compare Server
FROM python:3.11-slim

# Install system dependencies for PyMuPDF, Tesseract OCR, and Shapely
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pdf_compare/ ./pdf_compare/
COPY ui/ ./ui/

# Create directories for data persistence
RUN mkdir -p /app/data/uploads /app/data/outputs /app/data/db

# Environment variables for server configuration
ENV CPU_LIMIT=4 \
    PDF_MIN_SEGMENT_LEN=0.50 \
    PDF_MIN_FILL_AREA=0.50 \
    PDF_BEZIER_SAMPLES=24 \
    PYTHONUNBUFFERED=1

# Expose ports
# 8000: API server (future)
# 8501: Streamlit UI
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pdf_compare; print('OK')" || exit 1

# Default command runs Streamlit UI
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]