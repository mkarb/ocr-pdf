# vLLM Service for PDF Analysis - AMD ROCm GPU Support
# Runs Qwen2.5 models with high-performance inference
#
# Build: docker build -t pdf-compare-vllm -f docker/vllm-service/Dockerfile .
# Run: See docker-compose-vllm.yml

FROM rocm/pytorch:latest

LABEL maintainer="PDF Compare Project"
LABEL description="vLLM inference service with AMD ROCm support for Qwen2.5 models"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install vLLM and core dependencies
RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.0.0 \
    python-multipart \
    pillow>=10.0.0

# Install Qwen2-VL specific utilities
RUN pip install --no-cache-dir \
    qwen-vl-utils \
    einops \
    torchvision

# Create directory for models cache
RUN mkdir -p /root/.cache/huggingface

# Copy service code
COPY docker/vllm-service/app/ /app/

# Expose vLLM API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run vLLM OpenAI-compatible API server
CMD ["python", "main.py"]
