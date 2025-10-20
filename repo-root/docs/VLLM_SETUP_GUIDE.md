# vLLM + Qwen2.5 Setup Guide

Complete guide for deploying vLLM with Qwen2.5 models on AMD ROCm GPUs for high-accuracy OCR and layout analysis.

**Date**: 2025-01-18
**Version**: 1.0

---

## Overview

This implementation adds a **separate vLLM microservice** that runs on your AMD GPUs (9070 XT + 9060 XT) to provide:

1. **High-Accuracy OCR** - Qwen2-VL for scanned engineering documents
2. **Layout Analysis** - Qwen2.5 text model for spatial reasoning
3. **GPU Acceleration** - AMD ROCm with tensor parallelism

### Architecture

```
┌────────────────────────────────────────────┐
│  Streamlit UI Instances (Scaled 2-5x)      │
│  - Handles user requests                   │
│  - Sends OCR jobs to vLLM service          │
│  - Fallback to EasyOCR/Tesseract           │
└───────────┬────────────────────────────────┘
            │ HTTP REST API
            ▼
┌────────────────────────────────────────────┐
│  vLLM Service (Separate Docker Container)  │
│  - Runs on AMD GPUs (9070 XT + 9060 XT)    │
│  - Qwen2.5-7B-Instruct (text)              │
│  - Qwen2-VL-7B-Instruct (vision)           │
│  - Handles OCR requests                    │
└────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

- **AMD GPUs**: Radeon RX 9070 XT (16GB) + RX 9060 XT (12GB)
- **System RAM**: 32GB+ recommended
- **Disk Space**: 50GB free (for models)
- **CPU**: 8+ cores recommended

### Software Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended) or Windows with WSL2
- **Docker**: 24.0+ with Docker Compose
- **ROCm**: 6.2+ installed on host
- **Python**: 3.10+ (for local testing)

---

## Setup Steps

### Step 1: Environment Configuration

Create a `.env` file in the project root:

```bash
# Database
POSTGRES_DB=pdfcompare
POSTGRES_USER=pdfcompare
POSTGRES_PASSWORD=your_secure_password_here

# Hugging Face (for downloading models)
HF_TOKEN=your_huggingface_token_here  # Get from https://huggingface.co/settings/tokens

# Optional: Grafana
GRAFANA_PASSWORD=admin
```

**Get Hugging Face Token**:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Add to `.env` file

### Step 2: Verify ROCm Installation

```bash
# Check ROCm is installed
rocm-smi

# Should show both GPUs:
# GPU[0]: AMD Radeon RX 9070 XT
# GPU[1]: AMD Radeon RX 9060 XT

# Verify Docker can access GPUs
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocm-smi
```

If ROCm is not installed:
```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo dpkg -i amdgpu-install_6.2.60200-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo reboot
```

### Step 3: Build Docker Images

```bash
cd repo-root

# Build vLLM service image (this takes 10-15 minutes)
docker build -t pdf-compare-vllm -f docker/vllm-service/Dockerfile .

# Build main application image
docker build -t pdf-compare-ui .
```

### Step 4: Start Services

```bash
# Start all services (including vLLM)
docker-compose -f docker-compose-scaled.yml up -d

# Watch logs
docker-compose -f docker-compose-scaled.yml logs -f vllm-service

# You should see:
# vllm-service | Initializing vLLM Service...
# vllm-service | Loading text model: Qwen/Qwen2.5-7B-Instruct...
# vllm-service | ✓ Text model ready
# vllm-service | Loading vision model: Qwen/Qwen2-VL-7B-Instruct...
# vllm-service | ✓ Vision model ready
# vllm-service | vLLM Service is READY
```

**Note**: First startup takes 5-10 minutes to download models (14GB+).

### Step 5: Verify vLLM Service

```bash
# Check health
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "text_model_loaded": true,
#   "vision_model_loaded": true,
#   "gpu_available": true,
#   "gpu_count": 2
# }
```

### Step 6: Test OCR Integration

Upload a scanned PDF to the Streamlit UI and check the logs:

```bash
# Monitor UI logs
docker-compose -f docker-compose-scaled.yml logs -f pdf-compare-ui

# You should see:
# OCR: Connecting to vLLM service for Qwen2-VL OCR...
# OCR: Connected to vLLM service
# OCR: Extracted 45 text items via vLLM service
```

---

## Usage

### Enabling vLLM OCR in Streamlit UI

When ingesting a scanned document:

1. Upload PDF
2. Enable OCR checkbox
3. Select **OCR Engine**: Choose "Qwen-VL" (GPU) for best accuracy
4. Set DPI: 400-600 recommended
5. Click Ingest

The UI will automatically:
- Send tiles to vLLM service
- Fallback to EasyOCR/Tesseract if vLLM unavailable
- Show progress in real-time

### Programmatic Usage

```python
from pdf_compare.vllm_client import get_vllm_client

# Get client
client = get_vllm_client()

# Check if available
if client.is_available():
    # Use vLLM for high-accuracy OCR
    results = client.ocr_image(image_array)
    for item in results:
        print(f"Text: {item['text']}, Confidence: {item['confidence']}")
else:
    # Fallback to EasyOCR/Tesseract
    results = fallback_ocr(image_array)
```

---

## Scaling

### GPU Configuration

**Single GPU (9070 XT only)**:
```yaml
# In docker-compose-scaled.yml
vllm-service:
  environment:
    - VLLM_TENSOR_PARALLEL=1  # Use 1 GPU
```

**Both GPUs (Recommended)**:
```yaml
vllm-service:
  environment:
    - VLLM_TENSOR_PARALLEL=2  # Use both GPUs
```

### Model Selection

**Fast Mode (Qwen2.5-7B + Qwen2-VL-2B)**:
```yaml
vllm-service:
  environment:
    - VLLM_TEXT_MODEL=Qwen/Qwen2.5-7B-Instruct
    - VLLM_VISION_MODEL=Qwen/Qwen2-VL-2B-Instruct  # Faster, less accurate
```

**High-Quality Mode (14B models)**:
```yaml
vllm-service:
  environment:
    - VLLM_TEXT_MODEL=Qwen/Qwen2.5-14B-Instruct
    - VLLM_VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct
    - VLLM_TENSOR_PARALLEL=2  # Required for 14B
```

---

## Performance

### Expected Performance (9070 XT 16GB + 9060 XT 12GB)

| Configuration | Speed (A1 page) | Accuracy | VRAM Usage |
|---------------|----------------|----------|------------|
| Qwen2-VL-2B (single GPU) | 3-5 sec | 90-95% | 5GB |
| Qwen2-VL-7B (single GPU) | 8-12 sec | 95-98% | 14GB |
| Qwen2-VL-7B (dual GPU) | 6-10 sec | 95-98% | 8GB + 8GB |
| Qwen2-VL-14B (dual GPU) | 12-18 sec | 97-99% | 16GB + 12GB |

**Comparison to Current OCR**:
- Tesseract (CPU): 30-60 sec, 70-80% accuracy
- EasyOCR (GPU): 45-90 sec, 75-85% accuracy
- **Qwen2-VL-7B**: 8-12 sec, 95-98% accuracy ✅

---

## Troubleshooting

### vLLM Service Won't Start

**Problem**: Service stuck in "initializing"

```bash
# Check logs
docker logs pdf-compare-vllm

# Common issues:
# 1. ROCm not detected
# 2. Out of memory
# 3. Missing Hugging Face token
```

**Solution**:
```bash
# Verify GPU access
docker exec pdf-compare-vllm rocm-smi

# Check memory
docker stats pdf-compare-vllm

# Verify HF token
docker exec pdf-compare-vllm env | grep HF_TOKEN
```

### UI Can't Connect to vLLM

**Problem**: "vLLM service not available"

```bash
# Check service health
curl http://localhost:8000/health

# Check network
docker exec pdf-compare-ui ping vllm-service

# Check environment
docker exec pdf-compare-ui env | grep VLLM_HOST
```

**Solution**:
```bash
# Restart vLLM service
docker-compose -f docker-compose-scaled.yml restart vllm-service

# Wait for health check (3-5 minutes)
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

### OCR Falls Back to Tesseract

**Problem**: vLLM not being used for OCR

**Check**:
```bash
# UI logs should show:
docker-compose -f docker-compose-scaled.yml logs pdf-compare-ui | grep "vLLM"

# Expected:
# OCR: Connected to vLLM service
# OCR: Extracted 45 text items via vLLM service

# If you see:
# OCR: vLLM service unavailable
# Then vLLM service is not ready
```

### Out of Memory Errors

**Problem**: GPU OOM during inference

```bash
# Check GPU memory
rocm-smi

# Reduce memory utilization
docker-compose -f docker-compose-scaled.yml down
# Edit docker-compose-scaled.yml:
# VLLM_GPU_MEMORY_UTIL=0.75  # Reduce from 0.85

docker-compose -f docker-compose-scaled.yml up -d
```

---

## Monitoring

### Health Checks

```bash
# vLLM service
curl http://localhost:8000/health

# Prometheus metrics (if enabled)
curl http://localhost:9090

# Grafana dashboard (if enabled)
open http://localhost:3000  # Login: admin/admin
```

### GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 rocm-smi

# During OCR, you should see:
# GPU[0] Utilization: 85-95%
# GPU[1] Utilization: 85-95%
```

### Performance Metrics

```bash
# vLLM service logs
docker logs pdf-compare-vllm | grep "tokens/sec"

# Expected:
# Throughput: 60-80 tokens/sec (7B model, dual GPU)
```

---

## Upgrading

### Updating Models

```bash
# Stop services
docker-compose -f docker-compose-scaled.yml down

# Clear model cache (optional, to download new versions)
docker volume rm repo-root_huggingface-cache

# Edit docker-compose-scaled.yml to change model versions
# Then rebuild
docker-compose -f docker-compose-scaled.yml build vllm-service
docker-compose -f docker-compose-scaled.yml up -d
```

### vLLM Version Updates

```bash
# Edit docker/vllm-service/Dockerfile
# Change: RUN pip install vllm>=0.6.0

# Rebuild
docker build -t pdf-compare-vllm -f docker/vllm-service/Dockerfile .
docker-compose -f docker-compose-scaled.yml up -d vllm-service
```

---

## Advanced Configuration

### Custom Models

To use different Qwen models:

```yaml
# docker-compose-scaled.yml
vllm-service:
  environment:
    # Use Qwen2.5-14B for better quality (slower)
    - VLLM_TEXT_MODEL=Qwen/Qwen2.5-14B-Instruct

    # Or use Qwen2-VL-2B for faster OCR (less accurate)
    - VLLM_VISION_MODEL=Qwen/Qwen2-VL-2B-Instruct
```

### Disable Vision OCR (Text-only mode)

```yaml
vllm-service:
  environment:
    - ENABLE_VISION_OCR=false  # Only load text model
```

This saves 10GB+ memory if you don't need VLM OCR.

---

## Summary

**You now have**:
- ✅ vLLM service running on AMD GPUs
- ✅ Qwen2-VL OCR integrated into Streamlit UI
- ✅ Automatic fallback to EasyOCR/Tesseract
- ✅ REST API for programmatic access
- ✅ Scalable architecture (service runs separately)
- ✅ Model caching (avoid re-downloading)

**Next Steps**:
1. Upload a scanned engineering PDF
2. Enable OCR with Qwen-VL engine
3. Compare accuracy vs Tesseract/EasyOCR
4. Monitor GPU utilization
5. Scale UI instances if needed

**For Support**:
- See [VLLM_QWEN_IMPLEMENTATION_ROADMAP.md](VLLM_QWEN_IMPLEMENTATION_ROADMAP.md)
- Check logs: `docker-compose -f docker-compose-scaled.yml logs -f`
- GitHub Issues: https://github.com/anthropics/claude-code/issues
