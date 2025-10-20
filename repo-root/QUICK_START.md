# Quick Start Guide

## One-Command Startup

```powershell
.\start-all.ps1
```

This will:
1. ✅ Start vLLM service (Qwen models on AMD GPU)
2. ✅ Wait for models to load
3. ✅ Start all Docker containers
4. ✅ Open UI in browser

## One-Command Shutdown

```powershell
.\stop-all.ps1
```

---

## Manual Control

### Start Services Individually

**vLLM Service Only**:
```powershell
.\start-all.ps1 -VLLMOnly
```

**Docker Only** (assumes vLLM already running):
```powershell
.\start-all.ps1 -SkipVLLM
```

### Stop Services Individually

**Docker Only** (keep vLLM running):
```powershell
.\stop-all.ps1 -SkipVLLM
```

**vLLM Only** (keep Docker running):
```powershell
.\stop-all.ps1 -VLLMOnly
```

---

## Accessing Services

| Service | URL |
|---------|-----|
| **Streamlit UI** | http://localhost |
| **vLLM API** | http://localhost:8000 |
| **vLLM Health** | http://localhost:8000/health |
| **PostgreSQL** | localhost:5432 |
| **Ollama API** | http://localhost:11434 |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 |

---

## Using Qwen-VL OCR

1. Open UI: http://localhost
2. Upload a PDF (especially scanned engineering drawings)
3. In OCR settings, select engine: **"qwen-vl"**
4. Run OCR
5. Get 95-98% accuracy on technical documents!

---

## Troubleshooting

### vLLM Service Won't Start

```powershell
# Check if it's already running
curl http://localhost:8000/health

# View vLLM window for error messages
# Common issues:
# - HF_TOKEN not set in .env
# - GPU drivers not updated (need v25.20.01.14+)
# - Out of GPU memory (close other apps)
```

### Docker Services Won't Start

```powershell
# Check Docker is running
docker --version

# View logs
docker-compose -f docker-compose-scaled.yml logs -f

# Rebuild if needed
docker-compose -f docker-compose-scaled.yml up -d --build
```

### UI Can't Connect to vLLM

```powershell
# Test from host
curl http://localhost:8000/health

# Test from container
docker exec pdf-compare-ui-1 curl http://host.docker.internal:8000/health

# Check Windows Firewall allows port 8000
```

---

## First-Time Setup

If this is your first time, you need to:

1. **Install Python 3.12** (required for AMD ROCm)
2. **Create virtual environment**:
   ```powershell
   python -m venv venv-vllm-py312
   .\venv-vllm-py312\Scripts\Activate.ps1
   ```
3. **Install dependencies** - see [VLLM_WINDOWS_SETUP.md](docs/VLLM_WINDOWS_SETUP.md)
4. **Set HF_TOKEN in .env**:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

After setup, use `.\start-all.ps1` for all future startups.

---

## Performance Tips

### Faster Startup
- Models are cached after first download (~29GB)
- Subsequent startups: ~2-3 minutes

### Better OCR Accuracy
- Use DPI 500-600 for engineering drawings
- Enable tiled OCR for large pages (A0/A1)
- Use "qwen-vl" engine for scanned documents
- Use "easyocr" or "tesseract" for born-digital PDFs

### Memory Optimization
If you run out of GPU memory:
```powershell
# In .env or before starting:
$env:VLLM_GPU_MEMORY_UTIL="0.75"  # Lower from 0.85
$env:ENABLE_VISION_OCR="false"    # Use text model only
```

---

## Logs and Monitoring

### vLLM Logs
Check the PowerShell window where vLLM is running

### Docker Logs
```powershell
# All services
docker-compose -f docker-compose-scaled.yml logs -f

# Specific service
docker-compose -f docker-compose-scaled.yml logs -f pdf-compare-ui

# Last 100 lines
docker-compose -f docker-compose-scaled.yml logs --tail=100
```

### Database Logs
```powershell
docker-compose -f docker-compose-scaled.yml logs -f postgres
```

---

## Scaling UI Instances

For more concurrent users:

```powershell
# Scale to 5 UI instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=5

# Nginx will load balance across them
```

---

## Help

```powershell
.\start-all.ps1 -Help
.\stop-all.ps1 -Help
```

**Full documentation**: [docs/VLLM_WINDOWS_SETUP.md](docs/VLLM_WINDOWS_SETUP.md)

**Issues**: Check vLLM window and Docker logs first, then create issue on GitHub.
