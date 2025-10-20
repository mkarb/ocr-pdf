# vLLM Service Setup for Windows with AMD ROCm

This guide explains how to run the Qwen2.5 inference service on Windows with AMD Radeon RX 9000/7000 series GPUs.

## Why Windows Native Instead of Docker?

AMD ROCm support in Docker Desktop on Windows is not available. The service must run natively on Windows to access AMD GPUs.

## Architecture

```
┌─────────────────────────────────────────┐
│ Windows Host                             │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │ Qwen Inference Service           │  │
│  │ (Python + PyTorch ROCm)          │  │
│  │ Port: 8000                        │  │
│  │ GPU: AMD RX 9070 XT              │  │
│  └──────────────────────────────────┘  │
│                ↕ HTTP                   │
│  ┌──────────────────────────────────┐  │
│  │ Docker Desktop                    │  │
│  │ - Streamlit UI                    │  │
│  │ - PostgreSQL                      │  │
│  │ - Connects to host:8000           │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Prerequisites

### 1. AMD Drivers

**Required**: AMD Adrenalin driver version **25.20.01.14** or newer

Check your driver version:
```powershell
Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*AMD*"} | Select-Object Name, DriverVersion
```

If needed, download from: https://www.amd.com/en/support

### 2. Python 3.12

PyTorch ROCm 6.4.4 for Windows requires Python 3.12.

Download: https://www.python.org/downloads/release/python-31212/

**Important**: Check "Add Python 3.12 to PATH" during installation

### 3. Hugging Face Token

Create a free account and get a token:
1. Sign up at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens
3. Create a new token (Read access is sufficient)
4. Add to `.env` file:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

## Installation

### 1. Create Virtual Environment

```powershell
cd H:\repo-root\ocr-pdf\repo-root

# Create venv with Python 3.12
python -m venv venv-vllm-py312

# Activate
.\venv-vllm-py312\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install PyTorch with ROCm

```powershell
# Install PyTorch 2.8 with ROCm 6.4.4 (Windows-specific wheels)
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl
pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl

# Verify GPU detection
python -c "import torch; print('ROCm Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output**:
```
ROCm Available: True
GPU Count: 2
GPU: AMD Radeon RX 9070 XT
```

### 3. Install Dependencies

```powershell
# Core dependencies
pip install transformers>=4.40.0
pip install accelerate>=0.28.0
pip install fastapi>=0.109.0
pip install "uvicorn[standard]>=0.27.0"
pip install pydantic>=2.0.0
pip install python-multipart
pip install pillow>=10.0.0
pip install qwen-vl-utils
pip install einops
```

## Running the Service

### Option 1: Using Startup Script (Recommended)

```powershell
# Run the startup script
.\start-vllm-service.ps1
```

The script will:
1. Activate the virtual environment
2. Load configuration from `.env`
3. Start the service on port 8000

### Option 2: Manual Start

```powershell
# Activate venv
cd H:\repo-root\ocr-pdf\repo-root
.\venv-vllm-py312\Scripts\Activate.ps1

# Set environment variables
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
$env:VLLM_TEXT_MODEL="Qwen/Qwen2.5-7B-Instruct"
$env:VLLM_VISION_MODEL="Qwen/Qwen2-VL-7B-Instruct"
$env:ENABLE_VISION_OCR="true"
$env:PORT="8000"

# Start service
cd docker\vllm-service\app
python main_windows.py
```

### First Startup

The first time you run the service, it will download models (~29GB total):
- Qwen2.5-7B-Instruct: ~14GB
- Qwen2-VL-7B-Instruct: ~15GB

Download time: 10-20 minutes (depending on internet speed)

Models are cached in `C:\Users\<username>\.cache\huggingface\` and won't be re-downloaded.

### Expected Output

```
INFO - Initializing Qwen Inference Service (Windows/Transformers)...
INFO -   Text Model: Qwen/Qwen2.5-7B-Instruct
INFO -   Vision Model: Qwen/Qwen2-VL-7B-Instruct
INFO -   Device: cuda
INFO -   GPU Count: 2
INFO -   GPU 0: AMD Radeon RX 9070 XT
INFO - Loading text model: Qwen/Qwen2.5-7B-Instruct...
INFO - ✓ Text model ready
INFO - Loading vision model: Qwen/Qwen2-VL-7B-Instruct...
INFO - ✓ Vision model ready
INFO - Qwen Inference Service is READY
INFO - Uvicorn running on http://0.0.0.0:8000
```

## Testing the Service

Open a new PowerShell window and test:

```powershell
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "text_model_loaded": true,
#   "vision_model_loaded": true,
#   "gpu_available": true,
#   "gpu_count": 2,
#   "device": "cuda"
# }
```

## Starting Docker Services

Once the vLLM service is running, start the Docker services:

```powershell
# In a NEW PowerShell window (keep vLLM service running)
cd H:\repo-root\ocr-pdf\repo-root

# Start Docker services
docker-compose -f docker-compose-scaled.yml up -d

# Check logs
docker-compose -f docker-compose-scaled.yml logs -f pdf-compare-ui
```

The Streamlit UI will connect to `http://host.docker.internal:8000` to access the vLLM service running on Windows.

## Stopping the Services

```powershell
# Stop Docker services
docker-compose -f docker-compose-scaled.yml down

# Stop vLLM service (in the PowerShell window where it's running)
# Press CTRL+C
```

## Troubleshooting

### GPU Not Detected

**Symptom**: `ROCm Available: False`

**Solution**:
1. Update AMD drivers to version 25.20.01.14+
2. Reboot Windows
3. Verify driver: `Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*AMD*"}`

### Out of Memory Error

**Symptom**: `CUDA out of memory`

**Solutions**:
1. Close other GPU-intensive applications
2. Reduce GPU memory usage:
   ```powershell
   $env:VLLM_GPU_MEMORY_UTIL="0.75"  # Lower from default 0.85
   ```
3. Use only text model (disable vision):
   ```powershell
   $env:ENABLE_VISION_OCR="false"
   ```

### Models Download Slowly

**Solution**: Use a download accelerator or wait. Models are cached after first download.

### Service Won't Start - "Module not found"

**Symptom**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Virtual environment not activated
```powershell
.\venv-vllm-py312\Scripts\Activate.ps1
python --version  # Should show Python 3.12.x
```

### Docker Can't Connect to Service

**Symptom**: `Connection refused to host.docker.internal:8000`

**Solutions**:
1. Verify service is running: `curl http://localhost:8000/health`
2. Check Windows Firewall allows port 8000
3. Restart Docker Desktop

## Performance

### Expected Speeds (RX 9070 XT 16GB)

| Task | Speed | Accuracy |
|------|-------|----------|
| Text inference (1 page) | 2-5 sec | N/A |
| Vision OCR (A1 scanned page) | 6-10 sec | 95-98% |
| Batch inference (10 pages) | 15-30 sec | N/A |

### Memory Usage

- Text model only: ~8GB VRAM
- Text + Vision models: ~16GB VRAM total
- Recommended: 16GB+ GPU memory

## API Endpoints

The service provides three endpoints:

### 1. Health Check
```
GET /health
```

### 2. Text Query
```
POST /api/v1/query
Body: {
  "prompt": "Analyze this layout...",
  "temperature": 0.1,
  "max_tokens": 512
}
```

### 3. Vision OCR
```
POST /api/v1/ocr
Body: {
  "image_base64": "base64_encoded_image_data",
  "focus_technical": true,
  "min_confidence": 0.5
}
```

## Auto-Start on Windows Boot (Optional)

To automatically start the service when Windows boots:

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: "When the computer starts"
4. Action: "Start a program"
5. Program: `powershell.exe`
6. Arguments: `-File "H:\repo-root\ocr-pdf\repo-root\start-vllm-service.ps1"`

## Updating Models

To update to newer Qwen models:

```powershell
$env:VLLM_TEXT_MODEL="Qwen/Qwen2.5-14B-Instruct"  # Larger model
$env:VLLM_VISION_MODEL="Qwen/Qwen2-VL-7B-Instruct"
```

Models will be downloaded on first use.

## Support

For issues:
- AMD ROCm: https://rocm.docs.amd.com/
- Qwen Models: https://huggingface.co/Qwen
- Project Issues: https://github.com/mkarb/ocr-pdf/issues
