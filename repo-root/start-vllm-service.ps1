# Start vLLM Service on Windows with AMD ROCm GPUs
# This script activates the virtual environment and starts the Qwen inference service

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting Qwen Inference Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to repo directory
Set-Location H:\repo-root\ocr-pdf\repo-root

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv-vllm-py312\Scripts\Activate.ps1

# Load environment variables from .env file
if (Test-Path .env) {
    Write-Host "Loading configuration from .env..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "env:$name" -Value $value
        }
    }
} else {
    Write-Host "WARNING: .env file not found" -ForegroundColor Red
    Write-Host "Please create .env with your HF_TOKEN" -ForegroundColor Red
    exit 1
}

# Set service configuration
$env:VLLM_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
$env:VLLM_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
$env:ENABLE_VISION_OCR = "true"
$env:PORT = "8000"

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Text Model: $env:VLLM_TEXT_MODEL" -ForegroundColor White
Write-Host "  Vision Model: $env:VLLM_VISION_MODEL" -ForegroundColor White
Write-Host "  Port: $env:PORT" -ForegroundColor White
Write-Host "  HF Token: " -NoNewline -ForegroundColor White
if ($env:HF_TOKEN) {
    Write-Host "Set ✓" -ForegroundColor Green
} else {
    Write-Host "NOT SET ✗" -ForegroundColor Red
}
Write-Host ""

# Navigate to service directory
Set-Location docker\vllm-service\app

# Start service
Write-Host "Starting service on http://localhost:8000 ..." -ForegroundColor Cyan
Write-Host "Press CTRL+C to stop" -ForegroundColor Yellow
Write-Host ""

python main_windows.py
