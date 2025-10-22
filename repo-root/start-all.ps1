# Complete Startup Script for PDF Compare + vLLM Service
# This script starts both the Windows vLLM service and Docker containers

param(
    [switch]$SkipVLLM,
    [switch]$VLLMOnly,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host ""
    Write-Host "PDF Compare Startup Script" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start-all.ps1           - Start everything (vLLM + Docker)"
    Write-Host "  .\start-all.ps1 -VLLMOnly - Start only vLLM service"
    Write-Host "  .\start-all.ps1 -SkipVLLM - Start only Docker containers"
    Write-Host "  .\start-all.ps1 -Help     - Show this help"
    Write-Host ""
    exit 0
}

if ($Help) {
    Show-Help
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " PDF Compare Complete Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to repo directory
$RepoRoot = "H:\repo-root\ocr-pdf\repo-root"
Set-Location $RepoRoot

function Stop-VllmService {
    $vllmProcesses = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*main_windows.py*"
    }

    if ($vllmProcesses) {
        Write-Host "  Stopping existing vLLM process..." -ForegroundColor Yellow
        foreach ($proc in $vllmProcesses) {
            try {
                Stop-Process -Id $proc.Id -Force
                Write-Host "    Stopped PID $($proc.Id)" -ForegroundColor Green
            }
            catch {
                Write-Host "    Failed to stop PID $($proc.Id)" -ForegroundColor Red
            }
        }
        Start-Sleep -Seconds 2
    }
    else {
        Write-Host "  No running vLLM process detected" -ForegroundColor Gray
    }
}

function Start-VllmService {
    param(
        [string]$RepoRoot
    )

    Write-Host ""
    Write-Host "  Starting new vLLM service instance..." -ForegroundColor Yellow

    $vllmScriptPath = Join-Path $RepoRoot "docker\vllm-service\app\main_windows.py"
    $pythonExe = Join-Path $RepoRoot "venv-vllm-py312\Scripts\python.exe"

    if (-not (Test-Path $pythonExe)) {
        Write-Host "ERROR: Python venv not found at: $pythonExe" -ForegroundColor Red
        exit 1
    }

    $cmd = "Set-Location '$RepoRoot'; "
    $cmd += "`$env:HF_TOKEN='$env:HF_TOKEN'; "
    $cmd += "`$env:VLLM_TEXT_MODEL='$env:VLLM_TEXT_MODEL'; "
    $cmd += "`$env:VLLM_VISION_MODEL='$env:VLLM_VISION_MODEL'; "
    $cmd += "`$env:ENABLE_VISION_OCR='$env:ENABLE_VISION_OCR'; "
    $cmd += "`$env:PORT='$env:PORT'; "
    $cmd += "Write-Host 'Starting Qwen Service...' -ForegroundColor Cyan; "
    $cmd += "& '$pythonExe' '$vllmScriptPath'"

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd

    Write-Host "  Waiting for vLLM service (max 5 min)..." -ForegroundColor Yellow

    $maxWait = 300
    $waited = 0
    $healthy = $false
    $health = $null

    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 5
        $waited += 5

        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            $health = $response.Content | ConvertFrom-Json

            if ($health.status -eq "healthy" -and $health.text_model_loaded -eq $true -and $health.vision_model_loaded -eq $true) {
                $healthy = $true
                break
            }
        }
        catch {
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
    }

    Write-Host ""

    if ($healthy) {
        Write-Host "  vLLM service started successfully - Models loaded:" -ForegroundColor Green
        Write-Host "    Text Model:   Qwen2.5-7B-Instruct (GPU)" -ForegroundColor Gray
        Write-Host "    Vision Model: Qwen2-VL-7B-Instruct (GPU)" -ForegroundColor Gray
        Write-Host "    GPU Count:    $($health.gpu_count)" -ForegroundColor Gray
        return $true
    }
    else {
        Write-Host "  WARNING: vLLM not responding after $maxWait seconds" -ForegroundColor Yellow
        Write-Host "  Press Enter to continue or CTRL+C to abort..." -ForegroundColor Yellow
        Read-Host | Out-Null
        return $false
    }
}

# ============================================================================
# Step 1: Start vLLM Service
# ============================================================================

if (-not $SkipVLLM) {
    Write-Host "[1/4] Starting vLLM Service..." -ForegroundColor Yellow
    Write-Host "------------------------------" -ForegroundColor DarkGray

    # Check if .env exists
    if (-not (Test-Path .env)) {
        Write-Host "ERROR: .env file not found!" -ForegroundColor Red
        Write-Host "Please create .env with your HF_TOKEN" -ForegroundColor Red
        exit 1
    }

    # Load environment variables from .env
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "env:$name" -Value $value
        }
    }

    # Set service configuration
    $env:VLLM_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    $env:VLLM_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    $env:ENABLE_VISION_OCR = "true"
    $env:PORT = "8000"

    Write-Host "  Configuration loaded" -ForegroundColor Gray

    # Check if vLLM service is already running
    $vllmRunning = $false
    $startVllm = $false
    $restartExisting = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $health = $response.Content | ConvertFrom-Json

        # Verify models are loaded
        if ($health.status -eq "healthy" -and $health.text_model_loaded -eq $true -and $health.vision_model_loaded -eq $true) {
            Write-Host ""
            Write-Host "  vLLM service already running - Models loaded:" -ForegroundColor Green
            Write-Host "    Text Model:   $($health.text_model_loaded)" -ForegroundColor Gray
            Write-Host "    Vision Model: $($health.vision_model_loaded)" -ForegroundColor Gray
            Write-Host "    GPU Count:    $($health.gpu_count)" -ForegroundColor Gray
            Write-Host ""
            $vllmRunning = $true
        } else {
            Write-Host ""
            Write-Host "  vLLM service found but models not loaded, restarting..." -ForegroundColor Yellow
            Write-Host ""
            $startVllm = $true
            $restartExisting = $true
        }
    }
    catch {
        $startVllm = $true
    }

    if ($startVllm) {
        if ($restartExisting) {
            Stop-VllmService
        }

        $vllmRunning = Start-VllmService -RepoRoot $RepoRoot

        if (-not $vllmRunning) {
            Write-Host ""
        }
    }

    Write-Host ""
}

if ($VLLMOnly) {
    Write-Host "vLLM service is running." -ForegroundColor Green
    Write-Host "Access at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# ============================================================================
# Step 2: Check Docker
# ============================================================================

Write-Host "[2/4] Checking Docker..." -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor DarkGray

try {
    $dockerServerVersion = docker info --format '{{.ServerVersion}}'

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($dockerServerVersion)) {
        throw "Docker daemon not reachable"
    }

    $dockerServerVersion = $dockerServerVersion.Trim()
    Write-Host "  Docker daemon reachable (Server version: $dockerServerVersion)" -ForegroundColor Green
}
catch {
    Write-Host "  ERROR: Docker daemon not running or unreachable" -ForegroundColor Red
    exit 1
}

Write-Host ""

# ============================================================================
# Step 3: Start Docker Services
# ============================================================================

Write-Host "[3/4] Starting Docker Services..." -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor DarkGray

# Check if already running
$runningContainers = docker ps --filter "name=pdf-compare" --format "{{.Names}}"

if ($runningContainers) {
    Write-Host "  Services already running" -ForegroundColor Yellow
    Write-Host "  Stop them first? (y/N): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host

    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "  Stopping..." -ForegroundColor Yellow
        docker-compose -f docker-compose-scaled.yml down
        Write-Host ""
    }
}

Write-Host "  Starting services..." -ForegroundColor Yellow
docker-compose -f docker-compose-scaled.yml up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Docker services started" -ForegroundColor Green
}
else {
    Write-Host "  ERROR: Docker failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Wait a bit
Start-Sleep -Seconds 10

# ============================================================================
# Step 4: Open UI
# ============================================================================

Write-Host "[4/4] Opening UI..." -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor DarkGray

$maxWait = 30
$waited = 0
$uiReady = $false

while ($waited -lt $maxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $uiReady = $true
        break
    }
    catch {
        Start-Sleep -Seconds 2
        $waited += 2
    }
}

if ($uiReady) {
    Write-Host "  Opening browser..." -ForegroundColor Yellow
    Start-Process "http://localhost"
    Write-Host "  UI opened" -ForegroundColor Green
}
else {
    Write-Host "  UI not ready, open manually: http://localhost" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Startup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor White
Write-Host "  UI:        http://localhost" -ForegroundColor Cyan
Write-Host "  vLLM API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop: .\stop-all.ps1" -ForegroundColor Yellow
Write-Host ""
