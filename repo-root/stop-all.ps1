# Stop Script for PDF Compare + vLLM Service
# This script stops both Docker containers and the vLLM service

param(
    [switch]$SkipVLLM,
    [switch]$VLLMOnly,
    [switch]$Help
)

function Show-Help {
    Write-Host ""
    Write-Host "PDF Compare Stop Script" -ForegroundColor Cyan
    Write-Host "=======================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\stop-all.ps1           - Stop everything (vLLM + Docker)"
    Write-Host "  .\stop-all.ps1 -VLLMOnly - Stop only vLLM service"
    Write-Host "  .\stop-all.ps1 -SkipVLLM - Stop only Docker containers"
    Write-Host "  .\stop-all.ps1 -Help     - Show this help"
    Write-Host ""
    exit 0
}

if ($Help) {
    Show-Help
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Stopping PDF Compare Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to repo directory
$RepoRoot = "H:\repo-root\ocr-pdf\repo-root"
Set-Location $RepoRoot

# ============================================================================
# Stop Docker Services
# ============================================================================

if (-not $VLLMOnly) {
    Write-Host "[1/2] Stopping Docker Services..." -ForegroundColor Yellow
    Write-Host "------------------------------" -ForegroundColor DarkGray

    try {
        docker-compose -f docker-compose-scaled.yml down

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Docker services stopped ✓" -ForegroundColor Green
        } else {
            Write-Host "  Warning: Docker stop had issues (code: $LASTEXITCODE)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "  ERROR: Failed to stop Docker services" -ForegroundColor Red
        Write-Host "  $_" -ForegroundColor Red
    }

    Write-Host ""
}

# ============================================================================
# Stop vLLM Service
# ============================================================================

if (-not $SkipVLLM) {
    Write-Host "[2/2] Stopping vLLM Service..." -ForegroundColor Yellow
    Write-Host "------------------------------" -ForegroundColor DarkGray

    # Find Python processes running main_windows.py
    $vllmProcesses = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*main_windows.py*"
    }

    if ($vllmProcesses) {
        Write-Host "  Found vLLM service process(es):" -ForegroundColor Gray
        $vllmProcesses | ForEach-Object {
            Write-Host "    PID: $($_.Id)" -ForegroundColor Gray
        }

        Write-Host ""
        Write-Host "  Stopping processes..." -ForegroundColor Yellow

        $vllmProcesses | ForEach-Object {
            try {
                Stop-Process -Id $_.Id -Force
                Write-Host "    Stopped PID $($_.Id) ✓" -ForegroundColor Green
            }
            catch {
                Write-Host "    Failed to stop PID $($_.Id) ✗" -ForegroundColor Red
            }
        }
    } else {
        # Check if service is accessible (might be running without CommandLine info)
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            Write-Host "  Warning: vLLM service is responding but process not found" -ForegroundColor Yellow
            Write-Host "  Please manually close the vLLM PowerShell window" -ForegroundColor Yellow
        }
        catch {
            Write-Host "  vLLM service not running ✓" -ForegroundColor Green
        }
    }

    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Services Stopped" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify everything is stopped
Write-Host "Verification:" -ForegroundColor Yellow
Write-Host ""

# Check Docker
$runningContainers = docker ps --filter "name=pdf-compare" --format "{{.Names}}" 2>$null
if ($runningContainers) {
    Write-Host "  Docker: Still running" -ForegroundColor Red
    $runningContainers | ForEach-Object {
        Write-Host "    - $_" -ForegroundColor Red
    }
} else {
    Write-Host "  Docker: Stopped ✓" -ForegroundColor Green
}

# Check vLLM
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  vLLM:   Still running" -ForegroundColor Red
}
catch {
    Write-Host "  vLLM:   Stopped ✓" -ForegroundColor Green
}

Write-Host ""
