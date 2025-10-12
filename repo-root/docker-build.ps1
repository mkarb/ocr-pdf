# Simple Docker Build Script for Windows
# Optimized for building the PDF Compare image with heavy dependencies

param(
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  PDF Compare - Docker Image Build" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check Docker is running
try {
    docker ps | Out-Null
    Write-Host "✓ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker daemon is not running" -ForegroundColor Red
    Write-Host "Start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Cyan
Write-Host "This may take 10-15 minutes on first build due to heavy dependencies:" -ForegroundColor Yellow
Write-Host "  - PyTorch (via sentence-transformers)" -ForegroundColor Yellow
Write-Host "  - ChromaDB" -ForegroundColor Yellow
Write-Host "  - LangChain ecosystem" -ForegroundColor Yellow
Write-Host ""

$buildArgs = @(
    "build",
    "--progress=plain",
    "--network=host",
    "-t", "pdf-compare:latest",
    "-f", "Dockerfile",
    "."
)

if ($Verbose) {
    Write-Host "Build command: docker $($buildArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""
}

# Set environment for better build performance
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

try {
    & docker @buildArgs

    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "  Build Complete!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Image: pdf-compare:latest" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To start the full stack:" -ForegroundColor Yellow
    Write-Host "  docker-compose -f docker-compose-full.yml up -d" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use the test script:" -ForegroundColor Yellow
    Write-Host "  .\docker-build-test.ps1 -DeploymentType full" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  Build Failed!" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Docker daemon not running - Start Docker Desktop" -ForegroundColor White
    Write-Host "  2. Insufficient disk space - Free up at least 10GB" -ForegroundColor White
    Write-Host "  3. Network timeout - Check internet connection" -ForegroundColor White
    Write-Host "  4. Memory limit - Increase Docker memory to 8GB+" -ForegroundColor White
    Write-Host ""
    Write-Host "To see detailed logs, run with -Verbose flag" -ForegroundColor Yellow
    exit 1
}

exit 0
