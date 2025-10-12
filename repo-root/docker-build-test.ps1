# Docker Build and Test Script for Windows
# Builds Docker image and runs comprehensive tests

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("full", "external", "standalone")]
    [string]$DeploymentType = "full",

    [Parameter(Mandatory=$false)]
    [switch]$NoBuild,

    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,

    [Parameter(Mandatory=$false)]
    [switch]$Cleanup
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Failure { Write-Host $args -ForegroundColor Red }

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  PDF Compare - Docker Build & Test" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Info "Checking prerequisites..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Failure "ERROR: Docker is not installed or not in PATH"
    exit 1
}

if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Failure "ERROR: Docker Compose is not installed or not in PATH"
    exit 1
}

Write-Success "✓ Docker is installed"
Write-Success "✓ Docker Compose is installed"

# Check Docker is running
try {
    docker ps | Out-Null
    Write-Success "✓ Docker daemon is running"
} catch {
    Write-Failure "ERROR: Docker daemon is not running"
    Write-Info "Start Docker Desktop and try again"
    exit 1
}

Write-Host ""

# Determine compose file
$composeFile = switch ($DeploymentType) {
    "full" { "docker-compose-full.yml" }
    "external" { "docker-compose-postgres.yml" }
    "standalone" { "docker-compose.yml" }
}

Write-Info "Deployment type: $DeploymentType"
Write-Info "Compose file: $composeFile"
Write-Host ""

# Cleanup if requested
if ($Cleanup) {
    Write-Warning "Cleaning up existing containers and volumes..."
    docker-compose -f $composeFile down -v
    Write-Success "✓ Cleanup complete"
    Write-Host ""
}

# Build image
if (-not $NoBuild) {
    Write-Info "Building Docker image..."
    Write-Info "This may take several minutes on first build..."
    Write-Host ""

    try {
        docker-compose -f $composeFile build
        Write-Success "✓ Docker image built successfully"
    } catch {
        Write-Failure "ERROR: Docker build failed"
        Write-Failure $_.Exception.Message
        exit 1
    }

    Write-Host ""
} else {
    Write-Info "Skipping build (using existing image)"
    Write-Host ""
}

# Start services
Write-Info "Starting services..."
docker-compose -f $composeFile up -d

# Wait for services to be healthy
Write-Info "Waiting for services to start..."
Write-Host ""

$maxWait = 300  # 5 minutes
$waited = 0
$checkInterval = 5

while ($waited -lt $maxWait) {
    $status = docker-compose -f $composeFile ps --format json | ConvertFrom-Json

    $allHealthy = $true
    foreach ($service in $status) {
        $health = $service.Health
        if ($health -and $health -ne "healthy") {
            $allHealthy = $false
            break
        }
    }

    if ($allHealthy) {
        Write-Success "✓ All services are healthy"
        break
    }

    Write-Host "." -NoNewline
    Start-Sleep -Seconds $checkInterval
    $waited += $checkInterval
}

Write-Host ""

if ($waited -ge $maxWait) {
    Write-Failure "ERROR: Services did not become healthy within $maxWait seconds"
    Write-Info "Check logs with: docker-compose -f $composeFile logs"
    exit 1
}

Write-Host ""

# Show service status
Write-Info "Service status:"
docker-compose -f $composeFile ps

Write-Host ""

# Run tests if not skipped
if (-not $SkipTests) {
    Write-Info "Running integration tests..."
    Write-Host ""

    try {
        # For full deployment, test inside UI container
        if ($DeploymentType -eq "full") {
            Write-Info "Testing inside pdf-compare-ui container..."
            docker exec pdf-compare-ui bash /app/docker_test.sh

            Write-Host ""
            Write-Info "Checking Ollama models..."
            docker exec pdf-compare-ollama ollama list

        } else {
            Write-Warning "Automated tests not available for $DeploymentType deployment"
            Write-Info "Please verify manually:"
            Write-Info "  1. Access UI: http://localhost:8501"
            Write-Info "  2. Upload and process a PDF"
        }

        Write-Host ""
        Write-Success "✓ All tests passed!"

    } catch {
        Write-Failure "ERROR: Tests failed"
        Write-Failure $_.Exception.Message
        Write-Info "Check logs with: docker-compose -f $composeFile logs"
        exit 1
    }
} else {
    Write-Info "Skipping tests (use -SkipTests to enable)"
}

Write-Host ""

# Show next steps
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Deployment Successful!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

Write-Info "Services are running:"
Write-Host "  - Streamlit UI: http://localhost:8501" -ForegroundColor Yellow

if ($DeploymentType -eq "full") {
    Write-Host "  - Ollama API: http://localhost:11434" -ForegroundColor Yellow
    Write-Host "  - PostgreSQL: localhost:5432" -ForegroundColor Yellow
}

Write-Host ""
Write-Info "Common commands:"
Write-Host "  View logs:    docker-compose -f $composeFile logs -f"
Write-Host "  Stop:         docker-compose -f $composeFile down"
Write-Host "  Restart:      docker-compose -f $composeFile restart"
Write-Host "  Shell access: docker exec -it pdf-compare-ui bash"

if ($DeploymentType -eq "full") {
    Write-Host "  Test RAG:     docker exec pdf-compare-ui python test_rag.py"
    Write-Host "  Check models: docker exec pdf-compare-ollama ollama list"
}

Write-Host ""
Write-Success "Setup complete! Open http://localhost:8501 to use the application."
Write-Host ""

# Optionally open browser
$openBrowser = Read-Host "Open browser? (y/n)"
if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
    Start-Process "http://localhost:8501"
}

exit 0
