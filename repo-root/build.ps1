# PDF Compare - Docker Build and Deployment Script (PowerShell)
# Windows equivalent of build.sh

param(
    [Parameter(Position=0)]
    [string]$Mode = "help",

    [Parameter(Position=1)]
    [string]$Action = "up",

    [Parameter(Position=2)]
    [string]$Options = ""
)

$ErrorActionPreference = "Continue"

# Color functions
function Write-Log {
    param([string]$Message)
    Write-Host "[PDF-COMPARE] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARNING] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

# Check Docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

switch ($Mode) {
    "standalone" {
        Write-Log "Building standalone container (all-in-one)"
        docker-compose build pdf-compare-standalone

        if ($Action -eq "up") {
            Write-Log "Starting standalone container..."
            docker-compose up -d pdf-compare-standalone
            Write-Info "Access UI at: http://localhost:8501"
        }
    }

    "full" {
        Write-Log "Building full deployment (Postgres + Ollama + UI)"
        docker-compose -f docker-compose-full.yml build

        if ($Action -eq "up") {
            Write-Log "Starting full deployment..."
            docker-compose -f docker-compose-full.yml up -d
            Write-Info "Access UI at: http://localhost:8501"
            Write-Info "Postgres: localhost:5432"
            Write-Info "Ollama: localhost:11434"
        }
    }

    "infra" {
        Write-Log "Building infrastructure only (Postgres + Ollama)"
        Write-Info "Infrastructure containers use pre-built images, no build needed"

        if ($Action -eq "up") {
            Write-Log "Starting infrastructure services..."
            docker-compose -f docker-compose-full.yml up -d postgres ollama

            Write-Log "Waiting for services to be healthy..."
            docker-compose -f docker-compose-full.yml up -d ollama-init

            Write-Info "Infrastructure ready!"
            Write-Info "Postgres: localhost:5432"
            Write-Info "Ollama: localhost:11434"
        }
        elseif ($Action -eq "down") {
            Write-Log "Stopping infrastructure services..."
            docker-compose -f docker-compose-full.yml stop postgres ollama ollama-init
            Write-Info "Infrastructure stopped (data preserved in volumes)"
        }
    }

    "postgres" {
        Write-Log "Managing PostgreSQL container only"
        Write-Info "PostgreSQL uses pre-built image: postgis/postgis:16-3.4"

        if ($Action -eq "up") {
            Write-Log "Starting PostgreSQL..."
            docker-compose -f docker-compose-full.yml up -d postgres
            Write-Log "Waiting for PostgreSQL to be healthy..."
            Start-Sleep -Seconds 3
            docker-compose -f docker-compose-full.yml ps postgres
            Write-Info "Postgres: localhost:5432"
        }
        elseif ($Action -eq "down") {
            Write-Log "Stopping PostgreSQL..."
            docker-compose -f docker-compose-full.yml stop postgres
            Write-Info "PostgreSQL stopped (data preserved in postgres-data volume)"
        }
        elseif ($Action -eq "restart") {
            Write-Log "Restarting PostgreSQL..."
            docker-compose -f docker-compose-full.yml restart postgres
            Write-Info "PostgreSQL restarted"
        }
    }

    "ollama" {
        Write-Log "Managing Ollama container only"
        Write-Info "Ollama uses pre-built image: ollama/ollama:latest"

        if ($Action -eq "up") {
            Write-Log "Starting Ollama..."
            docker-compose -f docker-compose-full.yml up -d ollama
            Write-Log "Waiting for Ollama to be healthy..."
            Start-Sleep -Seconds 3

            Write-Log "Initializing models..."
            docker-compose -f docker-compose-full.yml up -d ollama-init
            Write-Info "Ollama: localhost:11434"
            Write-Info "Models: llama3.2, nomic-embed-text"
        }
        elseif ($Action -eq "down") {
            Write-Log "Stopping Ollama..."
            docker-compose -f docker-compose-full.yml stop ollama ollama-init
            Write-Info "Ollama stopped (models preserved in ollama-data volume)"
        }
        elseif ($Action -eq "restart") {
            Write-Log "Restarting Ollama..."
            docker-compose -f docker-compose-full.yml restart ollama
            Write-Info "Ollama restarted"
        }
    }

    "app" {
        Write-Log "Building application container only (UI + PDF Compare)"

        if ($Action -eq "build") {
            Write-Log "Building application image..."
            docker-compose -f docker-compose-full.yml build pdf-compare-ui
            Write-Info "Application image built successfully"
        }
        elseif ($Action -eq "up") {
            Write-Log "Building and starting application container..."
            docker-compose -f docker-compose-full.yml build pdf-compare-ui
            docker-compose -f docker-compose-full.yml up -d pdf-compare-ui
            Write-Info "Access UI at: http://localhost:8501"
        }
        elseif ($Action -eq "down") {
            Write-Log "Stopping application container..."
            docker-compose -f docker-compose-full.yml stop pdf-compare-ui
            Write-Info "Application stopped"
        }
        elseif ($Action -eq "restart") {
            Write-Log "Restarting application container..."
            docker-compose -f docker-compose-full.yml restart pdf-compare-ui
            Write-Info "Application restarted"
        }
    }

    "multi-user" {
        Write-Log "Building multi-user containers (UI + worker)"
        docker-compose --profile multi-user build

        if ($Action -eq "up") {
            Write-Log "Starting multi-user deployment..."
            docker-compose --profile multi-user up -d
            Write-Info "Access UI at: http://localhost:8501"
            $workerCount = (docker-compose ps | Select-String "worker").Count
            Write-Info "Workers: $workerCount"
        }
    }

    "worker" {
        Write-Log "Building worker container only"
        docker-compose --profile multi-user build pdf-compare-worker

        if ($Action -eq "up") {
            $workers = if ($Options) { $Options } else { "2" }
            Write-Log "Starting $workers worker container(s)..."
            docker-compose --profile multi-user up -d --scale pdf-compare-worker=$workers pdf-compare-worker
            Write-Info "Workers: $workers"
        }
    }

    "clean" {
        Write-Warn "Stopping all containers and removing volumes..."
        docker-compose --profile multi-user down -v
        docker-compose -f docker-compose-full.yml down -v
        docker-compose down -v
        Write-Log "Cleanup complete"
    }

    "test" {
        Write-Log "Running test build..."
        docker build --target test -t pdf-compare:test .
        docker run --rm pdf-compare:test pytest
    }

    "benchmark" {
        Write-Log "Running performance benchmark..."
        docker-compose up -d pdf-compare-standalone
        docker exec pdf-compare-standalone python -m pytest tests/test_performance.py -v
    }

    "logs" {
        $service = if ($Action) { $Action } else { "pdf-compare-standalone" }
        docker-compose logs -f $service
    }

    "shell" {
        $service = if ($Action) { $Action } else { "pdf-compare-standalone" }
        docker exec -it $service /bin/bash
    }

    default {
        Write-Host "PDF Compare - Build and Deployment Script (PowerShell)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\build.ps1 <mode> <action> [options]" -ForegroundColor White
        Write-Host ""
        Write-Host "=== CONTAINER MODES ===" -ForegroundColor Yellow
        Write-Host "Full Stack:" -ForegroundColor White
        Write-Host "  full           Full deployment (Postgres + Ollama + UI)"
        Write-Host "  standalone     Single container (UI + processing, no DB)"
        Write-Host ""
        Write-Host "Individual Services (can be managed independently):" -ForegroundColor White
        Write-Host "  postgres       PostgreSQL database only"
        Write-Host "  ollama         Ollama AI service only"
        Write-Host "  app            Application (UI + PDF Compare code) only"
        Write-Host "  infra          Both Postgres + Ollama together"
        Write-Host ""
        Write-Host "Multi-User:" -ForegroundColor White
        Write-Host "  multi-user     Separate UI and worker containers"
        Write-Host "  worker         Worker containers only"
        Write-Host ""
        Write-Host "Management:" -ForegroundColor White
        Write-Host "  clean          Stop and remove all containers"
        Write-Host "  test           Run test suite"
        Write-Host "  benchmark      Run performance tests"
        Write-Host "  logs           View container logs"
        Write-Host "  shell          Open shell in container"
        Write-Host ""
        Write-Host "=== ACTIONS ===" -ForegroundColor Yellow
        Write-Host "  build          Build images only (where applicable)"
        Write-Host "  up             Start containers (builds if needed)"
        Write-Host "  down           Stop containers (preserves data)"
        Write-Host "  restart        Restart containers"
        Write-Host ""
        Write-Host "=== EXAMPLES ===" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Initial Setup (First Time):" -ForegroundColor Cyan
        Write-Host "  .\build.ps1 full up                    # Start everything" -ForegroundColor White
        Write-Host ""
        Write-Host "Daily Development (Fast Updates):" -ForegroundColor Cyan
        Write-Host "  .\build.ps1 app build                  # Rebuild app code only (30s)" -ForegroundColor White
        Write-Host "  .\build.ps1 app up                     # Deploy new code" -ForegroundColor White
        Write-Host "  .\build.ps1 app restart                # Quick restart without rebuild" -ForegroundColor White
        Write-Host ""
        Write-Host "Independent Service Management:" -ForegroundColor Cyan
        Write-Host "  .\build.ps1 postgres up                # Start Postgres only" -ForegroundColor White
        Write-Host "  .\build.ps1 postgres down              # Stop Postgres (keeps data)" -ForegroundColor White
        Write-Host "  .\build.ps1 postgres restart           # Restart Postgres" -ForegroundColor White
        Write-Host ""
        Write-Host "  .\build.ps1 ollama up                  # Start Ollama + download models" -ForegroundColor White
        Write-Host "  .\build.ps1 ollama down                # Stop Ollama (keeps models)" -ForegroundColor White
        Write-Host "  .\build.ps1 ollama restart             # Restart Ollama" -ForegroundColor White
        Write-Host ""
        Write-Host "  .\build.ps1 infra up                   # Start both Postgres + Ollama" -ForegroundColor White
        Write-Host "  .\build.ps1 infra down                 # Stop both" -ForegroundColor White
        Write-Host ""
        Write-Host "Typical Workflow:" -ForegroundColor Cyan
        Write-Host "  # Start infrastructure once" -ForegroundColor Green
        Write-Host "  .\build.ps1 infra up" -ForegroundColor White
        Write-Host ""
        Write-Host "  # Work on your code, then rebuild only the app" -ForegroundColor Green
        Write-Host "  .\build.ps1 app build; .\build.ps1 app restart" -ForegroundColor White
        Write-Host ""
        Write-Host "  # Infrastructure keeps running - no downtime!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Other Commands:" -ForegroundColor Cyan
        Write-Host "  .\build.ps1 standalone up              # Start all-in-one container" -ForegroundColor White
        Write-Host "  .\build.ps1 multi-user up              # Start UI + workers" -ForegroundColor White
        Write-Host "  .\build.ps1 logs pdf-compare-ui        # View UI logs" -ForegroundColor White
        Write-Host "  .\build.ps1 shell pdf-compare-ui       # Open shell in UI container" -ForegroundColor White
        Write-Host "  .\build.ps1 clean                      # Clean up everything" -ForegroundColor White
        Write-Host ""
        Write-Host "Note: For bash/Git Bash users, use ./build.sh instead" -ForegroundColor Gray
        exit 0
    }
}

Write-Log "Done!"
