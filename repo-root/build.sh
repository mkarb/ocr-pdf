#!/bin/bash
# PDF Compare - Docker Build and Deployment Script

set -e

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_BLUE="\033[0;34m"

function log() {
    echo -e "${COLOR_GREEN}[PDF-COMPARE]${COLOR_RESET} $1"
}

function warn() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $1"
}

function info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

# Parse arguments
MODE=${1:-standalone}
ACTION=${2:-up}

case $MODE in
    standalone)
        log "Building standalone container (all-in-one)"
        docker-compose build pdf-compare-standalone

        if [ "$ACTION" == "up" ]; then
            log "Starting standalone container..."
            docker-compose up -d pdf-compare-standalone
            info "Access UI at: http://localhost:8501"
        fi
        ;;

    full)
        log "Building full deployment (Postgres + Ollama + UI)"
        docker-compose -f docker-compose-full.yml build

        if [ "$ACTION" == "up" ]; then
            log "Starting full deployment..."
            docker-compose -f docker-compose-full.yml up -d
            info "Access UI at: http://localhost:8501"
            info "Postgres: localhost:5432"
            info "Ollama: localhost:11434"
        fi
        ;;

    infra)
        log "Building infrastructure only (Postgres + Ollama)"
        info "Infrastructure containers use pre-built images, no build needed"

        if [ "$ACTION" == "up" ]; then
            log "Starting infrastructure services..."
            docker-compose -f docker-compose-full.yml up -d postgres ollama

            log "Waiting for services to be healthy..."
            docker-compose -f docker-compose-full.yml up -d ollama-init

            info "Infrastructure ready!"
            info "Postgres: localhost:5432"
            info "Ollama: localhost:11434"
        elif [ "$ACTION" == "down" ]; then
            log "Stopping infrastructure services..."
            docker-compose -f docker-compose-full.yml stop postgres ollama ollama-init
            info "Infrastructure stopped (data preserved in volumes)"
        fi
        ;;

    postgres)
        log "Managing PostgreSQL container only"
        info "PostgreSQL uses pre-built image: postgis/postgis:16-3.4"

        if [ "$ACTION" == "up" ]; then
            log "Starting PostgreSQL..."
            docker-compose -f docker-compose-full.yml up -d postgres
            log "Waiting for PostgreSQL to be healthy..."
            sleep 3
            docker-compose -f docker-compose-full.yml ps postgres
            info "Postgres: localhost:5432"
        elif [ "$ACTION" == "down" ]; then
            log "Stopping PostgreSQL..."
            docker-compose -f docker-compose-full.yml stop postgres
            info "PostgreSQL stopped (data preserved in postgres-data volume)"
        elif [ "$ACTION" == "restart" ]; then
            log "Restarting PostgreSQL..."
            docker-compose -f docker-compose-full.yml restart postgres
            info "PostgreSQL restarted"
        fi
        ;;

    ollama)
        log "Managing Ollama container only"
        info "Ollama uses pre-built image: ollama/ollama:latest"

        if [ "$ACTION" == "up" ]; then
            log "Starting Ollama..."
            docker-compose -f docker-compose-full.yml up -d ollama
            log "Waiting for Ollama to be healthy..."
            sleep 3

            log "Initializing models..."
            docker-compose -f docker-compose-full.yml up -d ollama-init
            info "Ollama: localhost:11434"
            info "Models: llama3.2, nomic-embed-text"
        elif [ "$ACTION" == "down" ]; then
            log "Stopping Ollama..."
            docker-compose -f docker-compose-full.yml stop ollama ollama-init
            info "Ollama stopped (models preserved in ollama-data volume)"
        elif [ "$ACTION" == "restart" ]; then
            log "Restarting Ollama..."
            docker-compose -f docker-compose-full.yml restart ollama
            info "Ollama restarted"
        fi
        ;;

    app)
        log "Building application container only (UI + PDF Compare)"

        if [ "$ACTION" == "build" ]; then
            log "Building application image..."
            docker-compose -f docker-compose-full.yml build pdf-compare-ui
            info "Application image built successfully"
        elif [ "$ACTION" == "up" ]; then
            log "Building and starting application container..."
            docker-compose -f docker-compose-full.yml build pdf-compare-ui
            docker-compose -f docker-compose-full.yml up -d pdf-compare-ui
            info "Access UI at: http://localhost:8501"
        elif [ "$ACTION" == "down" ]; then
            log "Stopping application container..."
            docker-compose -f docker-compose-full.yml stop pdf-compare-ui
            info "Application stopped"
        elif [ "$ACTION" == "restart" ]; then
            log "Restarting application container..."
            docker-compose -f docker-compose-full.yml restart pdf-compare-ui
            info "Application restarted"
        fi
        ;;

    multi-user)
        log "Building multi-user containers (UI + worker)"
        docker-compose --profile multi-user build

        if [ "$ACTION" == "up" ]; then
            log "Starting multi-user deployment..."
            docker-compose --profile multi-user up -d
            info "Access UI at: http://localhost:8501"
            info "Workers: $(docker-compose ps | grep worker | wc -l)"
        fi
        ;;

    worker)
        log "Building worker container only"
        docker-compose --profile multi-user build pdf-compare-worker

        if [ "$ACTION" == "up" ]; then
            WORKERS=${3:-2}
            log "Starting $WORKERS worker container(s)..."
            docker-compose --profile multi-user up -d --scale pdf-compare-worker=$WORKERS pdf-compare-worker
            info "Workers: $WORKERS"
        fi
        ;;

    clean)
        warn "Stopping all containers and removing volumes..."
        docker-compose --profile multi-user down -v
        docker-compose -f docker-compose-full.yml down -v
        docker-compose down -v
        log "Cleanup complete"
        ;;

    test)
        log "Running test build..."
        docker build --target test -t pdf-compare:test .
        docker run --rm pdf-compare:test pytest
        ;;

    benchmark)
        log "Running performance benchmark..."
        docker-compose up -d pdf-compare-standalone
        docker exec pdf-compare-standalone python -m pytest tests/test_performance.py -v
        ;;

    logs)
        SERVICE=${3:-pdf-compare-standalone}
        docker-compose logs -f $SERVICE
        ;;

    shell)
        SERVICE=${3:-pdf-compare-standalone}
        docker exec -it $SERVICE /bin/bash
        ;;

    *)
        echo "PDF Compare - Build and Deployment Script"
        echo ""
        echo "Usage: $0 <mode> <action> [options]"
        echo ""
        echo "=== CONTAINER MODES ==="
        echo "Full Stack:"
        echo "  full           Full deployment (Postgres + Ollama + UI)"
        echo "  standalone     Single container (UI + processing, no DB)"
        echo ""
        echo "Individual Services (can be managed independently):"
        echo "  postgres       PostgreSQL database only"
        echo "  ollama         Ollama AI service only"
        echo "  app            Application (UI + PDF Compare code) only"
        echo "  infra          Both Postgres + Ollama together"
        echo ""
        echo "Multi-User:"
        echo "  multi-user     Separate UI and worker containers"
        echo "  worker         Worker containers only"
        echo ""
        echo "Management:"
        echo "  clean          Stop and remove all containers"
        echo "  test           Run test suite"
        echo "  benchmark      Run performance tests"
        echo "  logs           View container logs"
        echo "  shell          Open shell in container"
        echo ""
        echo "=== ACTIONS ==="
        echo "  build          Build images only (where applicable)"
        echo "  up             Start containers (builds if needed)"
        echo "  down           Stop containers (preserves data)"
        echo "  restart        Restart containers"
        echo ""
        echo "=== EXAMPLES ==="
        echo ""
        echo "Initial Setup (First Time):"
        echo "  $0 full up                    # Start everything"
        echo ""
        echo "Daily Development (Fast Updates):"
        echo "  $0 app build                  # Rebuild app code only (30s)"
        echo "  $0 app up                     # Deploy new code"
        echo "  $0 app restart                # Quick restart without rebuild"
        echo ""
        echo "Independent Service Management:"
        echo "  $0 postgres up                # Start Postgres only"
        echo "  $0 postgres down              # Stop Postgres (keeps data)"
        echo "  $0 postgres restart           # Restart Postgres"
        echo ""
        echo "  $0 ollama up                  # Start Ollama + download models"
        echo "  $0 ollama down                # Stop Ollama (keeps models)"
        echo "  $0 ollama restart             # Restart Ollama"
        echo ""
        echo "  $0 infra up                   # Start both Postgres + Ollama"
        echo "  $0 infra down                 # Stop both"
        echo ""
        echo "Typical Workflow:"
        echo "  # Start infrastructure once"
        echo "  $0 infra up"
        echo ""
        echo "  # Work on your code, then rebuild only the app"
        echo "  $0 app build && $0 app restart"
        echo ""
        echo "  # Infrastructure keeps running - no downtime!"
        echo ""
        echo "Other Commands:"
        echo "  $0 standalone up              # Start all-in-one container"
        echo "  $0 multi-user up              # Start UI + workers"
        echo "  $0 logs pdf-compare-ui        # View UI logs"
        echo "  $0 shell pdf-compare-ui       # Open shell in UI container"
        echo "  $0 clean                      # Clean up everything"
        exit 1
        ;;
esac

log "Done!"