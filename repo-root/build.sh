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
        echo "Modes:"
        echo "  standalone     Single container (UI + processing)"
        echo "  multi-user     Separate UI and worker containers"
        echo "  worker         Worker containers only"
        echo "  clean          Stop and remove all containers"
        echo "  test           Run test suite"
        echo "  benchmark      Run performance tests"
        echo "  logs           View container logs"
        echo "  shell          Open shell in container"
        echo ""
        echo "Actions:"
        echo "  build          Build images only"
        echo "  up             Build and start containers"
        echo "  down           Stop containers"
        echo ""
        echo "Examples:"
        echo "  $0 standalone up              # Start all-in-one container"
        echo "  $0 multi-user up              # Start UI + workers"
        echo "  $0 worker up 4                # Start 4 worker containers"
        echo "  $0 logs pdf-compare-worker    # View worker logs"
        echo "  $0 shell pdf-compare-ui       # Open shell in UI container"
        echo "  $0 clean                      # Clean up everything"
        exit 1
        ;;
esac

log "Done!"