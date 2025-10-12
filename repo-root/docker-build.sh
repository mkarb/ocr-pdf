#!/bin/bash
# Docker build script with optimization flags
# Usage: ./docker-build.sh

set -e

echo "Building PDF Compare Docker image..."
echo "This may take 10-15 minutes on first build due to heavy dependencies"
echo ""

# Build with increased timeout and better error handling
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --network=host \
    --build-arg BUILDKIT_STEP_LOG_MAX_SIZE=10000000 \
    -t pdf-compare:latest \
    -f Dockerfile \
    .

echo ""
echo "Build complete! Image: pdf-compare:latest"
echo ""
echo "To start the full stack:"
echo "  docker-compose -f docker-compose-full.yml up -d"
