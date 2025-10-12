# Docker Deployment Guide

Complete guide for deploying PDF Compare in Docker containers with full AI/RAG features.

## Overview

Three deployment options:

1. **Full Stack** (Recommended): PostgreSQL + Ollama + PDF Compare UI
2. **With External Ollama**: Connect to Ollama running on host machine
3. **Self-Contained**: All-in-one container with Ollama embedded

---

## Option 1: Full Stack Deployment (Recommended)

Complete system with all components in containers.

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- **8 GB RAM minimum, 16 GB recommended**
- **20 GB free disk space** (includes models)

### Quick Start

```bash
# 1. Build and start all services
docker-compose -f docker-compose-full.yml up -d

# 2. Wait for models to download (first run only, ~5-10 minutes)
docker-compose -f docker-compose-full.yml logs -f ollama-init

# 3. Verify everything is running
docker-compose -f docker-compose-full.yml ps

# 4. Run integration tests
docker exec pdf-compare-ui bash /app/docker_test.sh

# 5. Access UI
# http://localhost:8501
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| pdf-compare-ui | 8501 | Streamlit web interface |
| postgres | 5432 | PostgreSQL + PostGIS database |
| ollama | 11434 | Ollama LLM service |
| ollama-init | - | One-time model download |

### First Run

The first time you start the stack, it will:

1. Pull Docker images (~2 GB)
2. Download Ollama models (~2.3 GB):
   - `llama3.2` (~2 GB)
   - `nomic-embed-text` (~274 MB)
3. Initialize PostgreSQL database
4. Start Streamlit UI

**Total time: 10-15 minutes on first run**

### Verify Deployment

```bash
# Check all containers are running
docker-compose -f docker-compose-full.yml ps

# Should show:
# NAME                      STATUS
# pdf-compare-ui            Up (healthy)
# pdf-compare-postgres      Up (healthy)
# pdf-compare-ollama        Up (healthy)
# pdf-compare-ollama-init   Exited (0)

# Check Ollama models
docker exec pdf-compare-ollama ollama list

# Should show:
# NAME                    ID              SIZE
# llama3.2:latest         a80c4f17acd5    2.0 GB
# nomic-embed-text:latest 0a109f422b47    274 MB

# Run integration tests
docker exec pdf-compare-ui bash /app/docker_test.sh
```

### Test RAG Features

```bash
# Copy a test PDF into the container
docker cp your_diagram.pdf pdf-compare-ui:/tmp/test.pdf

# Run RAG tests
docker exec pdf-compare-ui python test_rag.py /tmp/test.pdf

# Interactive chat
docker exec -it pdf-compare-ui python pdf_compare/rag_simple.py /tmp/test.pdf
```

### Logs

```bash
# View all logs
docker-compose -f docker-compose-full.yml logs -f

# View specific service
docker-compose logs -f pdf-compare-ui
docker-compose logs -f ollama
docker-compose logs -f postgres
```

### Stop & Restart

```bash
# Stop all services
docker-compose -f docker-compose-full.yml down

# Stop and remove data (WARNING: deletes database and models)
docker-compose -f docker-compose-full.yml down -v

# Restart services
docker-compose -f docker-compose-full.yml restart

# Restart specific service
docker-compose restart pdf-compare-ui
```

---

## Option 2: External Ollama

Connect to Ollama running on your host machine.

### Prerequisites

- Ollama installed on host
- Models already pulled: `ollama pull llama3.2 && ollama pull nomic-embed-text`

### Setup

1. **Start Ollama on host:**

```bash
# Windows
ollama serve

# Linux/Mac (if not running as service)
ollama serve
```

2. **Update docker-compose-postgres.yml:**

```yaml
services:
  pdf-compare-ui:
    environment:
      # Windows/Mac: use host.docker.internal
      OLLAMA_HOST: http://host.docker.internal:11434

      # Linux: use docker0 bridge IP
      # OLLAMA_HOST: http://172.17.0.1:11434
```

3. **Start services:**

```bash
docker-compose -f docker-compose-postgres.yml up -d

# Test Ollama connection
docker exec pdf-compare-ui curl http://host.docker.internal:11434/api/version
```

### Troubleshooting External Ollama

**Connection refused:**

```bash
# Verify Ollama is listening on all interfaces
# Edit ~/.ollama/config.json (Linux/Mac) or registry (Windows)
# Set: "host": "0.0.0.0"

# Restart Ollama service
```

**Linux specific:**

```bash
# Find Docker bridge IP
ip addr show docker0

# Use that IP in OLLAMA_HOST
export OLLAMA_HOST=http://172.17.0.1:11434
```

---

## Option 3: Self-Contained Container

All components in a single container.

### Build & Run

```bash
# Build image with Ollama
docker build -f Dockerfile.with-ollama -t pdf-compare-full .

# Run container
docker run -d \
  --name pdf-compare-full \
  -p 8501:8501 \
  -p 11434:11434 \
  -v $(pwd)/data:/app/data \
  pdf-compare-full

# Wait for startup (pulls models on first run)
docker logs -f pdf-compare-full

# Run tests
docker exec pdf-compare-full bash /app/docker_test.sh
```

### Pros & Cons

**Pros:**
- Simple single-container deployment
- No external dependencies
- Portable

**Cons:**
- Larger image size (~5 GB)
- Slower startup (must initialize Ollama each time)
- Less efficient resource usage

---

## GPU Support (Optional)

Enable GPU acceleration for faster LLM inference.

### Prerequisites

- NVIDIA GPU with CUDA support
- nvidia-docker2 installed

### Linux Setup

```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Enable in docker-compose-full.yml

Uncomment GPU section in ollama service:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Restart:

```bash
docker-compose -f docker-compose-full.yml up -d ollama
```

Verify:

```bash
docker exec pdf-compare-ollama nvidia-smi
```

---

## Configuration

### Environment Variables

Edit `docker-compose-full.yml` to customize:

```yaml
services:
  pdf-compare-ui:
    environment:
      # Performance
      CPU_LIMIT: 15                    # Worker processes (max 15)
      PDF_MIN_SEGMENT_LEN: 0.50       # Min line segment length
      PDF_BEZIER_SAMPLES: 24          # Bezier curve sampling

      # Database
      DATABASE_URL: postgresql://user:pass@postgres:5432/pdfcompare

      # Ollama
      OLLAMA_HOST: http://ollama:11434

    # Resource limits
    cpus: 16        # Max CPU cores
    mem_limit: 8g   # Max RAM
```

### Ollama Model Selection

Use different models by changing `ollama-init` command:

```yaml
ollama-init:
  command:
    - |
      # Use smaller, faster model
      ollama pull llama3.2:1b

      # OR use larger, more accurate model
      # ollama pull llama3.2:8b

      ollama pull nomic-embed-text
```

### Persistent Data

Data is stored in Docker volumes:

```bash
# List volumes
docker volume ls | grep pdf-compare

# Inspect volume
docker volume inspect pdf-compare_postgres-data
docker volume inspect pdf-compare_ollama-data

# Backup volumes
docker run --rm -v pdf-compare_postgres-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz /data

# Restore volumes
docker run --rm -v pdf-compare_postgres-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs pdf-compare-ui

# Common issues:
# 1. Port already in use
docker ps | grep 8501  # Check if port is occupied

# 2. Out of memory
docker stats  # Check resource usage

# 3. Build failed
docker-compose build --no-cache pdf-compare-ui
```

### Ollama Issues

```bash
# Check Ollama is running
docker exec pdf-compare-ollama ollama list

# Re-pull models
docker exec pdf-compare-ollama ollama pull llama3.2
docker exec pdf-compare-ollama ollama pull nomic-embed-text

# Check Ollama logs
docker logs pdf-compare-ollama

# Test Ollama directly
docker exec pdf-compare-ollama ollama run llama3.2 "Hello"
```

### Database Issues

```bash
# Check PostgreSQL is running
docker exec pdf-compare-postgres pg_isready

# Connect to database
docker exec -it pdf-compare-postgres psql -U pdfuser -d pdfcompare

# Reset database
docker-compose down -v  # WARNING: Deletes all data
docker-compose up -d
```

### Performance Issues

**Slow LLM responses:**

```bash
# Use smaller model
docker exec pdf-compare-ollama ollama pull llama3.2:1b

# Check CPU/GPU usage
docker stats
```

**Out of memory:**

```bash
# Reduce worker count
# Edit docker-compose-full.yml:
environment:
  CPU_LIMIT: 4  # Reduce from 15

# Increase container memory
mem_limit: 16g  # Increase from 8g
```

**Slow extraction:**

```bash
# Increase CPU cores
cpus: 32  # If host has more cores available
```

### Network Issues

**Cannot reach services:**

```bash
# Check network
docker network inspect pdf-compare_pdf-network

# Verify services are on same network
docker inspect pdf-compare-ui | grep NetworkMode
docker inspect pdf-compare-ollama | grep NetworkMode

# Test connectivity between containers
docker exec pdf-compare-ui ping ollama
docker exec pdf-compare-ui curl http://ollama:11434/api/version
```

---

## Production Deployment

### Security

1. **Change default passwords:**

```yaml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: CHANGE_THIS_PASSWORD
  pdf-compare-ui:
    environment:
      DATABASE_URL: postgresql://pdfuser:NEW_PASSWORD@postgres:5432/pdfcompare
```

2. **Use Docker secrets:**

```bash
# Create secrets
echo "my_secure_password" | docker secret create db_password -

# Reference in docker-compose
secrets:
  db_password:
    external: true

services:
  postgres:
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
```

3. **Enable TLS:**

Configure nginx or traefik as reverse proxy with SSL certificates.

### Monitoring

```bash
# Resource usage
docker stats

# Health checks
docker inspect pdf-compare-ui | grep -A 10 Health

# Custom monitoring with Prometheus
# Add prometheus/grafana containers to stack
```

### Backups

```bash
# Automated backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker exec pdf-compare-postgres pg_dump -U pdfuser pdfcompare > \
  $BACKUP_DIR/db_$DATE.sql

# Backup volumes
docker run --rm -v pdf-compare_ollama-data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/ollama_$DATE.tar.gz /data

echo "Backup complete: $DATE"
EOF

chmod +x backup.sh

# Run daily via cron
crontab -e
# Add: 0 2 * * * /path/to/backup.sh
```

### Scaling

For high-load scenarios:

```yaml
services:
  # Add worker pool for parallel processing
  pdf-worker:
    build: .
    deploy:
      replicas: 4  # Scale horizontally
    command: python -m pdf_compare.worker  # Dedicated worker process
```

---

## Testing

### Integration Test Suite

```bash
# Run full test suite
docker exec pdf-compare-ui bash /app/docker_test.sh

# Test specific components
docker exec pdf-compare-ui python -c "import pdf_compare; print('OK')"
docker exec pdf-compare-ui python test_rag.py

# Test with actual PDF
docker cp test.pdf pdf-compare-ui:/tmp/
docker exec pdf-compare-ui python test_rag.py /tmp/test.pdf
```

### Performance Benchmarks

```bash
# Measure extraction speed
docker exec pdf-compare-ui python -c "
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
import time

start = time.time()
vm = pdf_to_vectormap_server('/tmp/test.pdf', workers=15)
elapsed = time.time() - start

print(f'Pages: {len(vm.pages)}')
print(f'Time: {elapsed:.2f}s')
print(f'Speed: {len(vm.pages)/elapsed:.2f} pages/sec')
"
```

---

## Updating

### Update Images

```bash
# Pull latest base images
docker-compose pull

# Rebuild with latest code
docker-compose build --no-cache

# Restart with new images
docker-compose down
docker-compose up -d
```

### Update Models

```bash
# Update to newer Ollama models
docker exec pdf-compare-ollama ollama pull llama3.2
docker exec pdf-compare-ollama ollama pull nomic-embed-text

# Remove old models
docker exec pdf-compare-ollama ollama rm old-model-name
```

---

## Support

- **Docker issues**: Check Docker Engine logs
- **Ollama issues**: https://github.com/ollama/ollama/issues
- **PostgreSQL issues**: https://www.postgresql.org/docs/
- **Project issues**: See project README

---

## Quick Reference

```bash
# Start
docker-compose -f docker-compose-full.yml up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f

# Test
docker exec pdf-compare-ui bash /app/docker_test.sh

# Shell access
docker exec -it pdf-compare-ui bash

# Resource usage
docker stats

# Cleanup
docker-compose down -v  # WARNING: Deletes data
docker system prune -a  # Free disk space
```
