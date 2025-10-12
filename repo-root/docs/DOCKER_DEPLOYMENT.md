# Docker Deployment Guide

This guide explains how to deploy PDF Compare using Docker with the 3-container architecture.

## Architecture Overview

The deployment uses **3 separate containers**:

1. **pdf-compare-ui** - Streamlit UI + PDF processing workers
2. **ollama** - LLM inference service (llama3.2 + nomic-embed-text)
3. **postgres** - PostgreSQL database with PostGIS extension

Plus a temporary **ollama-init** container that pulls models on first startup.

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- At least 10GB free disk space
- 8GB+ RAM allocated to Docker
- Internet connection (for pulling images and models on first run)

### Check Prerequisites

```bash
# Check Docker
docker --version
docker-compose --version

# Check Docker is running
docker ps
```

## Quick Start

### Option 1: Using Build Script (Recommended)

**Windows:**
```powershell
# Build the image
.\docker-build.ps1

# Or build and test everything
.\docker-build-test.ps1 -DeploymentType full
```

**Linux/Mac:**
```bash
chmod +x docker-build.sh
./docker-build.sh
```

### Option 2: Manual Steps

1. **Create environment file:**
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

2. **Build and start services:**
```bash
docker-compose -f docker-compose-full.yml up -d --build
```

3. **Wait for services to be ready:**
```bash
# Watch logs
docker-compose -f docker-compose-full.yml logs -f

# Check status
docker-compose -f docker-compose-full.yml ps
```

4. **Access the application:**
- Streamlit UI: http://localhost:8501
- Ollama API: http://localhost:11434
- PostgreSQL: localhost:5432

## Build Process

The Docker build installs dependencies in stages to avoid timeouts:

1. **Stage 1**: Core dependencies (PyMuPDF, Shapely, Streamlit) - ~2 min
2. **Stage 2**: Image processing (OpenCV, Tesseract) - ~2 min
3. **Stage 3**: Database drivers (SQLAlchemy, psycopg2) - ~1 min
4. **Stage 4**: LangChain ecosystem - ~3 min
5. **Stage 5**: Vector stores (ChromaDB, sentence-transformers) - ~5-8 min
   - This stage installs PyTorch (~1GB) and is the slowest

**Total build time**: 10-15 minutes on first build (cached afterwards)

### Build Troubleshooting

If build fails with timeout or EOF errors:

1. **Increase Docker resources:**
   - Memory: 8GB minimum (Docker Desktop → Settings → Resources)
   - Disk space: 10GB free minimum

2. **Retry the build:**
   ```bash
   docker-compose -f docker-compose-full.yml build --no-cache
   ```

3. **Check Docker daemon logs:**
   - Windows: Docker Desktop → Troubleshoot → View logs
   - Linux: `sudo journalctl -u docker`

4. **Use the optimized build script** which sets timeout flags

## Container Details

### pdf-compare-ui

- **Purpose**: Runs Streamlit UI and handles all PDF processing
- **Resources**: 16 CPUs, 8GB RAM
- **Ports**: 8501
- **Volumes**:
  - `./data/uploads` - PDF uploads
  - `./data/outputs` - Generated diffs
- **Environment**:
  - `DATABASE_URL` - PostgreSQL connection
  - `OLLAMA_HOST` - Ollama API endpoint
  - `CPU_LIMIT` - Max parallel workers

### ollama

- **Purpose**: LLM inference for RAG features
- **Resources**: Unlimited (can use GPU if available)
- **Ports**: 11434
- **Volumes**: `ollama-data` - Model storage (~4-8GB)
- **Models pulled**: llama3.2, nomic-embed-text

**GPU Support (Optional):**
Uncomment lines 47-54 in docker-compose-full.yml and install nvidia-docker:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### postgres

- **Purpose**: Persistent data storage with spatial queries
- **Image**: postgis/postgis:16-3.4
- **Ports**: 5432
- **Volumes**: `postgres-data` - Database persistence
- **Extensions**: PostGIS for spatial indexing

## Configuration

Edit `.env` file or set environment variables:

```bash
# Database
POSTGRES_DB=pdfcompare
POSTGRES_USER=pdfuser
POSTGRES_PASSWORD=pdfpassword

# Ollama endpoint
OLLAMA_HOST=http://ollama:11434

# Performance tuning
CPU_LIMIT=15              # Max parallel PDF processing workers
PDF_MIN_SEGMENT_LEN=0.50  # Vector filtering threshold
PDF_MIN_FILL_AREA=0.50    # Shape filtering threshold
PDF_BEZIER_SAMPLES=24     # Curve resolution
```

## Common Operations

### View Logs
```bash
# All services
docker-compose -f docker-compose-full.yml logs -f

# Specific service
docker-compose -f docker-compose-full.yml logs -f pdf-compare-ui
docker-compose -f docker-compose-full.yml logs -f ollama
docker-compose -f docker-compose-full.yml logs -f postgres
```

### Restart Services
```bash
# All services
docker-compose -f docker-compose-full.yml restart

# Specific service
docker-compose -f docker-compose-full.yml restart pdf-compare-ui
```

### Stop Services
```bash
# Stop but keep data
docker-compose -f docker-compose-full.yml down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose -f docker-compose-full.yml down -v
```

### Shell Access
```bash
# Access UI container
docker exec -it pdf-compare-ui bash

# Access Ollama container
docker exec -it pdf-compare-ollama bash

# Access PostgreSQL
docker exec -it pdf-compare-postgres psql -U pdfuser -d pdfcompare
```

### Test RAG Features
```bash
# Run RAG test inside container
docker exec pdf-compare-ui python test_rag.py

# Check Ollama models
docker exec pdf-compare-ollama ollama list

# Test Ollama API
curl http://localhost:11434/api/version
```

### Database Operations
```bash
# Backup database
docker exec pdf-compare-postgres pg_dump -U pdfuser pdfcompare > backup.sql

# Restore database
cat backup.sql | docker exec -i pdf-compare-postgres psql -U pdfuser pdfcompare

# Connect to database
docker exec -it pdf-compare-postgres psql -U pdfuser -d pdfcompare
```

## Health Checks

All services have health checks:

```bash
# Check service health
docker-compose -f docker-compose-full.yml ps

# Manual health checks
curl http://localhost:8501/_stcore/health  # Streamlit
curl http://localhost:11434/api/version     # Ollama
docker exec pdf-compare-postgres pg_isready -U pdfuser
```

## Data Persistence

### Volumes

- **postgres-data**: PostgreSQL database (persists across restarts)
- **ollama-data**: Ollama models (~4-8GB, persists across restarts)
- **./data/uploads**: PDF uploads (mounted from host)
- **./data/outputs**: Generated diffs (mounted from host)

### Backup Strategy

```bash
# Backup volumes
docker run --rm -v pdf-compare_postgres-data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data
docker run --rm -v pdf-compare_ollama-data:/data -v $(pwd):/backup ubuntu tar czf /backup/ollama-backup.tar.gz /data

# Backup host-mounted data
tar czf data-backup.tar.gz ./data/
```

## Deployment Options

### Option 1: Full Stack (Recommended)
3 containers: UI + Ollama + PostgreSQL
```bash
docker-compose -f docker-compose-full.yml up -d
```

### Option 2: PostgreSQL Only
2 containers: UI + PostgreSQL (no RAG features)
```bash
docker-compose -f docker-compose-postgres.yml up -d
```

### Option 3: Standalone
1 container: UI only with SQLite (development only)
```bash
docker-compose -f docker-compose.yml up -d
```

## Troubleshooting

### Build Issues

**Problem**: EOF error or timeout during build
**Solution**:
- Increase Docker memory to 8GB+
- Use the build scripts which set proper timeouts
- Check internet connection

**Problem**: "Failed to remove contents in temporary directory"
**Solution**: This is a warning, not an error. The build should continue.

### Runtime Issues

**Problem**: Ollama models not found
**Solution**:
- Check ollama-init container completed: `docker-compose -f docker-compose-full.yml ps`
- Manually pull models: `docker exec pdf-compare-ollama ollama pull llama3.2`

**Problem**: PostgreSQL connection refused
**Solution**:
- Wait for health check: `docker-compose -f docker-compose-full.yml ps`
- Check logs: `docker-compose -f docker-compose-full.yml logs postgres`

**Problem**: Streamlit not accessible
**Solution**:
- Check container is running: `docker ps`
- Check health: `curl http://localhost:8501/_stcore/health`
- View logs: `docker-compose -f docker-compose-full.yml logs pdf-compare-ui`

### Performance Issues

**Problem**: Slow PDF processing
**Solution**:
- Increase `CPU_LIMIT` in .env
- Increase Docker CPU allocation
- Check container resources: `docker stats`

**Problem**: Out of memory errors
**Solution**:
- Increase Docker memory limit (8GB minimum)
- Reduce `CPU_LIMIT` to lower concurrent workers
- Process smaller PDFs

## Security Considerations

### Production Deployment

1. **Change default passwords:**
   ```bash
   POSTGRES_PASSWORD=<strong-password>
   ```

2. **Don't expose ports externally:**
   Remove port mappings or use firewall rules

3. **Use secrets management:**
   ```yaml
   secrets:
     postgres_password:
       file: ./secrets/postgres_password.txt
   ```

4. **Enable SSL/TLS:**
   Add reverse proxy (nginx, traefik) with SSL certificates

5. **Regular updates:**
   ```bash
   docker-compose -f docker-compose-full.yml pull
   docker-compose -f docker-compose-full.yml up -d
   ```

## Performance Tuning

### For Large PDFs (100+ pages)

```bash
CPU_LIMIT=24              # More parallel workers
PDF_BEZIER_SAMPLES=12     # Lower curve resolution
```

### For High Accuracy

```bash
PDF_BEZIER_SAMPLES=48     # Higher curve resolution
PDF_MIN_SEGMENT_LEN=0.25  # More sensitive vector detection
PDF_MIN_FILL_AREA=0.25    # Catch smaller shapes
```

### For Memory-Constrained Environments

```bash
CPU_LIMIT=4               # Fewer workers
```

## Monitoring

### Resource Usage
```bash
# Real-time stats
docker stats

# Container-specific
docker stats pdf-compare-ui pdf-compare-ollama pdf-compare-postgres
```

### Logs Collection
```bash
# Export logs
docker-compose -f docker-compose-full.yml logs > deployment.log

# Follow specific service
docker-compose -f docker-compose-full.yml logs -f --tail=100 pdf-compare-ui
```

## Next Steps

- Configure [environment variables](.env.example)
- Set up [backup strategy](#backup-strategy)
- Review [security considerations](#security-considerations)
- Explore [RAG features](../README.md#rag-features)

## Support

- GitHub Issues: https://github.com/yourusername/pdf-compare/issues
- Documentation: https://github.com/yourusername/pdf-compare/docs
