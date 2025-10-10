# PDF Compare - Deployment Guide

This guide covers deploying PDF Compare in Docker containers for both single-user and multi-user scenarios.

## Architecture Overview

The system has two extraction modes:

1. **Standard Mode** (`pdf_extract.py`): Streamlit-compatible, serial processing when running in UI
2. **Server Mode** (`pdf_extract_server.py`): Docker-optimized, full multiprocessing, structured logging

## Deployment Options

### Option 1: Standalone Container (Single User)

Best for: Individual users, local deployments, testing

```bash
# Build and run
docker-compose up pdf-compare-standalone

# Access UI at http://localhost:8501
```

**Features:**
- All-in-one: UI + processing in single container
- 4 CPU cores, 4GB RAM (configurable)
- Persistent volumes for database and uploads
- Auto-restart on failure

**Configuration:**
```yaml
environment:
  - CPU_LIMIT=4                    # Max CPU cores to use
  - PDF_MIN_SEGMENT_LEN=0.50       # Min line segment length
  - PDF_MIN_FILL_AREA=0.50         # Min fill area
  - PDF_BEZIER_SAMPLES=24          # Bezier curve sampling density
```

---

### Option 2: Multi-User Deployment (Separate UI and Workers)

Best for: Team deployments, high-performance processing, network access

```bash
# Start UI + worker containers
docker-compose --profile multi-user up

# Access UI at http://localhost:8501
```

**Features:**
- Separate UI (1 core, 1GB) and worker (8 cores, 8GB) containers
- Shared data volume for database and files
- Horizontal scaling: add more workers as needed
- Resource isolation

**Scaling Workers:**
```bash
# Scale to 3 worker containers
docker-compose --profile multi-user up --scale pdf-compare-worker=3
```

---

### Option 3: Network Deployment (Multi-Machine)

Best for: Large teams, dedicated processing nodes

#### On Processing Node (Server):

```bash
# On beefy machine (high CPU/RAM)
docker run -d \
  --name pdf-worker \
  -v /path/to/shared/storage:/app/data \
  -e CPU_LIMIT=16 \
  -e PDF_SERVER_MODE=1 \
  --cpus=16 \
  --memory=16g \
  pdf-compare:latest \
  python -m pdf_compare.cli ingest --server
```

#### On User Machines:

```bash
# Connect to shared storage via NFS/SMB
docker run -d \
  --name pdf-ui \
  -p 8501:8501 \
  -v //server/shared:/app/data \
  -e CPU_LIMIT=1 \
  --cpus=1 \
  --memory=1g \
  pdf-compare:latest
```

---

## Environment Variables

### Processing Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CPU_LIMIT` | `os.cpu_count()` | Max CPU cores for worker processes |
| `PDF_SERVER_MODE` | `0` | Force server-optimized extraction (CLI) |
| `PDF_MIN_SEGMENT_LEN` | `0.50` | Drop line segments shorter than this |
| `PDF_MIN_FILL_AREA` | `0.50` | Drop fill areas smaller than this |
| `PDF_BEZIER_SAMPLES` | `24` | Max samples per Bezier curve (adaptive) |
| `PDF_SIMPLIFY_TOL` | `None` | Geometry simplification tolerance |

### Runtime Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/app/data/db/vectormap.sqlite` | SQLite database path |
| `UPLOADS_PATH` | `/app/data/uploads` | PDF upload directory |
| `OUTPUTS_PATH` | `/app/data/outputs` | Diff overlay output directory |
| `PYTHONUNBUFFERED` | `1` | Enable real-time logging |

---

## Building the Image

```bash
# Local build
docker build -t pdf-compare:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t pdf-compare:3.11 .

# Multi-architecture build (for ARM servers)
docker buildx build --platform linux/amd64,linux/arm64 -t pdf-compare:latest .
```

---

## Volume Management

### Data Persistence

```yaml
volumes:
  # Local directory mapping
  - ./data/db:/app/data/db           # SQLite database
  - ./data/uploads:/app/data/uploads # Uploaded PDFs
  - ./data/outputs:/app/data/outputs # Generated overlays
```

### Network Storage (NFS Example)

```yaml
volumes:
  - type: volume
    source: nfs-share
    target: /app/data
    volume:
      nocopy: true
      driver_opts:
        type: nfs
        o: addr=192.168.1.100,rw
        device: ":/mnt/pdf-compare-data"
```

---

## Resource Limits

### Recommended Specifications

| Deployment | CPU | RAM | Disk |
|------------|-----|-----|------|
| Standalone | 4 cores | 4GB | 10GB + storage |
| Multi-user UI | 1 core | 1GB | 5GB |
| Multi-user Worker | 8-16 cores | 8-16GB | 5GB |
| Network Node | 16+ cores | 16+ GB | 5GB |

### Setting Limits

```yaml
services:
  pdf-compare:
    cpus: 4                # CPU cores
    mem_limit: 4g          # RAM limit
    pids_limit: 1000       # Process limit
```

---

## Performance Tuning

### For Engineering Drawings (Typical)

```env
CPU_LIMIT=8
PDF_MIN_SEGMENT_LEN=0.50      # Default
PDF_MIN_FILL_AREA=0.50        # Default
PDF_BEZIER_SAMPLES=24         # Default adaptive max
```

### For Simple Diagrams (Faster)

```env
CPU_LIMIT=4
PDF_MIN_SEGMENT_LEN=1.0       # Larger = faster
PDF_MIN_FILL_AREA=1.0         # Larger = faster
PDF_BEZIER_SAMPLES=12         # Lower = faster
PDF_SIMPLIFY_TOL=0.1          # Enable simplification
```

### For High-Detail CAD (Quality)

```env
CPU_LIMIT=16
PDF_MIN_SEGMENT_LEN=0.1       # Capture tiny details
PDF_MIN_FILL_AREA=0.1         # Capture small fills
PDF_BEZIER_SAMPLES=48         # Higher sampling
```

---

## Monitoring

### Health Checks

```bash
# Check container health
docker ps
docker inspect pdf-compare-standalone | grep -A 10 Health

# View logs
docker logs -f pdf-compare-standalone
```

### Metrics in Logs

Server mode logs include performance metrics:

```
INFO: Extraction complete: doc_id=abc123, pages=50, geoms=12453, texts=3421, elapsed=8.23s, pages_per_sec=6.08
```

---

## CLI Usage in Container

### Server Mode (Optimized)

```bash
# Ingest with server-optimized extraction
docker exec pdf-compare-worker \
  python -m pdf_compare.cli ingest /app/data/uploads/drawing.pdf --server

# Standard mode (Streamlit-compatible)
docker exec pdf-compare-standalone \
  python -m pdf_compare.cli ingest /app/data/uploads/drawing.pdf
```

### Comparison

```bash
# Vector/text diff
docker exec pdf-compare-worker \
  python -m pdf_compare.cli compare old_doc_id new_doc_id \
  --out-overlay /app/data/outputs/diff.pdf \
  --base-pdf /app/data/uploads/new.pdf
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs pdf-compare-standalone

# Common issues:
# - Missing volumes: create directories first
# - Port conflict: change 8501:8501 to 8502:8501
# - Memory limit: increase mem_limit
```

### Slow Processing

```bash
# Check CPU allocation
docker stats pdf-compare-worker

# Increase workers
docker-compose --profile multi-user up --scale pdf-compare-worker=4

# Check logs for "Using N worker(s)" message
docker logs pdf-compare-worker | grep worker
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
mem_limit: 8g  # Was 4g

# Or reduce workers
CPU_LIMIT=4  # Was 8
```

---

## Security Considerations

### Production Deployment

1. **File Upload Limits**: Add nginx/traefik reverse proxy with file size limits
2. **Network Isolation**: Use Docker networks to isolate UI from workers
3. **Volume Permissions**: Set appropriate file permissions on mounted volumes
4. **Resource Limits**: Always set CPU/memory limits to prevent resource exhaustion

### Example with Traefik

```yaml
services:
  traefik:
    image: traefik:v2.10
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    labels:
      - "traefik.enable=true"

  pdf-compare-ui:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.pdf-compare.rule=Host(`pdf-compare.internal`)"
      - "traefik.http.services.pdf-compare.loadbalancer.server.port=8501"
      - "traefik.http.routers.pdf-compare.middlewares=limit-body"
      - "traefik.http.middlewares.limit-body.buffering.maxRequestBodyBytes=524288000"  # 500MB
```

---

## Backup and Maintenance

### Database Backup

```bash
# Backup SQLite database
docker exec pdf-compare-standalone \
  sqlite3 /app/data/db/vectormap.sqlite ".backup '/app/data/db/backup.sqlite'"

# Copy to host
docker cp pdf-compare-standalone:/app/data/db/backup.sqlite ./backup-$(date +%Y%m%d).sqlite
```

### Cleanup Old Data

```bash
# Remove processed PDFs older than 30 days
find ./data/uploads -type f -mtime +30 -delete

# Vacuum database (reclaim space)
docker exec pdf-compare-standalone \
  sqlite3 /app/data/db/vectormap.sqlite "VACUUM;"
```

---

## Next Steps

1. **API Server**: Add FastAPI endpoint for programmatic access
2. **Job Queue**: Add Celery/RQ for async processing
3. **Authentication**: Add OAuth/LDAP integration
4. **Kubernetes**: Deploy with Helm charts for enterprise scale

See `ROADMAP.md` for planned features.