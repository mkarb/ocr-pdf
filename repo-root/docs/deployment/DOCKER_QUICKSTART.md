# PDF Compare - Docker Quick Start

Get up and running with Docker in 5 minutes.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- 4GB+ RAM available

## Quick Start

### 1. Build and Run (Standalone)

```bash
# Linux/Mac
./build.sh standalone up

# Windows
build.bat standalone up
```

Access UI at: **http://localhost:8501**

### 2. Upload and Compare PDFs

1. Open http://localhost:8501
2. Upload PDF revisions
3. Click "Ingest selected PDFs"
4. Search or compare documents

### 3. Stop

```bash
# Linux/Mac
docker-compose down

# Windows
docker-compose down
```

---

## Deployment Modes

### Standalone (Recommended for Single Users)

**Best for:** Individual use, testing, local deployments

```bash
./build.sh standalone up
```

- 1 container with UI + processing
- 4 CPU cores, 4GB RAM
- Data persists in `./data/`

---

### Multi-User (Recommended for Teams)

**Best for:** Shared servers, high-performance processing

```bash
./build.sh multi-user up
```

- UI container: 1 core, 1GB RAM
- Worker container: 8 cores, 8GB RAM
- Scale workers: `./build.sh worker up 4`

---

## Using the CLI

### Inside Container

```bash
# Enter container shell
./build.sh shell pdf-compare-standalone

# Ingest PDF with server-optimized extraction
python -m pdf_compare.cli ingest /app/data/uploads/drawing.pdf --server

# Compare documents
python -m pdf_compare.cli compare doc1 doc2 \
  --out-overlay /app/data/outputs/diff.pdf \
  --base-pdf /app/data/uploads/new.pdf

# Exit shell
exit
```

### From Host (via docker exec)

```bash
docker exec pdf-compare-standalone \
  python -m pdf_compare.cli ingest /app/data/uploads/drawing.pdf --server
```

---

## Configuration

### Environment Variables

Edit `docker-compose.yml`:

```yaml
environment:
  - CPU_LIMIT=8              # More cores = faster
  - PDF_BEZIER_SAMPLES=48    # Higher = more detail
  - PDF_MIN_SEGMENT_LEN=0.25 # Lower = more detail
```

### Resource Limits

```yaml
cpus: 8          # CPU cores
mem_limit: 8g    # RAM limit
```

Restart after changes:
```bash
docker-compose down && docker-compose up -d
```

---

## Data Persistence

All data is stored in `./data/`:

```
data/
├── db/
│   └── vectormap.sqlite    # Vector database
├── uploads/                # Uploaded PDFs
└── outputs/                # Generated diff overlays
```

**Backup:**
```bash
# Create backup
tar -czf backup-$(date +%Y%m%d).tar.gz ./data/

# Restore backup
tar -xzf backup-20251010.tar.gz
```

---

## Performance Tips

### For Large PDFs (100+ pages)

1. Increase workers:
   ```bash
   ./build.sh worker up 8
   ```

2. Increase CPU limit:
   ```yaml
   environment:
     - CPU_LIMIT=16
   cpus: 16
   ```

3. Use server mode CLI:
   ```bash
   docker exec pdf-compare-worker \
     python -m pdf_compare.cli ingest /app/drawing.pdf --server
   ```

### For Faster Processing (Lower Quality)

```yaml
environment:
  - PDF_MIN_SEGMENT_LEN=1.0      # Skip tiny details
  - PDF_BEZIER_SAMPLES=12        # Lower sampling
  - PDF_SIMPLIFY_TOL=0.15        # Simplify geometry
```

---

## Troubleshooting

### Port 8501 Already in Use

Change port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Changed from 8501:8501
```

Access at: http://localhost:8502

### Container Stops Immediately

Check logs:
```bash
./build.sh logs pdf-compare-standalone
```

Common issues:
- Missing data directories → Script creates them automatically
- Memory limit too low → Increase `mem_limit`
- Invalid configuration → Check YAML syntax

### Slow Performance

```bash
# Check resource usage
docker stats

# Increase limits
docker-compose down
# Edit docker-compose.yml (increase cpus/mem_limit)
docker-compose up -d
```

---

## Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
./build.sh logs pdf-compare-worker

# Last 100 lines
docker-compose logs --tail=100 pdf-compare-standalone
```

---

## Network Access (LAN)

### Allow LAN Access

1. Edit `docker-compose.yml`:
   ```yaml
   ports:
     - "0.0.0.0:8501:8501"  # Listen on all interfaces
   ```

2. Restart:
   ```bash
   docker-compose down && docker-compose up -d
   ```

3. Access from other machines:
   ```
   http://<your-ip>:8501
   ```

### Find Your IP

```bash
# Linux/Mac
hostname -I | awk '{print $1}'

# Windows
ipconfig | findstr IPv4
```

---

## Advanced: Network Deployment

### Server Node (Processing)

```bash
# On beefy server (192.168.1.100)
docker run -d \
  --name pdf-worker \
  -v /mnt/shared:/app/data \
  -e CPU_LIMIT=32 \
  --cpus=32 \
  --memory=32g \
  pdf-compare:latest \
  tail -f /dev/null  # Keep running

# Process PDFs
docker exec pdf-worker \
  python -m pdf_compare.cli ingest /app/data/uploads/*.pdf --server
```

### Client Nodes (UI)

```bash
# On user machines
docker run -d \
  --name pdf-ui \
  -p 8501:8501 \
  -v //192.168.1.100/shared:/app/data \
  -e CPU_LIMIT=1 \
  pdf-compare:latest
```

---

## Clean Up

```bash
# Stop and remove containers
./build.sh clean

# Remove images
docker rmi pdf-compare:latest

# Remove all data (WARNING: deletes database!)
rm -rf ./data/
```

---

## Next Steps

- Read [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
- Read [SERVER_MODE_COMPARISON.md](SERVER_MODE_COMPARISON.md) for performance tuning
- Set up automated backups
- Configure reverse proxy (nginx/traefik)
- Add authentication layer

---

## Getting Help

```bash
# Check container health
docker ps
docker inspect pdf-compare-standalone

# View resource usage
docker stats

# Access container shell
./build.sh shell

# Check Python version
docker exec pdf-compare-standalone python --version

# List installed packages
docker exec pdf-compare-standalone pip list
```

---

## Summary Commands

| Task | Command |
|------|---------|
| Start standalone | `./build.sh standalone up` |
| Start multi-user | `./build.sh multi-user up` |
| Scale workers | `./build.sh worker up 8` |
| View logs | `./build.sh logs` |
| Open shell | `./build.sh shell` |
| Stop all | `docker-compose down` |
| Clean up | `./build.sh clean` |

**Access UI:** http://localhost:8501