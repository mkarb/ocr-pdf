# Docker Setup Guide

## Two-Container Setup: UI + PostgreSQL

This setup runs the PDF comparison tool in a containerized environment with:
- **Container 1**: Streamlit UI (Frontend + Processing)
- **Container 2**: PostgreSQL Database (with PostGIS for spatial data)

### Quick Start

```bash
# Start both containers
docker-compose -f docker-compose-postgres.yml up -d

# View logs
docker-compose -f docker-compose-postgres.yml logs -f

# Stop containers
docker-compose -f docker-compose-postgres.yml down

# Stop and remove all data
docker-compose -f docker-compose-postgres.yml down -v
```

### Access

- **Streamlit UI**: http://localhost:8501
- **PostgreSQL**: localhost:5432
  - Database: `pdfcompare`
  - User: `pdfuser`
  - Password: `pdfpassword`

### Architecture

```
┌─────────────────────────────────────┐
│   Streamlit UI Container           │
│   - PDF Extraction (16 cores)      │
│   - UI Interface                    │
│   - Connects to PostgreSQL          │
│   Port: 8501                        │
└──────────────┬──────────────────────┘
               │ DATABASE_URL
               ↓
┌─────────────────────────────────────┐
│   PostgreSQL Container              │
│   - PostGIS Extension               │
│   - Persistent Storage              │
│   - Full-Text Search                │
│   Port: 5432                        │
└─────────────────────────────────────┘
```

### Configuration

#### Environment Variables (UI Container)

```yaml
DATABASE_URL: postgresql://pdfuser:pdfpassword@postgres:5432/pdfcompare
CPU_LIMIT: 15                # Worker processes for PDF extraction
PDF_MIN_SEGMENT_LEN: 0.50    # Minimum segment length
PDF_MIN_FILL_AREA: 0.50      # Minimum fill area
PDF_BEZIER_SAMPLES: 24       # Bezier curve samples
```

#### PostgreSQL Configuration

Default credentials (change for production):
```
POSTGRES_DB: pdfcompare
POSTGRES_USER: pdfuser
POSTGRES_PASSWORD: pdfpassword
```

### Data Persistence

- **PostgreSQL Data**: Stored in Docker volume `postgres-data`
- **Uploaded Files**: Mounted at `./data/uploads`
- **Output Files**: Mounted at `./data/outputs`

### Production Deployment

1. **Change Default Credentials**:
```yaml
environment:
  POSTGRES_DB: your_db_name
  POSTGRES_USER: your_username
  POSTGRES_PASSWORD: strong_password_here
```

2. **Update DATABASE_URL** in UI container to match

3. **Use Secrets** (Docker Swarm/Kubernetes):
```yaml
secrets:
  - postgres_password
environment:
  POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

4. **Enable SSL** for PostgreSQL connections

5. **Configure Backups**:
```bash
# Backup database
docker exec pdf-compare-postgres pg_dump -U pdfuser pdfcompare > backup.sql

# Restore database
docker exec -i pdf-compare-postgres psql -U pdfuser pdfcompare < backup.sql
```

### Scaling

#### Increase UI Resources

```yaml
pdf-compare-ui:
  cpus: 32           # Use all available cores
  mem_limit: 16g     # Increase memory
  environment:
    CPU_LIMIT: 31    # Workers = cores - 1
```

#### Multiple UI Instances

```bash
# Scale UI to 3 instances (load balancer required)
docker-compose -f docker-compose-postgres.yml up -d --scale pdf-compare-ui=3
```

### Troubleshooting

#### Check Database Connection

```bash
# From UI container
docker exec pdf-compare-ui python -c "
from pdf_compare.db_backend import create_backend
backend = create_backend('postgresql://pdfuser:pdfpassword@postgres:5432/pdfcompare')
print('Connection successful!')
"
```

#### View PostgreSQL Logs

```bash
docker logs pdf-compare-postgres
```

#### Check Database Size

```bash
docker exec pdf-compare-postgres psql -U pdfuser -d pdfcompare -c "
SELECT pg_size_pretty(pg_database_size('pdfcompare'));
"
```

#### Monitor Performance

```bash
# PostgreSQL stats
docker exec pdf-compare-postgres psql -U pdfuser -d pdfcompare -c "
SELECT * FROM pg_stat_database WHERE datname='pdfcompare';
"
```

### Development vs Production

#### Development (SQLite)
```bash
# Use local SQLite
streamlit run ui/streamlit_app.py
```

#### Production (PostgreSQL)
```bash
# Use Docker Compose
docker-compose -f docker-compose-postgres.yml up -d
```

### Migration from SQLite to PostgreSQL

```python
# Migration script
from pdf_compare.store import open_db as open_sqlite
from pdf_compare.db_backend import create_backend
from pdf_compare.models import VectorMap

# Open SQLite
sqlite_conn = open_sqlite("vectormap.sqlite")

# Open PostgreSQL
pg_backend = create_backend("postgresql://pdfuser:pdfpassword@localhost:5432/pdfcompare")

# Migrate documents
# (Implementation depends on your specific needs)
```

### Network Configuration

Containers communicate via `pdf-network` bridge network.

External access:
- UI: `localhost:8501`
- PostgreSQL: `localhost:5432` (if you need direct access)

To restrict PostgreSQL access (recommended for production):
```yaml
postgres:
  ports: []  # Remove port mapping
  # UI can still access via internal network
```
