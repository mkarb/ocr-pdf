# Multi-User Scaling - Quick Start

This guide helps you deploy a production-ready, multi-user version of PDF Compare that can handle 50+ concurrent users.

## What's New?

Your application now has:

✅ **Load Balancer (Nginx)** - Distributes traffic across multiple UI instances
✅ **Multiple Streamlit Instances** - Run 3-10+ instances in parallel
✅ **PostgreSQL Read Replicas** - 2 read replicas to distribute database load
✅ **PgBouncer Connection Pooling** - Efficient database connection management
✅ **Session Isolation** - Proper user session separation
✅ **Monitoring** - Prometheus + Grafana dashboards

## Quick Start (5 Minutes)

### 1. Setup Environment

```bash
cd repo-root

# Copy the scaled environment template
cp .env.scaled .env

# Edit passwords (IMPORTANT!)
nano .env  # or use your favorite editor
```

**Change these in `.env`:**
- `POSTGRES_PASSWORD=your_secure_password_here`
- `POSTGRES_REPLICATION_PASSWORD=your_replication_password`
- `GRAFANA_PASSWORD=your_grafana_password`

### 2. Start Scaled Deployment

```bash
# Start all services (3 Streamlit instances by default)
docker-compose -f docker-compose-scaled.yml up -d

# Watch the startup logs
docker-compose -f docker-compose-scaled.yml logs -f
```

**Wait for:**
- ✅ PostgreSQL primary and replicas to sync (~2 mins)
- ✅ Ollama models to download (~5 mins, first time only)
- ✅ All health checks to pass

### 3. Access Your Application

- **App**: http://localhost (load balanced!)
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)

### 4. Test Multi-User Support

Open multiple browser windows to http://localhost - each will be routed to different Streamlit instances with isolated sessions!

## Scaling Operations

### Add More UI Instances

```bash
# Scale to 5 instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=5

# Scale to 10 instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=10
```

### Check What's Running

```bash
# View all containers
docker-compose -f docker-compose-scaled.yml ps

# Check health
docker-compose -f docker-compose-scaled.yml ps | grep healthy
```

## Architecture Diagram

```
Users → Nginx (Load Balancer) → Streamlit Instances (3-10+)
                                      ↓
                                  PgBouncer (Connection Pool)
                                      ↓
                          ┌───────────┴───────────┐
                          ↓                       ↓
                    PostgreSQL Primary      PostgreSQL Replicas
                    (Write Operations)      (Read Operations)
```

## Key Files Created

| File | Purpose |
|------|---------|
| `docker-compose-scaled.yml` | Multi-container orchestration with scaling |
| `nginx.conf` | Load balancer with sticky sessions |
| `init-replication.sh` | PostgreSQL replication setup |
| `.env.scaled` | Environment configuration template |
| `pdf_compare/db_backend_scaled.py` | Database backend with read/write splitting |
| `ui/streamlit_session_manager.py` | Session isolation utilities |
| `prometheus.yml` | Monitoring configuration |
| `docs/deployment/SCALED_DEPLOYMENT.md` | Complete deployment guide |

## Performance Monitoring

### Check Database Replication

```bash
docker exec pdf-compare-postgres-primary psql -U pdfuser -d pdfcompare -c "SELECT * FROM pg_stat_replication;"
```

Should show 2 replicas streaming.

### Check Connection Pool

```bash
docker exec pdf-compare-pgbouncer psql -p 6432 -U pdfuser pgbouncer -c "SHOW POOLS;"
```

Shows active connections per database.

### Check Load Balancer

```bash
curl http://localhost/health
```

Should return "healthy".

## Capacity Guide

| Users | Streamlit Instances | RAM | CPU Cores |
|-------|---------------------|-----|-----------|
| 10-20 | 2-3 | 8 GB | 4 |
| 20-50 | 3-5 | 16 GB | 8 |
| 50-100 | 5-10 | 32 GB | 16 |

## Common Commands

```bash
# Start everything
docker-compose -f docker-compose-scaled.yml up -d

# Stop everything
docker-compose -f docker-compose-scaled.yml down

# View logs
docker-compose -f docker-compose-scaled.yml logs -f [service-name]

# Restart a service
docker-compose -f docker-compose-scaled.yml restart [service-name]

# Check resource usage
docker stats

# Scale UI instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=5
```

## Troubleshooting

### "Cannot connect to database"

Check PostgreSQL is healthy:
```bash
docker-compose -f docker-compose-scaled.yml ps postgres-primary
docker-compose -f docker-compose-scaled.yml logs postgres-primary
```

### "Replica not syncing"

Restart replicas:
```bash
docker-compose -f docker-compose-scaled.yml restart postgres-replica-1 postgres-replica-2
```

### "Out of memory"

Reduce Streamlit instances:
```bash
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=2
```

### "Slow performance"

1. Check connection pool stats (see above)
2. Increase `DEFAULT_POOL_SIZE` in `docker-compose-scaled.yml`
3. Add more read replicas

## Using the Scaled Database Backend

To use read/write splitting in your code:

```python
from pdf_compare.db_backend_scaled import create_scaled_backend

# Automatically uses DATABASE_URL and DATABASE_READ_URL_* from environment
backend = create_scaled_backend()

# Read operations automatically use replicas
docs = backend.list_documents()  # Uses read replica
results = backend.search_text("query")  # Uses read replica

# Write operations use primary
backend.upsert_vectormap(vectormap)  # Uses primary
```

## Next Steps

1. **Enable HTTPS** - Add SSL certificates for production
2. **Setup Backups** - Automate PostgreSQL backups
3. **Configure Alerts** - Setup Prometheus alerting rules
4. **Optimize** - Tune connection pools and resource limits
5. **Monitor** - Watch Grafana dashboards for performance

## Complete Documentation

For detailed information, see:
- [Full Deployment Guide](docs/deployment/SCALED_DEPLOYMENT.md)
- [Session Management](ui/streamlit_session_manager.py)
- [Database Scaling](pdf_compare/db_backend_scaled.py)

## Comparison: Single vs Scaled

| Feature | Original Setup | Scaled Setup |
|---------|----------------|--------------|
| Concurrent Users | 1-5 | 50-100+ |
| UI Instances | 1 | 3-10+ (scalable) |
| Database | Single PostgreSQL | Primary + 2 Replicas |
| Load Balancing | None | Nginx with sticky sessions |
| Connection Pooling | Basic | PgBouncer with 1000+ connections |
| Session Isolation | Basic | Enhanced with user separation |
| Monitoring | None | Prometheus + Grafana |
| High Availability | No | Yes (replica failover) |

## Support

Questions? Check:
1. Container logs: `docker-compose -f docker-compose-scaled.yml logs -f`
2. Grafana dashboards: http://localhost:3000
3. This documentation and the full deployment guide

**You're now ready to serve multiple users concurrently!** 🚀
