# Scaled Multi-User Deployment Guide

This guide explains how to deploy the PDF Compare application for multiple concurrent users with horizontal scaling, load balancing, and database replication.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         USERS (50+)                            │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
      ┌──────────────────┐
      │  Nginx (Port 80) │  ← Load Balancer with Sticky Sessions
      │  Load Balancer   │
      └────────┬─────────┘
               │
   ┌───────────┼───────────┐
   │           │           │
   ▼           ▼           ▼
┌─────┐    ┌─────┐    ┌─────┐
│UI #1│    │UI #2│    │UI #N│  ← Streamlit Instances (Scaled)
└──┬──┘    └──┬──┘    └──┬──┘
   │          │          │
   └──────────┼──────────┘
              │
              ▼
       ┌──────────────┐
       │  PgBouncer   │  ← Connection Pool Manager
       └──────┬───────┘
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
  ┌──────┐ ┌──────┐ ┌──────┐
  │ PG   │ │ PG   │ │ PG   │
  │Primary→│Replica││Replica│  ← PostgreSQL (Write + Reads)
  └──────┘ └───1──┘ └───2──┘
```

## Features

- **Horizontal Scaling**: Run multiple Streamlit instances behind a load balancer
- **Database Replication**: PostgreSQL primary + 2 read replicas for load distribution
- **Connection Pooling**: PgBouncer manages database connections efficiently
- **Sticky Sessions**: Users stay on the same Streamlit instance (maintains session state)
- **Session Isolation**: Proper separation between concurrent user sessions
- **Monitoring**: Prometheus + Grafana for observability
- **Auto-scaling**: Easy to add more instances with `docker-compose scale`

## Prerequisites

- Docker and Docker Compose
- 16+ GB RAM recommended
- 4+ CPU cores
- 50+ GB disk space (for models and data)

## Quick Start

### 1. Configure Environment

Copy the environment template:

```bash
cd repo-root
cp .env.scaled .env
```

Edit `.env` and set:
- Strong passwords for `POSTGRES_PASSWORD` and `POSTGRES_REPLICATION_PASSWORD`
- Adjust `CPU_LIMIT` based on your hardware
- Set `STREAMLIT_INSTANCES` (default: 3)

### 2. Start All Services

```bash
docker-compose -f docker-compose-scaled.yml up -d
```

This starts:
- Nginx load balancer (port 80)
- 3 Streamlit instances (auto-scaled)
- PostgreSQL primary + 2 replicas
- PgBouncer connection pooler
- Ollama AI service
- Redis cache
- Prometheus + Grafana (monitoring)

### 3. Wait for Initialization

Check startup progress:

```bash
docker-compose -f docker-compose-scaled.yml logs -f
```

Wait for:
- PostgreSQL replicas to sync (2-3 minutes)
- Ollama models to download (~5 minutes first time)
- All health checks to pass

### 4. Access the Application

- **Application**: http://localhost
- **Grafana Monitoring**: http://localhost:3000 (admin / your_password)
- **Prometheus**: http://localhost:9090

## Scaling Operations

### Scale Streamlit Instances

Add more UI instances on-the-fly:

```bash
# Scale to 5 instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=5

# Scale to 10 instances
docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=10
```

Nginx automatically load balances across all instances.

### Scale Read Replicas

To add a third replica, edit `docker-compose-scaled.yml`:

```yaml
postgres-replica-3:
  image: postgis/postgis:16-3.4
  # ... (copy from replica-2, change ports and volume)
```

Then add the replica URL to `.env`:

```
DATABASE_READ_URL_3=postgresql://pdfuser:password@postgres-replica-3:5432/pdfcompare
```

## Performance Tuning

### Connection Pooling

Edit PgBouncer settings in `docker-compose-scaled.yml`:

```yaml
pgbouncer:
  environment:
    MAX_CLIENT_CONN: 1000      # Max connections from app
    DEFAULT_POOL_SIZE: 25      # Connections to DB per pool
    RESERVE_POOL_SIZE: 10      # Emergency pool
```

### Resource Limits

Adjust per-container limits in `docker-compose-scaled.yml`:

```yaml
pdf-compare-ui:
  deploy:
    resources:
      limits:
        cpus: '8'       # CPU cores per instance
        memory: 4g      # RAM per instance
      reservations:
        cpus: '2'       # Minimum guaranteed
        memory: 1g
```

### PostgreSQL Performance

Increase shared memory for better performance:

```yaml
postgres-primary:
  shm_size: 512mb  # Increase from 256mb
  command: |
    postgres
    -c shared_buffers=512MB     # Increase from 256MB
    -c effective_cache_size=2GB # Increase from 1GB
```

## Monitoring

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login with credentials from `.env`
3. Import dashboards:
   - PostgreSQL monitoring
   - Nginx metrics
   - Container resource usage

### Health Checks

Check service health:

```bash
# All services
docker-compose -f docker-compose-scaled.yml ps

# Nginx status
curl http://localhost/health

# Database replication status
docker exec pdf-compare-postgres-primary psql -U pdfuser -d pdfcompare -c "SELECT * FROM pg_stat_replication;"

# PgBouncer stats
docker exec pdf-compare-pgbouncer psql -p 6432 -U pdfuser pgbouncer -c "SHOW POOLS;"
```

### Logs

View logs by service:

```bash
# All services
docker-compose -f docker-compose-scaled.yml logs -f

# Specific service
docker-compose -f docker-compose-scaled.yml logs -f pdf-compare-ui
docker-compose -f docker-compose-scaled.yml logs -f nginx
docker-compose -f docker-compose-scaled.yml logs -f postgres-primary
```

## Maintenance

### Backup Database

```bash
# Backup primary database
docker exec pdf-compare-postgres-primary pg_dump -U pdfuser pdfcompare > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20241014.sql | docker exec -i pdf-compare-postgres-primary psql -U pdfuser pdfcompare
```

### Update Application Code

Rebuild and restart app without downtime:

```bash
# Rebuild app image
docker-compose -f docker-compose-scaled.yml build pdf-compare-ui

# Rolling update (one instance at a time)
docker-compose -f docker-compose-scaled.yml up -d --no-deps --scale pdf-compare-ui=3 pdf-compare-ui
```

### Clean Up Old Sessions

Redis automatically handles session cleanup, but you can manually clear:

```bash
docker exec pdf-compare-redis redis-cli FLUSHDB
```

## Troubleshooting

### Replica Not Syncing

Check replication status:

```bash
docker exec pdf-compare-postgres-primary psql -U pdfuser -d pdfcompare -c "SELECT * FROM pg_stat_replication;"
```

If no replicas shown, restart replicas:

```bash
docker-compose -f docker-compose-scaled.yml restart postgres-replica-1 postgres-replica-2
```

### High Memory Usage

Check container memory:

```bash
docker stats
```

Reduce Streamlit instances or adjust memory limits.

### Slow Queries

Check PgBouncer pool stats:

```bash
docker exec pdf-compare-pgbouncer psql -p 6432 -U pdfuser pgbouncer -c "SHOW STATS;"
```

Increase `DEFAULT_POOL_SIZE` if connections are maxed out.

### Load Balancer Issues

Check Nginx logs:

```bash
docker-compose -f docker-compose-scaled.yml logs nginx
```

Test upstream health:

```bash
docker exec pdf-compare-nginx nginx -t  # Test config
```

## Capacity Planning

### Resource Requirements per User

Typical resource usage per concurrent user:

- **CPU**: 0.5-1 core during PDF processing
- **RAM**: 100-200 MB for UI session
- **Database connections**: 2-5 connections
- **Storage**: ~10 MB per uploaded PDF

### Recommended Configurations

**Small (10-20 users)**
- 2 Streamlit instances
- 1 Primary + 1 Replica
- 8 GB RAM total
- 4 CPU cores

**Medium (20-50 users)**
- 3-5 Streamlit instances
- 1 Primary + 2 Replicas
- 16 GB RAM total
- 8 CPU cores

**Large (50-100 users)**
- 5-10 Streamlit instances
- 1 Primary + 2-3 Replicas
- 32 GB RAM total
- 16 CPU cores

## Security Considerations

### Enable HTTPS

1. Get SSL certificates (Let's Encrypt recommended)
2. Update `nginx.conf` to enable HTTPS section
3. Mount certificates in `docker-compose-scaled.yml`:

```yaml
nginx:
  volumes:
    - ./ssl/cert.pem:/etc/nginx/ssl/cert.pem:ro
    - ./ssl/key.pem:/etc/nginx/ssl/key.pem:ro
```

### Database Security

- Change default passwords in `.env`
- Restrict database ports (remove from `docker-compose-scaled.yml` if not needed externally)
- Enable SSL for PostgreSQL connections

### Network Isolation

Consider using Docker networks with custom subnets for better isolation.

## Advanced Topics

### External PostgreSQL

To use an external PostgreSQL cluster instead of containerized:

1. Comment out PostgreSQL services in `docker-compose-scaled.yml`
2. Update `DATABASE_URL` in `.env` to point to external cluster
3. Ensure external DB has PostGIS extension

### Redis Session Storage

For true session persistence across restarts, configure Redis:

```python
# In streamlit_app.py
from streamlit_session_manager import SessionManager, GlobalSessionStore

# Use Redis backend (requires redis-py)
import redis
redis_client = redis.Redis(host='redis', port=6379)
```

### Kubernetes Deployment

For Kubernetes, convert Docker Compose to K8s manifests:

```bash
kompose convert -f docker-compose-scaled.yml
```

Then customize the generated YAML files for your K8s cluster.

## Getting Help

- **Issues**: Report bugs at GitHub
- **Logs**: Always include relevant logs when reporting issues
- **Monitoring**: Check Grafana dashboards for performance insights

## Summary

You now have a production-ready, horizontally-scalable PDF comparison service that can handle 50+ concurrent users with:

✅ Load-balanced UI instances
✅ Database replication for read scaling
✅ Connection pooling for efficiency
✅ Session isolation for security
✅ Monitoring and observability
✅ Easy scaling operations

For questions or issues, check the logs and monitoring dashboards first!
