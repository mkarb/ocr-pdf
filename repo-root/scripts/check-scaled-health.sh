#!/bin/bash
# ============================================================================
# Health Check Script for Scaled PDF Compare Deployment
# ============================================================================
# Checks the health of all services in the scaled deployment
# Usage: ./scripts/check-scaled-health.sh
# ============================================================================

set -e

echo "=================================================="
echo "PDF Compare - Scaled Deployment Health Check"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is running
if ! docker-compose -f docker-compose-scaled.yml ps > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose not running${NC}"
    exit 1
fi

echo "1. Container Status"
echo "-------------------"
containers=$(docker-compose -f docker-compose-scaled.yml ps --format json 2>/dev/null | jq -r '.Name' 2>/dev/null || docker-compose -f docker-compose-scaled.yml ps --services)

for container in $containers; do
    if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container" 2>/dev/null; then
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "running")
        if [ "$status" = "healthy" ] || [ "$status" = "running" ]; then
            echo -e "${GREEN}✓${NC} $container - $status"
        else
            echo -e "${YELLOW}⚠${NC} $container - $status"
        fi
    else
        echo -e "${RED}✗${NC} $container - not running"
    fi
done

echo ""
echo "2. Nginx Load Balancer"
echo "----------------------"
nginx_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/health 2>/dev/null || echo "000")
if [ "$nginx_health" = "200" ]; then
    echo -e "${GREEN}✓${NC} Nginx responding (HTTP 200)"
else
    echo -e "${RED}✗${NC} Nginx not responding (HTTP $nginx_health)"
fi

echo ""
echo "3. PostgreSQL Replication"
echo "-------------------------"
replication_count=$(docker exec pdf-compare-postgres-primary psql -U pdfuser -d pdfcompare -t -c "SELECT COUNT(*) FROM pg_stat_replication;" 2>/dev/null | tr -d ' ')
if [ "$replication_count" -ge "2" ]; then
    echo -e "${GREEN}✓${NC} $replication_count replica(s) connected"
    docker exec pdf-compare-postgres-primary psql -U pdfuser -d pdfcompare -c "SELECT client_addr, state, sync_state FROM pg_stat_replication;" 2>/dev/null | tail -n +3
else
    echo -e "${YELLOW}⚠${NC} Only $replication_count replica(s) connected (expected 2)"
fi

echo ""
echo "4. PgBouncer Connection Pool"
echo "-----------------------------"
if docker exec pdf-compare-pgbouncer psql -p 6432 -U pdfuser pgbouncer -c "SHOW POOLS;" 2>/dev/null | grep -q "pdfcompare"; then
    echo -e "${GREEN}✓${NC} PgBouncer running"
    docker exec pdf-compare-pgbouncer psql -p 6432 -U pdfuser pgbouncer -t -c "SHOW POOLS;" 2>/dev/null | grep pdfcompare
else
    echo -e "${RED}✗${NC} PgBouncer not responding"
fi

echo ""
echo "5. Streamlit Instances"
echo "----------------------"
streamlit_count=$(docker ps --filter "name=pdf-compare-ui" --format "{{.Names}}" | wc -l)
echo "Running instances: $streamlit_count"

for instance in $(docker ps --filter "name=pdf-compare-ui" --format "{{.Names}}"); do
    health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/_stcore/health 2>/dev/null || echo "000")
    if [ "$health" = "200" ]; then
        echo -e "${GREEN}✓${NC} $instance - healthy"
    else
        echo -e "${YELLOW}⚠${NC} $instance - unhealthy (HTTP $health)"
    fi
done

echo ""
echo "6. Ollama AI Service"
echo "--------------------"
if docker exec pdf-compare-ollama ollama list 2>/dev/null | grep -q "llama3.2"; then
    echo -e "${GREEN}✓${NC} Ollama running with models"
    docker exec pdf-compare-ollama ollama list 2>/dev/null | tail -n +2
else
    echo -e "${YELLOW}⚠${NC} Ollama models not loaded"
fi

echo ""
echo "7. Redis Cache"
echo "--------------"
if docker exec pdf-compare-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo -e "${GREEN}✓${NC} Redis responding"
    keys=$(docker exec pdf-compare-redis redis-cli DBSIZE 2>/dev/null | cut -d: -f2)
    echo "   Keys in cache: $keys"
else
    echo -e "${YELLOW}⚠${NC} Redis not responding"
fi

echo ""
echo "8. Monitoring Services"
echo "----------------------"
prometheus_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/-/healthy 2>/dev/null || echo "000")
if [ "$prometheus_health" = "200" ]; then
    echo -e "${GREEN}✓${NC} Prometheus healthy (http://localhost:9090)"
else
    echo -e "${YELLOW}⚠${NC} Prometheus not responding"
fi

grafana_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null || echo "000")
if [ "$grafana_health" = "200" ]; then
    echo -e "${GREEN}✓${NC} Grafana healthy (http://localhost:3000)"
else
    echo -e "${YELLOW}⚠${NC} Grafana not responding"
fi

echo ""
echo "9. Resource Usage"
echo "-----------------"
echo "Top 5 containers by CPU:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6

echo ""
echo "10. Disk Usage"
echo "--------------"
echo "Docker volumes:"
docker volume ls --filter "name=ocr-pdf" --format "{{.Name}}" | while read volume; do
    size=$(docker system df -v 2>/dev/null | grep "$volume" | awk '{print $3}' || echo "Unknown")
    echo "  $volume: $size"
done

echo ""
echo "=================================================="
echo "Health Check Complete"
echo "=================================================="
echo ""
echo "Quick Actions:"
echo "  View logs:     docker-compose -f docker-compose-scaled.yml logs -f"
echo "  Scale UI:      docker-compose -f docker-compose-scaled.yml up -d --scale pdf-compare-ui=5"
echo "  Restart:       docker-compose -f docker-compose-scaled.yml restart [service]"
echo "  Stop all:      docker-compose -f docker-compose-scaled.yml down"
echo ""
