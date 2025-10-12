# Docker Build Verification Guide

Complete testing checklist for verifying Docker deployments work correctly.

## Quick Start

### Windows

```powershell
# Automated build and test
.\docker-build-test.ps1 -DeploymentType full

# Or manually
docker-compose -f docker-compose-full.yml up -d
docker exec pdf-compare-ui bash /app/docker_test.sh
```

### Linux/Mac

```bash
# Build and test
docker-compose -f docker-compose-full.yml up -d
docker exec pdf-compare-ui bash /app/docker_test.sh
```

---

## Pre-Build Checklist

Before building, verify you have:

- [ ] Docker Engine 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] **8 GB RAM minimum** (16 GB recommended)
- [ ] **20 GB free disk space** (includes models)
- [ ] Internet connection (for downloading models)

Check versions:

```bash
docker --version
docker-compose --version
docker info | grep -i memory
df -h
```

---

## Build Verification

### Step 1: Build the Image

```bash
# Build without cache for clean test
docker-compose -f docker-compose-full.yml build --no-cache

# Check image was created
docker images | grep pdf-compare
```

**Expected output:**

```
REPOSITORY      TAG       IMAGE ID       CREATED         SIZE
pdf-compare     latest    abc123def456   2 minutes ago   2.5GB
```

**Common build errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `COPY failed` | Missing files | Verify all files exist: `requirements.txt`, `pyproject.toml`, `README.md` |
| `pip install failed` | Missing system deps | Check Dockerfile has build-essential, libssl-dev |
| `No space left on device` | Disk full | Free up space: `docker system prune -a` |

### Step 2: Start Services

```bash
docker-compose -f docker-compose-full.yml up -d
```

Wait for services to be healthy:

```bash
watch -n 2 'docker-compose -f docker-compose-full.yml ps'
```

**Expected status after ~2-3 minutes:**

```
NAME                      STATUS
pdf-compare-ui            Up (healthy)
pdf-compare-postgres      Up (healthy)
pdf-compare-ollama        Up (healthy)
pdf-compare-ollama-init   Exited (0)
```

**If services are unhealthy:**

```bash
# Check logs
docker-compose logs pdf-compare-ui
docker-compose logs ollama

# Common issues:
# - Port already in use: Change port in docker-compose.yml
# - Out of memory: Reduce workers or increase Docker memory limit
# - Database connection: Wait longer, PostgreSQL may still be initializing
```

### Step 3: Verify Models Downloaded

**Check models are present:**

```bash
docker exec pdf-compare-ollama ollama list
```

**Expected output:**

```
NAME                    ID              SIZE
llama3.2:latest         a80c4f17acd5    2.0 GB
nomic-embed-text:latest 0a109f422b47    274 MB
```

**If models are missing:**

```bash
# Check ollama-init logs
docker logs pdf-compare-ollama-init

# Manually pull models
docker exec pdf-compare-ollama ollama pull llama3.2
docker exec pdf-compare-ollama ollama pull nomic-embed-text
```

---

## Functional Testing

### Test 1: Integration Test Suite

```bash
docker exec pdf-compare-ui bash /app/docker_test.sh
```

**Expected output:**

```
==================================
  PDF Compare - Container Test
==================================

1. Python Environment
-----------------------------------
[OK]    Python version (3.11+)      ✓ PASS
[OK]    pip installed                ✓ PASS

2. Core Python Packages
-----------------------------------
[OK]    pymupdf                      ✓ PASS
[OK]    shapely                      ✓ PASS
[OK]    numpy                        ✓ PASS
[OK]    streamlit                    ✓ PASS
[OK]    opencv                       ✓ PASS
[OK]    pytesseract                  ✓ PASS
[OK]    sqlalchemy                   ✓ PASS

3. RAG/AI Packages
-----------------------------------
[OK]    langchain                    ✓ PASS
[OK]    langchain_community          ✓ PASS
[OK]    chromadb                     ✓ PASS
[OK]    pypdf                        ✓ PASS

4. Project Modules
-----------------------------------
[OK]    pdf_compare module           ✓ PASS
[OK]    pdf_extract_server           ✓ PASS
[OK]    rag_simple                   ✓ PASS
[OK]    db_backend                   ✓ PASS

5. System Dependencies
-----------------------------------
[OK]    tesseract-ocr                ✓ PASS
[OK]    curl                         ✓ PASS
[OK]    sqlite3                      ✓ PASS

6. Ollama Connection
-----------------------------------
   Ollama host: http://ollama:11434
[OK]    Ollama service reachable     ✓ PASS
[OK]    llama3.2 model available     ✓ PASS
[OK]    nomic-embed-text available   ✓ PASS
[OK]    Ollama LLM functional        ✓ PASS
   LLM response successful
[OK]    Ollama embeddings functional ✓ PASS
   Embeddings response successful

7. Database Connection
-----------------------------------
   Database URL: postgresql://pdfuser:***@postgres:5432/pdfcompare
[OK]    Database connection          ✓ PASS

==================================
  Test Summary
==================================
Passed: 20
Failed: 0

✓ All tests passed! Container is ready.
```

**If tests fail:**

Check specific component based on failure:

```bash
# Test Python imports
docker exec pdf-compare-ui python -c "import pdf_compare"

# Test Ollama connection
docker exec pdf-compare-ui curl http://ollama:11434/api/version

# Test database
docker exec pdf-compare-ui python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.environ['DATABASE_URL'])
engine.connect()
print('Database OK')
"
```

### Test 2: UI Accessibility

```bash
# Check UI is responding
curl -f http://localhost:8501/_stcore/health

# Check specific endpoints
curl http://localhost:8501
```

**Expected: HTTP 200 response**

Open in browser: http://localhost:8501

**UI should show:**
- PDF Compare title
- File upload section
- Worker process slider (1-16)
- Performance settings

**If UI is not accessible:**

```bash
# Check Streamlit logs
docker logs pdf-compare-ui

# Check if port is bound
docker port pdf-compare-ui 8501

# Try different port
# Edit docker-compose-full.yml: "8502:8501"
```

### Test 3: RAG with Sample PDF

**Create test PDF:**

```bash
# Copy a test PDF into container
docker cp your_test_diagram.pdf pdf-compare-ui:/tmp/test.pdf
```

**Run RAG tests:**

```bash
docker exec pdf-compare-ui python test_rag.py /tmp/test.pdf
```

**Expected output:**

```
============================================================
RAG PDF ANALYSIS - TEST SUITE
============================================================
Testing Ollama connection...
  ✓ Ollama connection successful

Testing embedding model...
  ✓ Embeddings working (dimension: 768)

Testing PDF chat with: /tmp/test.pdf
  Loading PDF: /tmp/test.pdf
  Loaded 5 pages
  Split into 23 chunks
  Creating embeddings...
  Vector store created
  Asking test question...

  Answer: [LLM response about the document]

  ✓ PDF chat working!

Testing symbol extraction from: /tmp/test.pdf
  Extracting symbol legend...

  Found 12 symbols:
    - Control Valve: Main flow control
    - Pressure Gauge: Measures system pressure
    - Temperature Sensor: Monitors temperature
    - Safety Valve: Emergency pressure relief
    - Pump: Primary circulation pump
    ... and 7 more

  ✓ Symbol extraction working!

============================================================
TEST SUMMARY
============================================================
✓ PASS  - Ollama Connection
✓ PASS  - Embedding Model
✓ PASS  - PDF Chat
✓ PASS  - Symbol Extraction

Result: 4/4 tests passed

🎉 All tests passed! RAG system is ready to use.
```

**If RAG tests fail:**

```bash
# Test Ollama directly
docker exec pdf-compare-ollama ollama run llama3.2 "Say hello"

# Test embeddings
docker exec pdf-compare-ui python -c "
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')
result = embeddings.embed_query('test')
print(f'Embedding dimension: {len(result)}')
"

# Check Ollama host environment variable
docker exec pdf-compare-ui printenv OLLAMA_HOST
```

### Test 4: Interactive PDF Chat

```bash
# Start interactive chat
docker exec -it pdf-compare-ui python pdf_compare/rag_simple.py /tmp/test.pdf
```

**Test queries:**

```
Ask a question: What symbols are in the legend?
Ask a question: What is on page 3?
Ask a question: legend
Ask a question: pages
Ask a question: quit
```

**Expected: Intelligent responses based on PDF content**

### Test 5: PDF Extraction Performance

```bash
# Test extraction speed
docker exec pdf-compare-ui python -c "
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
import time

start = time.time()
vm = pdf_to_vectormap_server('/tmp/test.pdf', workers=15)
elapsed = time.time() - start

print(f'Pages extracted: {len(vm.pages)}')
print(f'Total time: {elapsed:.2f}s')
print(f'Speed: {len(vm.pages)/elapsed:.2f} pages/sec')
print(f'Workers: 15')

# Show stats
total_geoms = sum(len(p.geoms) for p in vm.pages)
total_texts = sum(len(p.texts) for p in vm.pages)
print(f'Geometries: {total_geoms}')
print(f'Text runs: {total_texts}')
"
```

**Expected performance:**
- Single-core: ~0.5 pages/sec
- Multi-core (15 workers): ~5-7 pages/sec

**If performance is poor:**

```bash
# Check CPU usage
docker stats pdf-compare-ui

# Should show ~100% CPU during extraction

# Check worker configuration
docker exec pdf-compare-ui printenv CPU_LIMIT

# Increase workers
# Edit docker-compose-full.yml:
environment:
  CPU_LIMIT: 15
```

### Test 6: Database Operations

```bash
# Test database connection and operations
docker exec pdf-compare-ui python -c "
from pdf_compare.db_backend import DatabaseBackend
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
import os

db = DatabaseBackend(os.environ['DATABASE_URL'])

# Extract PDF
vm = pdf_to_vectormap_server('/tmp/test.pdf', workers=1)

# Store in database
db.store_vectormap(vm)
print(f'Stored doc: {vm.meta.doc_id}')

# Retrieve
vm2 = db.get_vectormap(vm.meta.doc_id)
print(f'Retrieved doc with {len(vm2.pages)} pages')

# List all
docs = db.list_documents()
print(f'Total documents in DB: {len(docs)}')
"
```

**Expected: No errors, successful store/retrieve**

---

## Performance Benchmarks

### Expected Resource Usage

| Metric | Idle | Extraction | RAG Query |
|--------|------|------------|-----------|
| **CPU** | ~5% | ~90-100% | ~50-70% |
| **RAM** | ~2 GB | ~4-6 GB | ~3-4 GB |
| **Disk I/O** | Low | High | Medium |

Check with:

```bash
docker stats pdf-compare-ui pdf-compare-ollama
```

### Expected Speeds

| Operation | Speed | Notes |
|-----------|-------|-------|
| **PDF Extraction** | 5-7 pages/sec | 15 workers, 16-core CPU |
| **RAG Query** | 5-10 sec | First query (model load) |
| **RAG Query** | 1-3 sec | Subsequent queries |
| **Embedding** | 0.5 sec/page | nomic-embed-text |
| **LLM Response** | 2-5 sec | llama3.2, CPU-only |

---

## Troubleshooting

### Build Issues

**Error: `requirements.txt: no such file`**

```bash
# Verify file exists
ls -la requirements.txt

# Check .dockerignore doesn't exclude it
cat .dockerignore | grep requirements
```

**Error: `No space left on device`**

```bash
# Clean up Docker
docker system prune -a --volumes

# Check disk space
df -h
```

### Runtime Issues

**Container exits immediately:**

```bash
# Check logs
docker logs pdf-compare-ui

# Common causes:
# 1. Syntax error in code
# 2. Missing dependencies
# 3. Port conflict
```

**Ollama not responding:**

```bash
# Check Ollama is running
docker exec pdf-compare-ollama ps aux | grep ollama

# Restart Ollama
docker restart pdf-compare-ollama

# Check ollama-init completed
docker logs pdf-compare-ollama-init
```

**Database connection errors:**

```bash
# Check PostgreSQL is ready
docker exec pdf-compare-postgres pg_isready

# Test connection manually
docker exec pdf-compare-postgres psql -U pdfuser -d pdfcompare -c "SELECT 1"

# Check network
docker exec pdf-compare-ui ping postgres
```

### Cleanup and Reset

```bash
# Stop all services
docker-compose -f docker-compose-full.yml down

# Remove volumes (WARNING: Deletes data)
docker-compose -f docker-compose-full.yml down -v

# Clean Docker system
docker system prune -a

# Rebuild from scratch
docker-compose -f docker-compose-full.yml build --no-cache
docker-compose -f docker-compose-full.yml up -d
```

---

## Acceptance Criteria

Before considering deployment successful, verify:

- [ ] All containers are running and healthy
- [ ] Integration test suite passes (20/20 tests)
- [ ] UI is accessible at http://localhost:8501
- [ ] Ollama models are downloaded (llama3.2, nomic-embed-text)
- [ ] RAG tests pass with sample PDF
- [ ] PDF extraction completes successfully
- [ ] Database operations work (store/retrieve)
- [ ] Performance meets benchmarks (5+ pages/sec)
- [ ] No error messages in logs
- [ ] Can query PDF in interactive chat

---

## Production Readiness

Additional checks for production deployments:

- [ ] Change default passwords
- [ ] Enable TLS/HTTPS
- [ ] Configure firewall rules
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backup strategy
- [ ] Test disaster recovery
- [ ] Load testing completed
- [ ] Security audit passed

---

## Support

If tests fail after following this guide:

1. **Check logs:** `docker-compose logs -f`
2. **Verify versions:** `docker --version`, `docker-compose --version`
3. **Check resources:** `docker stats`, `df -h`
4. **Review docs:** See DOCKER_DEPLOYMENT.md
5. **Report issue:** Include test output and logs

---

## Quick Reference

```bash
# Build and test (automated)
.\docker-build-test.ps1 -DeploymentType full

# Manual build
docker-compose -f docker-compose-full.yml build

# Start
docker-compose -f docker-compose-full.yml up -d

# Test
docker exec pdf-compare-ui bash /app/docker_test.sh

# Logs
docker-compose logs -f

# Stop
docker-compose down

# Cleanup
docker-compose down -v
docker system prune -a
```
