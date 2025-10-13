# Docker Container Architecture

## Overview

This project uses a **multi-container architecture** where each service runs in its own container and can be managed independently.

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network (pdf-network)             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  POSTGRES    │  │   OLLAMA     │  │   APP (UI)       │  │
│  │              │  │              │  │                  │  │
│  │ Pre-built    │  │ Pre-built    │  │ Built locally   │  │
│  │ Image        │  │ Image        │  │ from Dockerfile │  │
│  │              │  │              │  │                  │  │
│  │ postgis/     │  │ ollama/      │  │ Multi-stage     │  │
│  │ postgis      │  │ ollama       │  │ build           │  │
│  │              │  │              │  │                  │  │
│  │ Port: 5432   │  │ Port: 11434  │  │ Port: 8501      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Volume:      │  │ Volume:      │  │ Volumes:         │  │
│  │ postgres-    │  │ ollama-      │  │ ./data/uploads   │  │
│  │ data         │  │ data         │  │ ./data/outputs   │  │
│  │ (~500MB)     │  │ (~4GB)       │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Container Details

### 1. PostgreSQL Container (`postgres`)
- **Image**: `postgis/postgis:16-3.4` (pre-built, pulled from Docker Hub)
- **Build time**: ~1 minute (first time, just pulls image)
- **Rebuild frequency**: Rarely (only when upgrading Postgres version)
- **Data**: Persisted in `postgres-data` volume
- **Port**: 5432
- **Purpose**: Stores comparison results, metadata, and spatial data

**Update command (Bash)**:
```bash
./build.sh postgres down
docker pull postgis/postgis:16-3.4
./build.sh postgres up
```

**Update command (PowerShell)**:
```powershell
.\build.ps1 postgres down
docker pull postgis/postgis:16-3.4
.\build.ps1 postgres up
```

### 2. Ollama Container (`ollama`)
- **Image**: `ollama/ollama:latest` (pre-built, pulled from Docker Hub)
- **Build time**: ~5 minutes first time (downloads models)
- **Rebuild frequency**: Rarely (only when upgrading Ollama version)
- **Data**: Models persisted in `ollama-data` volume (~4GB)
- **Port**: 11434
- **Purpose**: Provides AI/LLM capabilities for RAG features
- **Models**: llama3.2, nomic-embed-text

**Update command (Bash)**:
```bash
./build.sh ollama down
docker pull ollama/ollama:latest
./build.sh ollama up
```

**Update command (PowerShell)**:
```powershell
.\build.ps1 ollama down
docker pull ollama/ollama:latest
.\build.ps1 ollama up
```

### 3. Application Container (`pdf-compare-ui`)
- **Image**: Built locally from `Dockerfile` (multi-stage)
- **Build time**:
  - First time: ~10-15 minutes (installs all dependencies)
  - Code updates: ~30 seconds (only rebuilds app layer)
- **Rebuild frequency**: **FREQUENTLY** (every time you update code)
- **Data**: Source code mounted from `./ui/` and `./pdf_compare/`
- **Port**: 8501
- **Purpose**: Streamlit UI + PDF comparison logic

**Update command (Bash)** - fast!:
```bash
./build.sh app build
./build.sh app restart
```

**Update command (PowerShell)** - fast!:
```powershell
.\build.ps1 app build
.\build.ps1 app restart
```

## Multi-Stage Dockerfile

The Dockerfile uses a 3-stage build to optimize rebuild times:

```dockerfile
Stage 1: base (system dependencies)
   ↓ rarely changes
Stage 2: dependencies (Python packages)
   ↓ changes when requirements.txt changes
Stage 3: app (your code)
   ↓ changes FREQUENTLY
```

When you update code, Docker only rebuilds Stage 3, reusing cached layers from Stages 1 and 2.

## Independent Management

Each container can be managed independently:

### Bash (Linux/macOS/Git Bash):
```bash
# Start everything
./build.sh full up

# OR start containers individually:
./build.sh postgres up    # Just database
./build.sh ollama up      # Just AI service
./build.sh app up         # Just application

# Update only the app (doesn't affect Postgres/Ollama):
./build.sh app build
./build.sh app restart

# Restart individual services:
./build.sh postgres restart
./build.sh ollama restart
./build.sh app restart
```

### PowerShell (Windows):
```powershell
# Start everything
.\build.ps1 full up

# OR start containers individually:
.\build.ps1 postgres up    # Just database
.\build.ps1 ollama up      # Just AI service
.\build.ps1 app up         # Just application

# Update only the app (doesn't affect Postgres/Ollama):
.\build.ps1 app build
.\build.ps1 app restart

# Restart individual services:
.\build.ps1 postgres restart
.\build.ps1 ollama restart
.\build.ps1 app restart
```

## Why This Architecture?

### Benefits:
1. **Fast code updates**: Rebuild only app layer (~30s vs 10+ min)
2. **No database disruption**: Postgres keeps running, data intact
3. **No model re-downloads**: Ollama models stay cached
4. **Independent scaling**: Scale services separately
5. **Easier debugging**: Isolate issues to specific containers
6. **Flexible deployment**: Run only what you need

### Typical Workflow:

**Bash:**
```bash
# Day 1: Initial setup
./build.sh full up              # Takes ~15 minutes

# Day 2-N: Development
# Edit code in ui/ or pdf_compare/
./build.sh app build            # Takes ~30 seconds
./build.sh app restart          # Takes ~5 seconds
# Postgres and Ollama never stopped!
```

**PowerShell:**
```powershell
# Day 1: Initial setup
.\build.ps1 full up             # Takes ~15 minutes

# Day 2-N: Development
# Edit code in ui/ or pdf_compare/
.\build.ps1 app build           # Takes ~30 seconds
.\build.ps1 app restart         # Takes ~5 seconds
# Postgres and Ollama never stopped!
```

## Data Persistence

All data is preserved in Docker volumes:

- **postgres-data**: Database contents (~500MB+)
- **ollama-data**: AI models (~4GB)
- **./data/uploads**: User uploads (host directory)
- **./data/outputs**: Generated outputs (host directory)

Stop and start containers freely - your data persists!

## Resource Usage

Typical resource footprint:
- **Postgres**: ~200MB RAM, minimal CPU
- **Ollama**: ~2-4GB RAM (depends on model), CPU/GPU for inference
- **App**: ~1-2GB RAM, high CPU during PDF processing

Configure resource limits in `docker-compose-full.yml`.
