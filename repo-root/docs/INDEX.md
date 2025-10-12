# Documentation Index

Complete documentation for PDF Compare with AI-powered features.

## Getting Started

**New to the project? Start here:**

- **[Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md)** - End-to-end setup for local development and Docker deployment
- [Quick Reference](reference/QUICK_REFERENCE.md) - Common commands and quick answers

---

## Setup Guides

Step-by-step installation and configuration:

- [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) - Unified guide for local and Docker setup
- [Install Ollama (Windows)](setup/INSTALL_OLLAMA_WINDOWS.md) - Step-by-step Ollama installation for Windows

---

## Deployment

Docker containerization and production deployment:

- [Docker Deployment Guide](deployment/DOCKER_DEPLOYMENT.md) - Complete Docker deployment with all options
- [Docker Build Verification](deployment/DOCKER_BUILD_VERIFICATION.md) - Build testing checklist and troubleshooting
- [Docker Quick Start](deployment/DOCKER_QUICKSTART.md) - Fast Docker deployment
- [Docker Setup](deployment/DOCKER_SETUP.md) - Docker configuration details
- [Deployment Guide](deployment/DEPLOYMENT.md) - General deployment strategies

---

## Feature Guides

Using specific features:

- [RAG Quick Start](guides/QUICK_START_RAG.md) - AI/RAG features for symbol recognition
- [RAG Symbol Recognition](guides/RAG_SYMBOL_RECOGNITION_GUIDE.md) - Advanced symbol recognition with LLM
- [Raster Comparison Guide](guides/RASTER_COMPARISON_GUIDE.md) - Grid-based pixel comparison
- [Streamlit Features](guides/STREAMLIT_FEATURES.md) - Web UI features and configuration

---

## Technical Reference

Architecture and implementation details:

- [Quick Reference](reference/QUICK_REFERENCE.md) - Commands, troubleshooting, model selection
- [Implementation Summary](reference/IMPLEMENTATION_SUMMARY.md) - Complete feature list and technical details
- [Database Comparison](reference/DATABASE_COMPARISON.md) - SQLite vs PostgreSQL
- [Server Mode Comparison](reference/SERVER_MODE_COMPARISON.md) - Processing modes comparison
- [Server Mode README](reference/SERVER_MODE_README.md) - Server-side processing details

---

## Documentation by Task

### I want to install and run the application

**Local development:**
1. [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) → Local Development Setup section
2. [Install Ollama](setup/INSTALL_OLLAMA_WINDOWS.md)
3. [Quick Reference](reference/QUICK_REFERENCE.md) → Installation section

**Docker deployment:**
1. [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) → Docker Deployment section
2. [Docker Deployment Guide](deployment/DOCKER_DEPLOYMENT.md)
3. [Docker Build Verification](deployment/DOCKER_BUILD_VERIFICATION.md)

### I want to use AI features to analyze PDFs

1. [RAG Quick Start](guides/QUICK_START_RAG.md) - Get started in 3 lines of code
2. [RAG Symbol Recognition](guides/RAG_SYMBOL_RECOGNITION_GUIDE.md) - Advanced features
3. [Quick Reference](reference/QUICK_REFERENCE.md) → Python Quick Start section

### I want to compare two PDF revisions

1. [Streamlit Features](guides/STREAMLIT_FEATURES.md) - Using the web UI
2. [Raster Comparison Guide](guides/RASTER_COMPARISON_GUIDE.md) - Pixel-level comparison
3. [Quick Reference](reference/QUICK_REFERENCE.md) → Common Commands section

### I want to deploy to production

1. [Docker Deployment Guide](deployment/DOCKER_DEPLOYMENT.md) → Production Deployment section
2. [Database Comparison](reference/DATABASE_COMPARISON.md) - Choose database backend
3. [Server Mode Comparison](reference/SERVER_MODE_COMPARISON.md) - Choose processing mode

### I'm troubleshooting an issue

1. [Quick Reference](reference/QUICK_REFERENCE.md) → Troubleshooting section
2. [Docker Build Verification](deployment/DOCKER_BUILD_VERIFICATION.md) → Troubleshooting section
3. [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) → Troubleshooting section

---

## Documentation Structure

```
docs/
├── INDEX.md                    # This file
├── setup/                      # Installation and initial setup
│   ├── COMPLETE_SETUP_GUIDE.md
│   └── INSTALL_OLLAMA_WINDOWS.md
├── deployment/                 # Docker and production deployment
│   ├── DOCKER_DEPLOYMENT.md
│   ├── DOCKER_BUILD_VERIFICATION.md
│   ├── DOCKER_QUICKSTART.md
│   ├── DOCKER_SETUP.md
│   └── DEPLOYMENT.md
├── guides/                     # Feature-specific guides
│   ├── QUICK_START_RAG.md
│   ├── RAG_SYMBOL_RECOGNITION_GUIDE.md
│   ├── RASTER_COMPARISON_GUIDE.md
│   └── STREAMLIT_FEATURES.md
└── reference/                  # Technical reference
    ├── QUICK_REFERENCE.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── DATABASE_COMPARISON.md
    ├── SERVER_MODE_COMPARISON.md
    └── SERVER_MODE_README.md
```

---

## Quick Links

### Most Common Tasks

| Task | Documentation |
|------|---------------|
| **First-time setup** | [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) |
| **Docker deployment** | [Docker Deployment](deployment/DOCKER_DEPLOYMENT.md) |
| **AI/RAG features** | [RAG Quick Start](guides/QUICK_START_RAG.md) |
| **Command reference** | [Quick Reference](reference/QUICK_REFERENCE.md) |
| **Troubleshooting** | [Quick Reference](reference/QUICK_REFERENCE.md#troubleshooting) |

### By User Role

**Developer:**
- [Complete Setup Guide](setup/COMPLETE_SETUP_GUIDE.md) - Local development
- [Implementation Summary](reference/IMPLEMENTATION_SUMMARY.md) - Architecture
- [Server Mode Comparison](reference/SERVER_MODE_COMPARISON.md) - Processing modes

**DevOps/SysAdmin:**
- [Docker Deployment Guide](deployment/DOCKER_DEPLOYMENT.md) - Container deployment
- [Docker Build Verification](deployment/DOCKER_BUILD_VERIFICATION.md) - Testing
- [Database Comparison](reference/DATABASE_COMPARISON.md) - Database selection

**End User:**
- [Streamlit Features](guides/STREAMLIT_FEATURES.md) - Using the UI
- [RAG Quick Start](guides/QUICK_START_RAG.md) - AI features
- [Quick Reference](reference/QUICK_REFERENCE.md) - Common commands

---

## Contributing to Documentation

When adding new documentation:

1. **Setup guides** → `docs/setup/` - Installation and configuration
2. **Deployment guides** → `docs/deployment/` - Docker and production
3. **Feature guides** → `docs/guides/` - How to use specific features
4. **Technical reference** → `docs/reference/` - Architecture and implementation

Update this index when adding new documentation files.

---

## External Resources

- **Ollama**: https://github.com/ollama/ollama
- **LangChain**: https://python.langchain.com/
- **Streamlit**: https://docs.streamlit.io/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **Docker**: https://docs.docker.com/
