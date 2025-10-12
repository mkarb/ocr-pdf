# Quick Reference - PDF Compare

## Installation & Setup

### First Time Setup
```powershell
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama (see INSTALL_OLLAMA_WINDOWS.md for details)
# Download from: https://ollama.com/download/windows

# 3. Pull AI models
ollama pull llama3.2
ollama pull nomic-embed-text

# 4. Verify setup
python setup_check.py
```

---

## Common Commands

### Check System Status
```powershell
# Verify all dependencies
python setup_check.py

# Check Ollama
ollama --version
ollama list

# Test AI features
python test_rag.py
```

### Launch UI
```powershell
# Start Streamlit UI
streamlit run ui/streamlit_app.py

# Access at: http://localhost:8501
```

### Test RAG with PDF
```powershell
# Full test suite
python test_rag.py your_diagram.pdf

# Interactive chat
python pdf_compare/rag_simple.py your_diagram.pdf

# Compare two PDFs
python pdf_compare/rag_simple.py old.pdf new.pdf
```

---

## Python Quick Start

### Chat with a PDF
```python
from pdf_compare.rag_simple import chat_with_pdf

# Load PDF
chat = chat_with_pdf("diagram.pdf")

# Ask questions
answer = chat.ask("What symbols are in the legend?")
print(answer)

# Extract legend
legend = chat.extract_symbol_legend()
print(legend)

# Find symbol instances
result = chat.find_symbol_instances("Control Valve")
print(result)
```

### Compare Two PDFs
```python
from pdf_compare.rag_simple import compare_pdfs

# Load both PDFs
comparator = compare_pdfs("old.pdf", "new.pdf")

# Get comparison
diff = comparator.compare_symbols()
print(diff)

# Check if specific symbols are the same
result = comparator.is_same_symbol(
    bbox1=(100, 200, 150, 250),
    context1="Main inlet line with control valve",
    bbox2=(105, 205, 160, 260),
    context2="Main inlet line with control valve"
)
print(result)
```

### Extract PDF Vectors
```python
from pdf_compare.pdf_extract_server import pdf_to_vectormap_server

# Extract with 15 workers
vectormap = pdf_to_vectormap_server(
    "diagram.pdf",
    workers=15
)

# Access data
for page in vectormap.pages:
    print(f"Page {page.page_number}: {len(page.geoms)} geometries, {len(page.texts)} text runs")
```

---

## Streamlit UI Features

### Performance Settings
- **Worker Processes**: Slider (1-16 cores)
  - Default: CPU count - 1
  - More workers = faster extraction
  - More RAM usage per worker

- **Debug Mode**: Toggle to show:
  - Extraction progress by page
  - Performance metrics (pages/sec)
  - Geometry/text counts

### Raster Comparison
- **DPI**: 300-600 (higher = more detail, slower)
- **Grid Size**: 5-50 pixels
  - Smaller = more granular
  - Larger = faster, less detail

- **Sensitivity**: 3-10%
  - Lower = more sensitive (more changes detected)
  - Higher = less sensitive (only major changes)

- **White Space Skipping**: Enabled by default
  - Speeds up comparison 50-80%
  - Skips empty areas

---

## Troubleshooting

### Ollama Issues

**"ollama: command not found"**
```powershell
# Restart PowerShell
# Or check installation:
Get-Command ollama
Test-Path "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
```

**"Failed to pull model"**
```powershell
# Check internet
Test-Connection ollama.com

# Try smaller model
ollama pull llama3.2:1b

# Check disk space
Get-PSDrive C
```

**"Connection refused"**
```powershell
# Start Ollama service
Start-Process ollama serve

# Or click Ollama icon in system tray
```

### RAG/AI Issues

**"No module named 'langchain'"**
```powershell
pip install -r requirements.txt
```

**"Model not found"**
```powershell
ollama list  # Check what's installed
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Slow responses**
- Use smaller model: `llama3.2:1b`
- Reduce chunk_size in SimplePDFChat
- Check CPU/GPU usage (should be high)

### Performance Issues

**PDF extraction using only 1 core**
- Check worker slider in UI (should be > 1)
- Default is CPU count - 1
- Max is 16

**Raster comparison highlighting everything**
- Increase sensitivity (5% → 10%)
- Enable white space skipping
- Check DPI is consistent between PDFs

**Out of memory**
- Reduce worker count
- Close other applications
- Use lower DPI for raster comparison

---

## File Locations

### Configuration
```
repo-root/
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Package configuration
└── docker-compose-postgres.yml  # Docker setup
```

### Code
```
repo-root/pdf_compare/
├── pdf_extract_server.py   # Vector extraction
├── raster_grid_improved.py # Raster comparison
├── rag_simple.py          # AI/RAG integration
└── db_backend.py          # Database abstraction
```

### Documentation
```
repo-root/
├── INSTALL_OLLAMA_WINDOWS.md  # Ollama setup
├── QUICK_START_RAG.md         # RAG quick start
├── RASTER_COMPARISON_GUIDE.md # Raster guide
└── IMPLEMENTATION_SUMMARY.md  # Complete features
```

### Data Storage
```
# SQLite (default)
repo-root/data/comparisons.db

# PostgreSQL (configured via DATABASE_URL)
postgresql://user:pass@host:5432/pdfcompare

# Ollama models
C:\Users\<Username>\.ollama\models\
```

---

## Environment Variables

```bash
# Performance
CPU_LIMIT=15
PDF_MIN_SEGMENT_LEN=0.50
PDF_BEZIER_SAMPLES=24

# Database
DATABASE_URL=sqlite:///./data/comparisons.db
# or
DATABASE_URL=postgresql://user:pass@host:5432/pdfcompare

# Ollama
OLLAMA_HOST=http://localhost:11434
```

---

## Docker Commands

```powershell
# Build and run
docker-compose -f docker-compose-postgres.yml up

# Run in background
docker-compose -f docker-compose-postgres.yml up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Access UI
# http://localhost:8501
```

---

## Useful Ollama Commands

```powershell
# Check version
ollama --version

# List models
ollama list

# Pull model
ollama pull <model-name>

# Remove model
ollama rm <model-name>

# Test model
ollama run llama3.2 "Say hello"

# Show model info
ollama show llama3.2

# Check service
Get-Process ollama
```

---

## Model Recommendations

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `llama3.2:1b` | ~1GB | Fast | Quick tests, simple queries |
| `llama3.2` (3B) | ~2GB | Medium | **General use (recommended)** |
| `llama3.2:8b` | ~4.7GB | Slower | Complex diagrams, high accuracy |

```powershell
# Switch models
ollama pull llama3.2:1b   # Fastest
ollama pull llama3.2      # Balanced (default)
ollama pull llama3.2:8b   # Most accurate
```

---

## Performance Tips

### Extraction Speed
1. **Use all available cores**
   - Set worker slider to max (CPU count)
   - Monitor CPU usage (should be ~100%)

2. **Use SSD for temporary files**
   - PDFs load faster from SSD
   - Raster comparison benefits from fast storage

3. **Process PDFs in batches**
   - Extract once, store in database
   - Reuse extractions for multiple comparisons

### RAG Performance
1. **Use smaller models for testing**
   - `llama3.2:1b` for quick experiments
   - `llama3.2` for production

2. **Reduce chunk size**
   - Smaller chunks = faster retrieval
   - But may miss context

3. **Enable GPU acceleration**
   - Ollama auto-uses NVIDIA GPU if available
   - Check with: `nvidia-smi`

### Memory Optimization
1. **Reduce worker count** if running out of RAM
   - Each worker uses 2-4 GB
   - Leave 4 GB for system

2. **Use lower DPI** for raster comparison
   - 300 DPI is usually sufficient
   - 600 DPI only for fine details

3. **Clear Ollama cache** if models pile up
   ```powershell
   ollama rm llama3.2:8b  # Remove large models
   ```

---

## Next Steps

1. **Run setup check**: `python setup_check.py`
2. **Test with your PDF**: `python test_rag.py your_diagram.pdf`
3. **Launch UI**: `streamlit run ui/streamlit_app.py`
4. **Read guides**: See [QUICK_START_RAG.md](QUICK_START_RAG.md)

---

## Support & Resources

- **Project Docs**: See repo-root/README.md
- **Ollama**: https://github.com/ollama/ollama
- **LangChain**: https://python.langchain.com/
- **Streamlit**: https://docs.streamlit.io/
