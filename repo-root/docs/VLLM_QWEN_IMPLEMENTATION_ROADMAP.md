# vLLM + Qwen2.5 Integration Implementation Roadmap

**Document Version**: 1.0
**Date**: 2025-01-18
**Status**: Planning Phase

---

## Executive Summary

This document outlines the integration of vLLM with Qwen2.5 models running on AMD GPUs (9070 XT 16GB + 9060 XT 12GB) to enhance PDF analysis capabilities while preserving existing VectorMap extraction functionality.

**Key Decision**: **KEEP** all existing extraction - VectorMap and text are critical and irreplaceable.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  (User uploads PDF, requests comparison/extraction)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PDF Ingestion Pipeline                         │
│                                                                   │
│  1. pdf_extract.py → VectorMap (geoms, texts, bboxes)           │
│  2. Store in PostgreSQL (db_backend.py)                         │
│  3. [NEW] vLLM Layout Analyzer → Semantic enrichment            │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌─────────────────┐ ┌──────────────┐
│ RAG System   │ │ Table Extractor │ │ Comparison   │
│              │ │                 │ │              │
│ [ENHANCED]   │ │ [HYBRID]        │ │ [ENHANCED]   │
│ Ollama +     │ │ CV + vLLM       │ │ vLLM spatial │
│ vLLM Qwen2.5 │ │ Qwen2.5         │ │ reasoning    │
└──────────────┘ └─────────────────┘ └──────────────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     vLLM Service Layer                           │
│                                                                   │
│  • Model: Qwen2.5-7B-Instruct (primary)                         │
│  • Model: Qwen2.5-14B-Instruct (high-quality mode)              │
│  • Backend: ROCm (AMD 9070 XT + 9060 XT)                        │
│  • Serving: vLLM with tensor parallelism                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table Extraction Analysis

### Current Implementation (KEEP - DO NOT DEPRECATE)

**File**: `pdf_compare/analyzers/table_extractor.py` (1347 lines)

**Approach**: Pure computer vision
- Line Detection (Hough Transform) → bordered tables
- Whitespace Analysis → borderless tables
- OCR (Tesseract/EasyOCR) → cell text extraction
- VectorMap Integration → uses stroke geometry for grid lines
- Keyword Matching → classify table type (BOM, symbols, etc.)

**Strengths**:
- ✅ No external dependencies (works offline)
- ✅ Fast for well-structured tables
- ✅ Integrates with VectorMap geometry
- ✅ BOM-specific grid reconstruction from vectors
- ✅ Handles both bordered and borderless tables

**Weaknesses**:
- ❌ Fragile with complex layouts (merged cells, irregular spacing)
- ❌ Requires manual tuning (thresholds, keywords)
- ❌ No semantic understanding (can't infer missing structure)
- ❌ Poor with hand-drawn or scanned tables
- ❌ Column alignment relies on pixel-perfect spacing

### Recommended Strategy: Hybrid Approach

**DO NOT REMOVE** existing CV-based extraction. Instead, add LLM-based fallback:

```python
class HybridTableExtractor:
    """Use CV for simple tables, vLLM for complex ones."""

    def __init__(self):
        self.cv_extractor = TableExtractor(config)  # Current implementation
        self.llm_extractor = VLLMTableExtractor(model="Qwen2.5-7B")

    def extract_tables(self, page, vectormap):
        # Try CV first (fast path)
        cv_tables = self.cv_extractor.extract(page, vectormap)

        # Validate CV results
        validated_tables = []
        failed_regions = []

        for table in cv_tables:
            if self._is_high_quality(table):
                validated_tables.append(table)
            else:
                failed_regions.append(table.bbox)

        # Use LLM for failed regions (slow but accurate)
        if failed_regions:
            llm_tables = self.llm_extractor.extract_regions(
                page,
                regions=failed_regions,
                vectormap_context=vectormap
            )
            validated_tables.extend(llm_tables)

        return validated_tables
```

---

## Phase 1: Foundation (Week 1-2)

**Goal**: Set up vLLM with ROCm and validate GPU acceleration

### Tasks

1. ✅ Verify ROCm installation and GPU detection
2. ✅ Install vLLM with ROCm support
3. ✅ Download Qwen2.5-7B-Instruct model
4. ✅ Create benchmark script to compare Ollama vs vLLM performance
5. ✅ Test multi-GPU tensor parallelism

### Files to Create

- `pdf_compare/llm_service.py` - Service manager
- `scripts/benchmark_llm.py` - Performance testing
- `docker/Dockerfile.vllm` - ROCm + vLLM container

### Success Metrics

- vLLM running on both AMD GPUs
- 3-5x throughput improvement over Ollama
- <100ms first-token latency

---

## Phase 2: VectorMap Integration (Week 3-4)

**Goal**: Connect VectorMap geometric data to LLM reasoning

**CRITICAL**: VectorMap extraction remains unchanged and mandatory.

### Tasks

1. ✅ Implement `VLLMLayoutAnalyzer` class
2. ✅ Create VectorMap → text context converter
3. ✅ Add geometric feature extraction helpers
4. ✅ Test layout analysis on sample engineering PDFs
5. ✅ Integrate with existing ingestion pipeline (optional flag)

### Files to Create/Modify

- `pdf_compare/llm_layout_analyzer.py` - NEW
- `pdf_compare/pdf_extract.py` - Add optional layout analysis step (VectorMap extraction unchanged)
- `pdf_compare/models.py` - Add `PageLayout` dataclass

### Success Metrics

- Correctly identify page zones (title block, legend, main diagram)
- 80%+ accuracy on symbol-label association
- <5 seconds per page analysis
- **VectorMap extraction time unchanged**

---

## Phase 3: Hybrid Table Extraction (Week 5-6)

**Goal**: Enhance table extraction with LLM fallback

### Tasks

1. ✅ Implement `VLLMTableExtractor` class
2. ✅ Add quality validation to CV-based extraction
3. ✅ Create hybrid decision logic (CV first, LLM fallback)
4. ✅ Test on problematic tables (merged cells, irregular spacing)
5. ✅ Update Streamlit UI with "High Quality Mode" toggle

### Files to Modify

- `pdf_compare/analyzers/table_extractor.py` - Add `VLLMTableExtractor` (keep all existing code)
- `pdf_compare/table_workflows.py` - Add hybrid mode
- `ui/streamlit_app.py` - Add UI toggle

### Success Metrics

- 95%+ accuracy on complex tables (vs current 70%)
- <10 seconds per table (LLM path)
- 80% of tables still use fast CV path

---

## Phase 4: Enhanced RAG (Week 7-8)

**Goal**: Upgrade RAG system with geometric context

### Tasks

1. ✅ Enhance text embeddings with geometric features
2. ✅ Implement `ask_spatial()` method
3. ✅ Add VectorMap context to retrieval
4. ✅ Update Streamlit chat interface
5. ✅ Create example spatial queries

### Files to Modify

- `pdf_compare/rag_simple.py` - Extend with geometry
- `pdf_compare/rag_symbol_recognition.py` - Use vLLM backend
- `ui/streamlit_app.py` - Update chat interface

### Success Metrics

- Answer spatial questions ("What's left of valve PV-101?")
- 90%+ accuracy on symbol identification
- Support 10+ simultaneous chat sessions

---

## Phase 5: Production Deployment (Week 9-10)

**Goal**: Containerize and optimize for production

### Tasks

1. ✅ Create Docker Compose setup with vLLM service
2. ✅ Add health checks and monitoring
3. ✅ Implement request queueing
4. ✅ Add caching layer for repeated queries
5. ✅ Write deployment documentation

### Files to Create

- `docker-compose-vllm.yml` - Full stack with vLLM
- `docker/vllm-service/` - Standalone vLLM container
- `docs/VLLM_DEPLOYMENT.md` - Deployment guide

### Success Metrics

- Handle 100+ concurrent requests
- <1 second p95 latency
- Auto-restart on failure

---

## Deprecation Decision Matrix

### What to Keep (DO NOT REMOVE)

| Component | Status | Reason |
|-----------|--------|--------|
| **VectorMap extraction** | ✅ CRITICAL - KEEP | Core functionality, irreplaceable |
| **CV-based table extraction** | ✅ KEEP | Fast, deterministic, works for 80% of cases |
| **Text extraction pipeline** | ✅ CRITICAL - KEEP | Required for all downstream processing |
| **Ollama integration** | ✅ KEEP | Good for development/quick queries |
| **EasyOCR** | ✅ KEEP | Better OCR than Tesseract for small text |
| **RAG symbol recognition** | ✅ ENHANCE | Add VectorMap context |

### What to Deprecate

| Component | Status | Reason | Replacement |
|-----------|--------|--------|-------------|
| **Pure text-based RAG** | ⚠️ DEPRECATE | No geometric understanding | Enhanced RAG with VectorMap |
| **Simple keyword table classification** | ⚠️ DEPRECATE | Too rigid | vLLM semantic classification |
| **HuggingFace embeddings (CPU)** | ⚠️ DEPRECATE | Slow, not using AMD GPUs | vLLM embeddings (optional) |

### What to Add

| Component | Priority | Why |
|-----------|----------|-----|
| **vLLM service layer** | 🔴 HIGH | Core enabler for all enhancements |
| **VLLMLayoutAnalyzer** | 🔴 HIGH | Unlock semantic understanding |
| **Hybrid table extraction** | 🟡 MEDIUM | Improve accuracy on complex tables |
| **Geometric embeddings** | 🟡 MEDIUM | Better similarity search |
| **Multi-GPU orchestration** | 🟢 LOW | Optimization, not critical |

---

## Configuration

### Environment Variables

```bash
# Existing (unchanged)
DATABASE_URL=postgresql://user:pass@host:5432/pdfcompare
OLLAMA_HOST=http://localhost:11434

# New vLLM settings
VLLM_ENABLED=true                              # Enable vLLM features
VLLM_HOST=http://localhost:8000                # vLLM service endpoint
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct           # Model to load
VLLM_TENSOR_PARALLEL=2                         # Use both AMD GPUs
VLLM_GPU_MEMORY_UTILIZATION=0.85              # GPU memory usage
VLLM_MAX_MODEL_LEN=4096                       # Context length

# Feature flags
ENABLE_LAYOUT_ANALYSIS=false                   # VectorMap → semantic layout
ENABLE_LLM_TABLE_EXTRACTION=false              # Hybrid table extraction
ENABLE_GEOMETRIC_EMBEDDINGS=false              # Enhanced RAG
```

### Docker Compose Example

```yaml
# docker-compose-vllm.yml
version: '3.8'

services:
  postgres:
    # ... existing config

  vllm-service:
    image: vllm/vllm-openai:rocm6.3
    container_name: pdf-compare-vllm
    environment:
      - VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
      - VLLM_TENSOR_PARALLEL_SIZE=2
      - VLLM_GPU_MEMORY_UTILIZATION=0.85
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.85
      --trust-remote-code
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit-ui:
    # ... existing config
    environment:
      - VLLM_ENABLED=true
      - VLLM_HOST=http://vllm-service:8000
    depends_on:
      vllm-service:
        condition: service_healthy
```

---

## Migration Path (Zero-Downtime)

### Phase 1: Parallel Run (Month 1)

```python
# All new features behind feature flags
if os.getenv("ENABLE_LAYOUT_ANALYSIS") == "true":
    layout = vllm_analyzer.analyze_layout(page)
else:
    layout = None  # Old behavior
```

### Phase 2: Opt-In (Month 2)

```python
# Streamlit UI checkbox
use_llm_features = st.sidebar.checkbox(
    "Enable AI Layout Analysis (Beta)",
    value=False,
    help="Use vLLM for enhanced understanding"
)
```

### Phase 3: Default On (Month 3)

```python
# Make new features default, old path is fallback
try:
    layout = vllm_analyzer.analyze_layout(page)
except Exception:
    logger.warning("vLLM unavailable, using basic analysis")
    layout = basic_layout_analysis(page)
```

### Phase 4: Full Migration (Month 4+)

```python
# Remove old code paths after validation
layout = vllm_analyzer.analyze_layout(page)  # No fallback
```

---

## Model Selection: Qwen2.5 Series

### Primary Model: Qwen2.5-7B-Instruct

**Best for**:
- Standard layout analysis
- Table extraction
- Symbol identification
- General spatial reasoning

**Performance on AMD 9070 XT**:
- Speed: ~80 tokens/sec
- VRAM: ~8GB
- Latency: <100ms first token

### High-Quality Model: Qwen2.5-14B-Instruct

**Best for**:
- Complex multi-page analysis
- Ambiguous symbol recognition
- High-accuracy requirements

**Performance on Both GPUs (Tensor Parallel)**:
- Speed: ~40-50 tokens/sec
- VRAM: ~8GB per GPU (16GB total)
- Latency: ~150ms first token

---

## Critical Reminders

### DO THIS ✅

1. **ALWAYS extract VectorMap** - Non-negotiable, core functionality
2. **ALWAYS extract text** - Required for all downstream processing
3. Keep CV-based table extraction - Enhance, don't replace
4. Use feature flags for gradual rollout
5. Test on AMD GPUs before committing
6. Maintain backward compatibility

### DON'T DO THIS ❌

1. **NEVER skip VectorMap extraction** - Everything depends on it
2. **NEVER skip text extraction** - Critical for search/comparison
3. Don't remove existing table extraction code
4. Don't make vLLM required (should be optional)
5. Don't force users to migrate
6. Don't assume NVIDIA GPU (we're AMD-only)

---

## File Structure Changes

### New Files

```
pdf_compare/
├── llm_service.py                 # LLM service manager (Ollama + vLLM)
├── llm_layout_analyzer.py         # VectorMap → semantic layout
├── llm_table_extractor.py         # LLM-based table extraction
└── utils/
    ├── geometric_features.py      # VectorMap feature extraction
    └── spatial_clustering.py      # Geometric clustering helpers

docker/
├── vllm-service/
│   ├── Dockerfile                 # ROCm + vLLM image
│   └── entrypoint.sh
└── docker-compose-vllm.yml        # Full stack with vLLM

docs/
├── VLLM_INTEGRATION.md           # Integration guide
├── VLLM_DEPLOYMENT.md            # Deployment guide
└── VECTORMAP_SEMANTICS.md        # How VectorMap → semantics works
```

### Modified Files (VectorMap extraction UNCHANGED)

```
pdf_compare/
├── pdf_extract.py                 # Add optional layout analysis (after VectorMap)
├── rag_simple.py                  # Enhance with geometric context
├── table_extractor.py             # Wrapper for hybrid extraction
├── table_workflows.py             # Add LLM mode
└── analyzers/
    └── table_extractor.py         # Add validation + VLLMTableExtractor

ui/
└── streamlit_app.py              # Add vLLM health status, mode toggles
```

---

## Next Steps

### Immediate Actions

1. **Verify ROCm + AMD GPU setup**
   ```bash
   rocm-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install vLLM**
   ```bash
   pip install vllm
   ```

3. **Test Qwen2.5-7B**
   ```python
   from vllm import LLM
   llm = LLM("Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=2)
   ```

### Week 1 Focus

- Set up vLLM service
- Benchmark vs Ollama
- Validate multi-GPU support
- Create `llm_service.py` foundation

---

## Success Criteria

### Must Have

- ✅ VectorMap extraction unchanged and working
- ✅ Text extraction unchanged and working
- ✅ vLLM running on AMD GPUs
- ✅ Backward compatibility maintained
- ✅ Feature flags for all new features

### Should Have

- ✅ 3x performance improvement over Ollama
- ✅ 90%+ accuracy on layout analysis
- ✅ <5 second per-page processing
- ✅ Hybrid table extraction working

### Nice to Have

- Multi-GPU orchestration
- Advanced caching
- Real-time monitoring
- Auto-scaling

---

**Document Control**
Version: 1.0
Last Updated: 2025-01-18
Next Review: After Phase 1 completion
