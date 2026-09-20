# Deficiency Report — ocr-pdf / PDF Compare

Generated as a section-by-section audit. Severity legend:
**🔴 High** (breaks/blocks or correctness risk) · **🟡 Medium** (maintainability / latent bug) · **🟢 Low** (hygiene / polish)

**Project goal (active):** clean up the project to a functional state for **high-resolution OCR on large engineering diagrams**. Not pivoting. Handwriting OCR is a possible future, separate channel — out of scope now. Prioritize fixes that make the existing extraction → store → OCR → tables/search/overlay path work end to end.

Section status: all 10 sections reviewed; fixes applied per section (see each section's "Resolved this pass"). Pass 1 merged (PRs #1, #2). **Second pass below** — adversarial/correctness focus on the large-diagram OCR path.

---

## Second Pass — correctness on large diagrams

**Resolved this pass:** S1 (new shared `highres_ocr.ocr_page()` auto-tiles oversized pages; both `pdf_extract._extract_text` and CLI `_run_ocr_augment` now route through it, so `ocr-augment`/`compare --with-ocr` work on large sheets and the duplicate OCR-a-page logic is gone — tests in `tests/test_ocr_page.py`), S2 (bounded `submit_throttled` stall detection — aborts with context after 3×300 s of no progress instead of looping forever).
**Still open:** S5 (configurable OCR threshold).
**Resolved (run-experience pass):** S4 — OCR errors in `_extract_text` are no longer swallowed to stderr; they log at ERROR with traceback via a module logger, the debug prints became `logger.debug`, and the UI surfaces OCR effectiveness after ingest (`OCR added N spans` / a warning when OCR produced nothing). Pairs with 7.4 below.

### S6 🔴 `table_extractor` renders the whole page at full DPI (BOM path) — last tiling gap — RESOLVED
`detect_table_regions` and `extract_table` used to `get_pixmap` the **entire page** at `config.dpi` (400) — a ~17000×11000 pixmap on an E-size sheet — and `extract_table` rendered the full page *then* cropped, so memory peaked on the whole sheet even for a small table.
- **Fixed:** added `_capped_dpi` + `TableExtractionConfig.max_render_pixels` (8000). `detect_table_regions` now renders at a capped DPI (region-level detection doesn't need full res). `extract_table` renders **only the table region** via `get_pixmap(clip=...)` at full DPI when a bbox is known (cell OCR stays sharp, memory bounded to the table), and caps the no-bbox whole-page render. Coordinate mapping unchanged (offset = region top-left in full-page pixels). Tests in `tests/test_table_render_cap.py`. This was the last whole-page-render tiling gap (only the `overlay.py` raster fallback, S7, remains — low severity).

### S7 🟢 Overlay raster fallback renders whole page (`overlay.py:97`)
Fixed `Matrix(3,3)` (~216 DPI) whole-page render, but only on the fallback path (layered PDFs where direct annotation fails). Low severity.

**Resolved (third pass):** S3 — `raster_grid` now caps the render resolution (`max_render_pixels=8000`, one effective DPI for both pages) so large sheets are change-detected at a tractable size instead of rendering/ECC-aligning a ~17000×11000 image. Output boxes are PDF-space so coordinates are unchanged; `render_dpi` surfaced in metrics. Tests in `tests/test_raster_render_cap.py`.

### S1 🔴 `ocr-augment` / `compare --with-ocr` don't tile large pages → OOM / Tesseract limit
The ingest OCR path (`pdf_extract._extract_text`) checks `needs_tiling` and calls `tiled_ocr` for big pages. But `_run_ocr_augment` (CLI `ocr-augment` **and** `compare --with-ocr`) calls `highres_ocr(...)`, which renders the **whole page** at `cfg.dpi` (`_render_page_gray`) with no size check. On a large E-size sheet at 500 DPI that pixmap is ~150–500 MB and exceeds Tesseract's ~32,767-px hard limit → the augment path fails or OOMs on exactly the diagrams that are the project's purpose. (You said tiling is the key — and one of the two OCR entry points skips it.)
- **Fix:** factor a shared `ocr_page(...)` in `highres_ocr` that auto-tiles (the ingest logic), and route both `pdf_extract._extract_text` and `_run_ocr_augment` through it. Removes the duplicate OCR-a-page logic too.

### S2 🟡 `submit_throttled` can hang forever on a stuck worker (revisit of 2.6)
The `wait(..., timeout=300)` branch only logs and `continue`s. If a worker genuinely wedges (e.g. an OCR worker on a pathological page), the loop spins every 300 s indefinitely with no abort. On big parallel OCR jobs that's a real hang.
- **Fix:** bound consecutive empty waits (e.g. 3) then raise with context.

### S3 🟡 Raster compare renders + ECC-aligns the whole page (`raster_grid._render_gray`)
`raster_grid_changed_boxes` renders both pages whole at `dpi` (400) and runs `cv2.findTransformECC` on the full image. For a large sheet that's a multi-hundred-MB pair plus an ECC solve on a ~17000×11000 image — very slow and memory-heavy, and ECC can fail to converge at that size. The `compare-grid` CLI path is impractical on large diagrams without downscaling the alignment step.

### S4 🟡 OCR failures are swallowed silently (large-page UX)
`_extract_text` wraps the whole OCR block in `except Exception` → prints a traceback to **stderr** and returns only native text. Combined with S1, a large-page OCR failure looks like "document has no text" in the UI with no surfaced error. Worth surfacing the failure (and it interacts with 7.4's missing progress).

### S5 🟢 Native-text OCR threshold is a fixed `< 20` spans
`_extract_text` only OCRs when a page has `< 20` native spans. A scanned drawing whose title block carries ≥20 native spans but whose body is raster will skip OCR entirely. The threshold is arbitrary/uncongurable.

---

## Section 1 — Project hygiene & structure

**Resolved this pass:** 1.1 (pyproject synced), 1.2 (dead `store.py` deleted), 1.3 (wrapper renamed `table_extractor.py`→`table_extract_api.py`), 1.4 (`raster_diff.py` deleted, `raster_diff_debug.py`→`tools/`), 1.6 (stray files deleted), 1.7 (`outputs/` untracked + `.gitignore` fixed).
**Still open:** 1.5 (pdf_extract dedup → Section 2), 1.8 (test coverage), 1.9 (build scripts → Section 9).

### 1.1 🟡 `requirements.txt` and `pyproject.toml` dependency lists diverge
Both files exist in `repo-root/`. `requirements.txt` (the README's documented install path, `pip install -r requirements.txt`) is the **complete** list. `pyproject.toml` lists the same deps but **omits four**: `threadpoolctl`, `easyocr`, `rapidfuzz`, `pandas` — all of which the code imports (`easyocr`/`pandas` in the OCR/table analyzers, `rapidfuzz` for symbol validation).
- **Impact:** the documented `pip install -r requirements.txt` path works fine. The risk is the *packaged* path: `pyproject` defines a console script (`compare-pdf-revs`) and a build backend, so `pip install .` and any wheel/Docker build from pyproject yield an env where those imports fail at runtime. Two sources of truth, silently out of sync.
- **Fix:** make one authoritative. Either have `pyproject` read deps dynamically from `requirements.txt`, or add the four missing packages to `pyproject` and keep them in sync.

### 1.2 🟡 Three "`_new`" modules shadow their originals
`compare_new.py`, `store_new.py`, `search_new.py` are the canonical modules (imported by `__init__.py`, `cli.py`, `ui/streamlit_app.py`). The originals remain:
- `store.py` (142 lines) — **fully orphaned**, no importers anywhere.
- `compare.py` (360 lines) and `search.py` (46 lines) — imported only as *legacy fallbacks* inside the `_new` versions (`from .compare import diff_documents as legacy_diff`, `from .search import ...`).
- **Issue:** the `_new` suffix is a code smell that signals an incomplete migration. Readers can't tell which is authoritative without tracing imports. `store.py` is dead weight.
- **Fix:** delete `store.py`; either fold the legacy fallbacks into the canonical modules and drop the `_new` suffix, or document explicitly why the fallback path exists.

### 1.3 🟡 Duplicate module name `table_extractor.py` in two packages
`pdf_compare/table_extractor.py` (165 lines, a thin workflow wrapper) and `pdf_compare/analyzers/table_extractor.py` (1347 lines, the actual engine) share the same filename. The top-level one imports from the analyzers one. Same-name modules in sibling packages are confusing to navigate and easy to mis-import.
- **Fix:** rename the wrapper (e.g. `table_workflow_api.py`) or merge into `table_workflows.py`.

### 1.4 🟡 Orphaned raster modules
`raster_diff.py` (`raster_diff_boxes`, `raster_diff_boxes_aligned`) has **no importers** — `cli.py` uses `raster_grid.raster_grid_changed_boxes` instead. `raster_diff_debug.py` is a standalone `__main__` script. Both appear to be superseded by `raster_grid.py`.
- **Fix:** confirm and remove `raster_diff.py`, or document it as a kept alternative implementation. Move `raster_diff_debug.py` to a `scripts/` or `tools/` dir if still useful.

### 1.5 🟡 `pdf_extract.py` and `pdf_extract_server.py` are near-duplicates
Both (410 / 382 lines) wrap `worker_pool` and expose parallel extraction. `__init__` exports `pdf_to_vectormap` from the former; CLI/UI/benchmark use `pdf_to_vectormap_server` from the latter. High likelihood of copy-paste drift. (Detailed in Section 2.)

### 1.6 🟢 Stray empty files committed
`TEMPFILE` (empty) and `overview.txt` (empty) are committed junk at `repo-root/`.
- **Fix:** delete both.

### 1.7 🟡 Generated output artifacts committed; `.gitignore` rule targets wrong path
Four `diff_overlay_*.pdf` files are committed under top-level `outputs/`. The `.gitignore` ignores `repo-root/outputs/` — but the artifacts live at `OCR/outputs/` (repo top level), so the rule never matches them.
- **Fix:** add `outputs/` (unanchored) to `.gitignore` and `git rm --cached` the four PDFs.

### 1.8 🟢 Sparse, mislocated tests
Only three test files for ~11k LOC (`test_rag.py`, `test_table_extractor.py`, `test_strtree_mapping.py`), all at `repo-root/` rather than a `tests/` package. Core modules (`store_new`, `compare_new`, `db_backend`, `pdf_extract*`) have no unit tests — already flagged in `docs/PROJECT_ANALYSIS_2025.md`.
- **Fix:** create `tests/`, add coverage for the core extract/compare/store path.

### 1.9 🟢 Two build-script trios (reviewed in Section 9)

---

## Section 2 — Core PDF extraction (`pdf_extract.py`, `pdf_extract_server.py`, `worker_pool.py`)

**Resolved this pass:** 2.1 (shared `_extract_core.py`, both modules now import the helpers), 2.2 (added `re`/`qu` stroke extraction + regression tests in `tests/test_extract_core.py`), 2.5 (removed false retry claim), 2.9 (fixed DPI doc), 2.10 (removed unused imports).
**Still open:** 2.4 (server alias), 2.6 (hung-worker timeout), 2.8 (`print`→`logging`). **Resolved later:** 2.3 (wired in §4), 2.7 (resolved in the GPU/OCR-robustness detour — see below).

---

## Section 3 — Comparison & raster (`compare_new.py`, `compare.py`, `raster_grid.py`, `page_alignment.py`, `overlay.py`)

**Resolved this pass:** 3.1/3.2/3.3 (deleted `page_alignment.py`, legacy `compare.py`, `PAGE_ALIGNMENT_GUIDE.md`; removed the unreachable sqlite fallback so `compare_new` now requires a `DatabaseBackend`), 3.4 (fixed broken `Union[..., any]` annotation), 3.7 (correct iterative box merge + regression test), 3.9 (removed unused `json`/`Tuple`).
**Still open:** 3.5 (text-diff O(n²)/exact-match — fold into OCR work), 3.6 (document the unused "changed" category), 3.8 (overlay raster-fallback page-subset behavior), 3.10 (`print`→`logging`).

---

## Section 4 — Data layer (`db_backend.py`, `db_models.py`, `models.py`, `store_new.py`, `search_new.py`, `search.py`)

**Resolved this pass:** 4.1 (added `source` to `TextRun` + `text_run_from_dict` helper; populated in both extract modules; `upsert_vectormap` now writes `t.source`; `get_vectormap` reads it back; regression tests in `tests/test_models.py` — also closes 2.3), 4.2 (stripped all SQLite branches from `db_backend`, now PostgreSQL-only; fixed `db_models` messaging), 4.3 (deleted `search.py`, rewrote `search_new` to require `DatabaseBackend`, fixed the `any` annotation), 4.4 (removed the orphaned typed-diff layer from `models.py`), 4.7 (removed redundant `import json`).
**Still open:** 4.5 (bbox stored as JSON Text — needs a schema migration, deferred), 4.6 (replica read-after-write consistency), 4.8 (`store_new` thin pass-through).

---

## Section 5 — Analyzers & OCR (`analyzers/*`, `table_extract_api.py`, `table_workflows.py`)

Context: there are **four independent OCR call sites** — `highres_ocr`/`tiled_ocr` (live), `table_extractor.extract_cell_text` (live), `enhanced_ocr` (dead), and `legend_extractor` (dead). This whole section is the most affected by the planned handwriting-OCR pivot.

**Resolved this pass:** 5.1 + 5.2 + 5.6 + 5.9 (deleted `enhanced_ocr.py` and `legend_extractor.py` + their `__init__` exports — this removed the pickle-load security hole, the numeric-noise heuristic, and the Canny-on-binary bug along with the dead code), 5.4 (removed the dead `max_workers` param from `tiled_ocr` + its only caller, and the unused `ThreadPoolExecutor` import), 5.5 (EasyOCR readers now cached per `(lang, gpu)`).
**Resolved in the GPU/OCR-robustness detour:** 2.7 (engine/GPU fallback) and 5.8 (`ensure_vectormap` default) — see the Detour section below.
**Resolved later:** 5.3 cell-OCR (post-merge) — `extract_cell_text` now routes through `resolve_ocr_engine` (EasyOCR+GPU when available, Tesseract fallback); `TableExtractionConfig` gained `ocr_engine`/`ocr_use_gpu`; engine resolved once per table. So BOM/table extraction can use the 5090. Routing covered by `tests/test_table_cell_ocr.py`.
**Still open:** 5.3-preprocessing (a fully unified render/preprocess abstraction across all OCR sites remains a larger refactor), 5.7 (Tesseract path), 5.10 (`print`→`logging`) — lower-priority polish.

---

## Section 6 — RAG / AI (`rag_simple.py`, `rag_symbol_recognition.py`)

Both are Ollama/LangChain RAG layers built for **text-bearing engineering diagrams** (symbol legends, valve tags) — squarely the project's main purpose. `rag_simple` is live (UI `get_rag_chat` + `test_rag.py`); `rag_symbol_recognition` has no live callers.

**Resolved this pass:** 6.2 (migrated live `rag_simple` to `langchain_ollama`: `OllamaLLM`/`OllamaEmbeddings`/`langchain_text_splitters`), 6.3 (added `_extract_json` so legend/compare features tolerate prose/fenced JSON), 6.7 (removed unused `PromptTemplate`/`Path`).
**Kept (per user — diagram-symbol RAG stays relevant):** 6.1 `rag_symbol_recognition.py` retained despite being unused/duplicate; revisit for consolidation. **Still open:** 6.4/6.5 (its CWD-Chroma + no remote-Ollama), 6.6 (wire RAG to stored OCR text).

---

## Detour — GPU / OCR robustness (in service of "functional high-res OCR on large diagrams")

Targeted fixes to make the live OCR path work reliably and use the available GPU (an RTX 5090 is on hand).

### D.1 🔴 (was 2.7) OCR engine/GPU no longer hardcoded — graceful fallback added
Added `resolve_ocr_engine(engine=None, use_gpu=None)` to `highres_ocr.py`: engine = explicit arg > `OCR_ENGINE` env > EasyOCR-if-installed else Tesseract; GPU = explicit arg > `OCR_USE_GPU` env > CUDA autodetect (`torch.cuda.is_available()`). If EasyOCR is requested but missing, it falls back to Tesseract instead of failing into a swallowed traceback. `pdf_extract._extract_text` and the CLI `ocr` command now both route through it, so on a CUDA box (the 5090) they auto-select **EasyOCR + GPU**, and on a CPU-only host they still produce text. Covered by `tests/test_ocr_engine.py` (6 tests).

### D.2 🔴 (NEW) `cli.py` called a non-existent `backend.get_session()` — CLI `ocr` was broken
`DatabaseBackend` exposed only `SessionLocal`/`read_session`; `cli.py` lines 211 & 241 called `backend.get_session()`, an `AttributeError` that crashed the CLI `ocr` (and the page-selection) command before any OCR ran. Added a `get_session()` accessor (new read/write session on the primary engine). The CLI OCR path now executes.

### D.3 🟡 (was 5.8) `ensure_vectormap` no longer force-triggers OCR
Default flipped `enable_ocr=True` → `False`: table extraction uses vector geometry + its own per-cell OCR, so regenerating a missing vectormap no longer silently launches the full OCR pass. Opt in explicitly when an OCR'd text layer is also wanted.

**Note for the 5090:** RTX 5090 (Blackwell, sm_120) needs a recent PyTorch with CUDA 12.8+/Blackwell support; if `torch.cuda.is_available()` is False after install, that's a torch-build issue, not this code. Force with `OCR_USE_GPU=1 OCR_ENGINE=easyocr`. Remaining GPU opportunity: `table_extractor.extract_cell_text` is still Tesseract/CPU-only (finding 5.3) — a candidate to move onto EasyOCR/GPU next.

---

## Section 7 — UI (`ui/streamlit_app.py`, `ui/streamlit_session_manager.py`)

**Resolved this pass:** 7.1 (threaded `ocr_engine`/`ocr_use_gpu` through `pdf_to_vectormap` → `_extract_page_job` → `_extract_text` → `resolve_ocr_engine`; the UI dropdown now actually selects EasyOCR/GPU vs Tesseract), 7.2 (searchable-PDF output now uses `outputs_dir`, not `/app/outputs`), 7.3 (Streamlit floor bumped to `>=1.49` in both manifests), 7.6 (fixed stale layout docstring), 7.7 (removed unused `session_cached` + `get_session_info` + `wraps` import).
**Still open:** 7.5 (DPI-to-2000 guard), 7.8 (bare import / heavy RAG object in session). `GlobalSessionStore` kept (still called by `init_session` cleanup) though its store is never populated.
**Resolved (run-experience pass):** 7.4 — `pdf_to_vectormap` now accepts a `progress_callback`; the UI passes `update_progress` on the OCR ingest path too, so the page-progress bar advances during large-diagram OCR instead of looking frozen. (OCR still runs serial under Streamlit by design, but now with live feedback.)

---

## Section 8 — CLI, entry points & tests (`cli.py`, `benchmark_modes.py`, `setup_check.py`, root `test_*.py`)

**Resolved this pass:** 8.1 (`compare --with-ocr` now runs a shared `_run_ocr_augment` on the new doc before diffing; removed the unimplemented `changed-cells`/`ocr_psm` flags), 8.2 (CLI `ingest` gained `--ocr`/`--ocr-engine`/`--ocr-dpi`, routed to the OCR-capable extractor), 8.3 (moved the three demo scripts + `benchmark_modes.py` to `tools/` and added `[tool.pytest.ini_options] testpaths=["tests"]` so bare `pytest` collects only the unit suite — verified), 8.4 (fixed `tools/test_rag.py` deprecated LangChain imports), 8.5 (`setup_check` now requires only Python+core incl. `easyocr`/`pandas`/`rapidfuzz`, RAG/Ollama optional, fixed script path), 8.6 (`_run_ocr_augment` clears prior OCR rows → idempotent; uses `json.dumps` for bbox), 8.7 (`benchmark_modes.py` moved to `tools/`).
**Note:** `_run_ocr_augment` is the single OCR-augment path now shared by `ocr-augment` and `compare --with-ocr`.

---

## Section 9 — Infra / Docker / deployment (Dockerfiles, compose, nginx, pgbouncer, prometheus, build scripts)

**Resolved this pass:** 9.1 (dropped `test_rag.py` from both Dockerfile `COPY`s), 9.2 (base Dockerfile now `pip install -r requirements.txt` — single source, drift eliminated), 9.3 (removed `sqlite3`/`libsqlite3-dev` from both; added `libgl1`/`libglib2.0-0` to `Dockerfile.with-ollama` so `cv2` imports), 9.4 (`docker-compose-scaled.yml` reworked for **NVIDIA**: `ollama/ollama` CUDA image, removed all ROCm config, added `driver: nvidia` GPU reservations to both the `ollama` and `pdf-compare-ui` services so the 5090 drives Ollama + EasyOCR). Compose YAML validated.
**Still open:** 9.5 (`Dockerfile.with-ollama` echo-built start script), 9.6 (scaled-stack port bindings + Grafana port), 9.7 (`host.docker.internal` on Linux), 9.8 (README compose-file name → §10).

---

## Section 10 — Docs (`README.md` ×2, ~25 files under `docs/`)

**Resolved this pass:** 10.1 (fixed both READMEs — removed SQLite backend/`sqlite://` example, `test_rag.py`→`tools/test_rag.py`, `docker-compose-postgres.yml`→`-full.yml`, `raster_grid_improved.py`→`raster_grid.py`, SQLite-FTS→Postgres-FTS, OCR description updated), 10.2 (deleted obsolete `DATABASE_COMPARISON.md` + removed its links from `docs/INDEX.md`).
**Still open (deeper docs, deliberately out of scope):** 10.3 (~70 stale module refs across `PROJECT_ANALYSIS_2025`/`IMPLEMENTATION_SUMMARY`/`COMPLETE_SETUP_GUIDE`/`QUICK_REFERENCE`/etc. — and 4 of these now have **dangling links** to the deleted `DATABASE_COMPARISON.md`: `PROJECT_ANALYSIS_2025.md`, `docs/README.md`, `IMPLEMENTATION_SUMMARY.md`, `COMPLETE_SETUP_GUIDE.md`), 10.4 (8+ overlapping Docker docs to consolidate), 10.5/10.6 (RAG framing, INDEX residuals). These need a dedicated docs pass.

The docs are **systemically stale** — written during the SQLite era and the `_new` migration and never updated. 180 stale references across 21 files. The audit (Sections 1–9) widened the gap (deleted modules, moved scripts, Postgres-only).

### 10.1 🔴 Top-level `README.md` gives instructions that don't work
Concrete errors a new user hits immediately:
- "PostgreSQL **or SQLite** backend" and a `DATABASE_URL=sqlite:///./data/comparisons.db` example — SQLite was removed; `open_db`/`db_backend` now **reject** non-postgres URLs.
- `python test_rag.py your_diagram.pdf` and a "`test_rag.py` — Test suite" layout entry — the file moved to `tools/test_rag.py` (and was never a real test suite).
- `docker-compose -f docker-compose-postgres.yml up` — that file doesn't exist (9.8); the real ones are `docker-compose-full.yml` / `-scaled.yml`.

### 10.2 🟡 `docs/reference/DATABASE_COMPARISON.md` is obsolete
The whole document argues *"Current Architecture (SQLite) … PostgreSQL benefits … why you should migrate."* The migration is done and SQLite is gone, so it documents a state that no longer exists. It's a historical planning artifact, not current reference. Delete or move to an `archive/`.

### 10.3 🟡 Stale module/layout references throughout `docs/`
`store.py` / `search.py` / `compare.py` / `page_alignment` / `raster_diff` / `enhanced_ocr` / `legend_extractor` appear in `PROJECT_ANALYSIS_2025.md` (19 refs), `IMPLEMENTATION_SUMMARY.md` (13), `COMPLETE_SETUP_GUIDE.md` (13), `QUICK_REFERENCE.md` (8), `RAG_SYMBOL_RECOGNITION_GUIDE.md` (5), etc. — all naming modules that were renamed or deleted in this audit. Code examples referencing them will fail.

### 10.4 🟡 Docker-doc sprawl / duplication
The deployment story is spread across **8+** overlapping files: `docs/DOCKER_DEPLOYMENT.md` **and** `docs/deployment/DOCKER_DEPLOYMENT.md` (same name, two locations), plus `DOCKER_SETUP.md`, `DOCKER_QUICKSTART.md`, `DOCKER_BUILD_VERIFICATION.md`, `DEPLOYMENT.md`, `SCALED_DEPLOYMENT.md`, `DOCKER_ARCHITECTURE.md`. They disagree in places and multiply the maintenance surface. Consolidate to one deployment guide + one quickstart.

### 10.5 🟢 `INSTALL_OLLAMA_WINDOWS.md` / RAG guides assume RAG is core
Given RAG/Ollama are optional (§8.5), the setup guides over-emphasize them as required first-run steps. Minor framing.

### 10.6 🟢 `docs/INDEX.md` likely links to deleted/renamed docs
The index references guides including the now-deleted `PAGE_ALIGNMENT_GUIDE.md` (removed in §3) and the soon-obsolete `DATABASE_COMPARISON.md`. Needs a pass once 10.2/10.4 are decided.

### 9.1 🔴 Both Dockerfiles `COPY test_rag.py` — which no longer exists → build fails
`Dockerfile` (line 92) and `Dockerfile.with-ollama` (line 37) do `COPY test_rag.py setup_check.py ./`. §8 moved `test_rag.py` to `tools/`, so `docker build` now errors (`COPY failed: test_rag.py not found`). A dev demo script shouldn't be in the production image anyway.
- **Fix:** drop `test_rag.py` from both COPYs (keep `setup_check.py`).

### 9.2 🟡 Three sources of dependency truth — and the Dockerfile has already drifted
Deps are declared in `requirements.txt`, `pyproject.toml`, **and** the base `Dockerfile`'s 5-stage inline `pip install` list. The inline list still pins `streamlit>=1.32` even though §7 bumped the floor to `>=1.49` in the other two — so the image can resolve a Streamlit that crashes on `width="stretch"` (7.3). `Dockerfile.with-ollama` does the right thing (`pip install -r requirements.txt`); the base one duplicates the list.
- **Fix:** switch the base Dockerfile to `pip install -r requirements.txt` (single source), or at minimum bump its inline `streamlit` pin.

### 9.3 🟡 SQLite system packages installed despite SQLite being removed
Both Dockerfiles `apt-get install ... sqlite3 libsqlite3-dev`. After §4 the app is PostgreSQL-only; these are dead image weight.
- **Fix:** drop `sqlite3` and `libsqlite3-dev`.

### 9.4 🟡 `docker-compose-scaled.yml` is hardwired for AMD ROCm — won't use your NVIDIA 5090
The scaled stack runs `ollama/ollama:rocm` with `/dev/kfd`, `/dev/dri`, `ROCM_VISIBLE_DEVICES=all` — AMD-only. On your NVIDIA box that GPU config is wrong, and the `pdf-compare-ui` EasyOCR GPU reservation is commented out. So in Docker, **neither** Ollama nor EasyOCR would use the 5090.
- **Fix (if you Dockerize):** use `ollama/ollama` (CUDA) image, NVIDIA `deploy.resources.reservations.devices` with `driver: nvidia`, and uncomment the app-container GPU reservation. (Local non-Docker already uses the GPU via the §-detour resolver.)

### 9.5 🟡 `Dockerfile.with-ollama` start script is built with `echo '...\n...'` → broken file
The startup script is written via `RUN echo '#!/bin/bash\n\ ...'`. Under `/bin/sh` (dash), `echo` doesn't interpret `\n`, so `/app/start.sh` ends up containing literal `\n` instead of newlines — a non-functional script. Should `COPY` a real `start.sh` (or use a heredoc).

### 9.6 🟡 Scaled compose exposes infra on all interfaces; Grafana port malformed
`docker-compose-scaled.yml` publishes `5432:5432`, `11434:11434`, `6379:6379`, `9090:9090` on `0.0.0.0` (Postgres/Ollama/Redis/Prometheus reachable from the network), whereas `docker-compose-full.yml` correctly binds `127.0.0.1:`. Grafana maps `":3000:3000"` (leading colon → malformed/host-random). Tighten bindings for the scaled stack.

### 9.7 🟡 `OLLAMA_HOST=http://host.docker.internal:11434` in base Dockerfile is not Linux-portable
`host.docker.internal` resolves on Docker Desktop (Win/Mac) but not on stock Linux without `--add-host`. The compose files override it (`http://ollama:11434`), so it only bites direct `docker run` on Linux. Minor.

### 9.8 🟢 README references a non-existent `docker-compose-postgres.yml`
README's Docker section says `docker-compose -f docker-compose-postgres.yml up`, but only `-full` and `-scaled` exist (build scripts use those correctly). Doc bug — tracked in §10.

### 8.1 🟡 `compare --with-ocr` is a no-op stub with a full set of dead flags
The `compare` command accepts `with_ocr`, `ocr_mode` (`sparse|all|changed-cells`), `ocr_dpi`, `ocr_min_conf`, `ocr_psm`, and four `changed_cells_*` params — but when `with_ocr` is set it just prints *"OCR augmentation … not yet implemented in CLI"* and runs the diff without OCR. Every OCR knob on `compare` is accepted and ignored; the docstring advertises modes that do nothing. Now that `ocr_augment` works (after the `get_session` fix), `compare` could call it — or the dead flags should be removed.

### 8.2 🟡 CLI `ingest` can't OCR
`ingest` calls `pdf_to_vectormap(pdf, doc_id=doc_id)` with `enable_ocr` defaulting False and exposes no OCR flags, so ingesting a scanned/large drawing via the CLI stores **no OCR text** — you must run `ocr-augment` as a separate step. For a "functional high-res OCR" CLI, `ingest` should offer `--ocr` / `--ocr-engine` / `--ocr-dpi`.

### 8.3 🔴 The three root `test_*.py` are demo scripts that break/pollute `pytest` collection
`test_rag.py`, `test_table_extractor.py`, `test_strtree_mapping.py` are manual `__main__` scripts (require a PDF arg, a running Ollama, or just print debug output), not unit tests. Under a bare `pytest` at repo root, pytest **collects** their `test_`-prefixed functions: `test_ollama_connection`/`test_embeddings` run and make live network calls, while arg-taking ones (`test_pdf_chat(pdf_path)`) raise collection errors (missing fixture). So `pytest` is effectively broken at the root — only `pytest tests/` works. They should move to `tools/`/`examples/` (or lose the `test_` prefix).

### 8.4 🟡 `test_rag.py` uses the deprecated LangChain imports
It imports `from langchain_community.llms import Ollama` / `OllamaEmbeddings` — the same deprecated path fixed in `rag_simple` (6.2). If kept as a smoke script, it should match.

### 8.5 🟡 `setup_check.py` treats Ollama/RAG as required and misses the OCR deps
`all_ok = all(results.values())` fails the whole check if Ollama or the RAG packages are absent — but the core purpose (OCR/diagrams) doesn't need them. A working OCR setup reports "components missing" + exit 1. It also checks `langchain`/`chromadb` but **not** `easyocr`, `pandas`, or `rapidfuzz` (the deps the OCR/table paths actually require). RAG/Ollama should be an optional section; the OCR deps should be in the required set.

### 8.6 🟢 `ocr_augment` re-runs accumulate duplicate rows
It inserts OCR `TextRow`s without first clearing existing OCR rows for those pages, so running it twice doubles the text. Also stores bbox as `str(list(...))` (parseable JSON, but a third bbox format alongside the f-string and `json.dumps` variants — see 4.5).

### 8.7 🟢 `benchmark_modes.py` is a fine standalone utility, just mislocated
Works, no DB, compares client vs server extraction. Belongs in `tools/` with the other scripts rather than the package root.

### 7.1 🔴 The "OCR Engine" selector is decorative — it doesn't control the engine
The ingest UI shows an `OCR Engine` dropdown (`EasyOCR (GPU)` / `Tesseract (CPU)`) and a green "✓ GPU acceleration enabled" badge, but `ocr_engine` is **never passed to extraction**. The call is `pdf_to_vectormap(target, workers=..., enable_ocr=True, ocr_dpi=...)` — no engine/GPU args — and `pdf_to_vectormap` has no such parameters, so `_extract_text` falls back to `resolve_ocr_engine()` (env/autodetect). The user's selection is ignored. Directly undercuts the GPU control you want.
- **Fix:** thread `engine`/`use_gpu` through `pdf_to_vectormap` → `_extract_text` → `resolve_ocr_engine(engine=, use_gpu=)`, and pass the dropdown value (and GPU implied by EasyOCR).

### 7.2 🔴 Searchable-PDF output path hardcoded to `/app/outputs/` — breaks local (Windows) runs
Line 510: `output_path = f"/app/outputs/{output_filename}"`. That's a **Docker-internal absolute path**. Running locally (your Windows box), `create_searchable_pdf` will try to `makedirs("/app/outputs")` (→ `C:\app\outputs`) and write there, instead of the configured `outputs_dir` (`APP_DATA_DIR/outputs`). The Create-Searchable-PDF feature fails or writes to a surprising location off-app.
- **Fix:** use `outputs_dir / output_filename`.

### 7.3 🟡 `width="stretch"` requires Streamlit ≥1.49 but the pin allows ≥1.32
7 `st.dataframe(..., width="stretch")` calls. `width="stretch"` only exists in Streamlit ≥1.49 (installed dev env is 1.50, so it works here), but `pyproject`/`requirements` pin `streamlit>=1.32,<1.51`. Anyone resolving 1.32–1.48 hits a `StreamlitAPIException`/`TypeError` on every dataframe.
- **Fix:** bump the floor to `streamlit>=1.49` (or use `use_container_width=True`).

### 7.4 🟡 OCR ingest shows no progress and runs serial
`enable_ocr=True` routes to `pdf_to_vectormap` (client), which takes **no** `progress_callback` (only the server variant does) and force-serializes under Streamlit. So OCR ingestion of a large diagram shows a frozen page-progress bar and runs single-process — looks hung on exactly the big jobs that matter. The defined `update_progress` closure is only wired to the non-OCR path.

### 7.5 🟡 OCR DPI options go to 2000 with no guard
The DPI dropdown offers up to 2000. For a large E-size sheet, 2000 DPI is hundreds of megapixels per page; combined with serial OCR under Streamlit (7.4) this can OOM or hang with no feedback. Consider capping or warning above ~600–800 for large pages.

### 7.6 🟢 Stale module-layout docstring
The header docstring (lines 8–15) still lists `store.py`, `search.py`, `compare.py` — all deleted in this audit — as the package layout. Misleading.

### 7.7 🟢 Dead code in `streamlit_session_manager.py`
`GlobalSessionStore`, `session_cached`, and `SessionManager.get_session_info` are defined but unused by the app (it uses `init_session` + `SessionManager` get/set/initialize_defaults). The `_global_sessions` store and its 100th-request cleanup never run meaningfully.

### 7.8 🟢 Fragile bare import + RAG object cached in session
`from streamlit_session_manager import ...` relies on `ui/` being `sys.path[0]` (true under `streamlit run ui/streamlit_app.py`, fragile otherwise). `get_rag_chat` stores a live `SimplePDFChat` (FAISS store + LLM client) in session state — heavy, and tied to the text-only RAG limitation (6.6).

### 6.1 🟡 `rag_symbol_recognition.py` (431 lines) is dead, and duplicates `rag_simple`
Nothing imports `RAGPDFAnalyzer`, `SymbolLegendExtractor`, or `SymbolMatcher` outside the module itself. It re-implements the same PDF→chunk→embed→RetrievalQA pipeline as `rag_simple`, but with a different stack (Chroma + HuggingFace embeddings vs FAISS + Ollama embeddings). Classic unused "v2". 

### 6.2 🟡 The **live** module uses deprecated LangChain imports; the **dead** one uses the modern ones
`rag_simple` (live) imports `from langchain_community.llms import Ollama`, `from langchain_community.embeddings import OllamaEmbeddings`, and `from langchain.text_splitter import ...` — all deprecated and slated for removal, even though the project already depends on `langchain-ollama` (the replacement). `rag_symbol_recognition` (dead) uses the current `langchain_ollama.OllamaLLM` / `langchain_text_splitters`. So the migration was done on the wrong file. `RetrievalQA.from_chain_type` (both) is also legacy LangChain.

### 6.3 🟡 LLM JSON outputs parsed with bare `json.loads` — effectively always fails
`extract_symbol_legend`, `is_same_symbol`, `extract_legend_from_page`, `match_symbol`, `compare_symbols` all call `json.loads(response)` directly on raw LLM text. Models like `llama3.2` routinely wrap JSON in prose or ``` fences, so these almost always hit the `except` path and return the empty/error fallback. The features look implemented but rarely produce structured output. Needs JSON-substring extraction (or structured-output / function-calling).

### 6.4 🟡 (dead module) Chroma persists to CWD `./chroma_db` and accumulates across documents
`RAGPDFAnalyzer` defaults `persist_directory="./chroma_db"`, `mkdir` in CWD, and `Chroma.from_documents(..., persist_directory=...)` with the same dir across different PDFs mixes vectors → cross-document contamination, plus CWD pollution. Moot if 6.1 deletes it.

### 6.5 🟡 (dead module) No remote-Ollama support
`rag_symbol_recognition`'s `OllamaLLM(...)` has no `base_url`/`OLLAMA_HOST` plumbing, so it only talks to localhost — unlike `rag_simple`, which honors `OLLAMA_HOST`/`OLLAMA_BASE_URL`. Would break in the Docker/remote setup. Moot if deleted.

### 6.6 🟢 Text-only RAG — depends on a PDF text layer
Both modules load text via `PyPDFLoader` (the PDF text layer only); image-only/scanned pages yield nothing. For the **current goal** (large engineering diagrams, which usually carry vector text or have OCR'd text in the DB) this is mostly fine, but the RAG layer does not consume the OCR text the pipeline already extracts — it re-reads the raw PDF instead. Worth wiring RAG to the stored OCR text later; not blocking now.

### 6.7 🟢 Unused imports in `rag_simple`
`PromptTemplate` (line 17) and `Path` (line 6) are imported but never used.

### 5.1 🟡 `enhanced_ocr.py` (548 lines) and `legend_extractor.py` (322 lines) are dead code
Both are only re-exported by `analyzers/__init__.py`; nothing in `cli`, `ui`, `pdf_extract`, or the RAG modules calls `enhanced_ocr`, `EnhancedOCRConfig`, `SymbolLibrary`, `LegendExtractor`, or `validate_ocr_against_legend`. (The RAG layer has its *own* `SymbolLegendExtractor`.) ~870 lines of sophisticated-but-unused code.
- **Fix (decision):** delete them, or wire them in. If deleted, findings 5.2/5.6/5.8 below disappear with them.

### 5.2 🔴 (in dead `enhanced_ocr`) Pickle deserialization of cache files
`OCRCache` does `pickle.load()` on `./ocr_cache/<hash>.pkl`. Loading pickle from disk is an arbitrary-code-execution risk if that directory is ever shared, synced, or attacker-writable, and `cache_dir=Path("./ocr_cache")` is a CWD-relative path that litters wherever the app runs. Use JSON (the cached results are plain dicts). **Moot if 5.1 deletes the module.**

### 5.3 🟡 Live OCR engine selection is hardcoded and duplicated
The live path renders + OCRs in two unrelated places with divergent, hardcoded choices: `pdf_extract` calls `tiled_ocr(engine="easyocr", use_gpu=True)` (no fallback — see 2.7), while `table_extractor.extract_cell_text` is **Tesseract-only** (no EasyOCR/GPU option at all) and applies `equalizeHist` per cell. Preprocessing differs everywhere (bilateral `(9,75,75)` vs `(5,50,50)` vs `equalizeHist`). Swapping in a handwriting VLM later means editing every site. A single rendering+engine abstraction is the right target for the OCR rebuild.

### 5.4 🟡 `tiled_ocr` advertises parallelism it doesn't have
The signature/docstring expose `max_workers` ("Number of parallel workers"), but the implementation is hardcoded serial ("parallel support can be added later") and never uses it. `ThreadPoolExecutor`/`as_completed` are imported (highres_ocr line 11) but unused. Dead param + dead import.

### 5.5 🟡 EasyOCR reader singleton ignores `lang`/`use_gpu` after first call
`_get_easyocr_reader` memoizes a module-global `_EASYOCR_READER` on first use; later calls with a different language or `use_gpu=False` silently get the original reader. Wrong results if language/GPU mode ever varies within a process.

### 5.6 🟡 `SymbolLibrary.is_noise` flags mostly-numeric text as noise
`sum(c.isdigit())/len(text) > 0.8` marks numeric strings as "suspicious noise", and `len(text) < 2` drops single characters. For **handwritten maintenance logs** (quantities, dates, meter readings) this would discard exactly the data you want. Domain-inappropriate. (In dead `enhanced_ocr`; relevant if revived.)

### 5.7 🟡 Hardcoded Windows Tesseract path, only in one module
`highres_ocr` sets `pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"` at import; `enhanced_ocr`, `table_extractor`, and `legend_extractor` don't, so they rely on PATH. Inconsistent, and Windows-specific config baked into a module that also runs in Linux containers.

### 5.8 🟡 `ensure_vectormap` defaults `enable_ocr=True`
Table extraction (`extract_tables` → `ensure_vectormap`) regenerates missing vectormaps with `enable_ocr=True` by default, silently triggering the full EasyOCR+GPU pipeline (5.3 / 2.7). On a no-GPU host this is a surprising, slow, or failing side effect of "extract tables."

### 5.9 🟢 `detect_callouts` runs Canny on an already-binarized image
`enhanced_ocr` passes the adaptive-thresholded (binary) image into `detect_callouts`, which then runs `cv2.Canny` on it — edge detection on a binary mask is degenerate and yields poor regions. Latent bug (dead module).

### 5.10 🟢 `print(file=sys.stderr)` OCR logging
`highres_ocr` emits many `print(..., file=sys.stderr)` progress lines (same issue as 2.8). Should use `logging`.

### 4.1 🔴 Text `source` provenance is broken end-to-end — OCR text is permanently mislabeled "native"
The schema supports it (`db_models.TextRow.source`, default `"native"`, with its own index `idx_text_rows_source`), and the read paths honor it (`get_document_text_with_coords`, `export_document_text` return/print `source`). The extractor even computes it (`pdf_extract._extract_text` tags `"native"`/`"ocr"`). But the chain is severed in two places:
1. `models.TextRun` has **no `source` field**, so the tag is dropped at dict→dataclass conversion (this is the deferred 2.3).
2. `db_backend.upsert_vectormap` **hardcodes `source="native"`** for every row (line 230).

Net effect: OCR'd text is always stored as "native", the indexed `source` column is meaningless, and searchable-PDF export can never distinguish OCR from native text. An entire schema feature is inert.
- **Fix:** add `source` to `TextRun`, populate it in `pdf_extract`/`pdf_extract_server`, and write `t.source` (defaulting to `"native"`) in `upsert_vectormap`.

### 4.2 🟡 `db_backend` still fully supports SQLite, contradicting "SQLite support removed"
The class accepts `sqlite://` URLs and carries SQLite-specific code (StaticPool, WAL pragmas, `LIKE` search fallback, `is_sqlite` branches), and its docstring advertises *'SQLite: "sqlite:///..." (for local dev/testing)'*. Yet `requirements.txt`, `pdf_compare/__init__.py`, and `store_new.open_db` all declare PostgreSQL-only and **reject** non-postgres URLs. The messaging is contradictory.
- **Note:** the SQLite path is actually useful for unit tests (no Postgres needed). Decision: either officially support it (fix the docs/`open_db` that deny it) or strip the branches.

### 4.3 🟡 `search.py` is the last sqlite3 legacy module; `search_new` repeats the dead-fallback + broken annotation
`search_new.search_text` annotates `conn_or_backend: Union[DatabaseBackend, any]` (lowercase `any` builtin again — same bug as 3.4) and falls back to `search.py`, which is raw `sqlite3` against the removed schema. `search.search_geometry_bbox` is unused entirely. This is the exact pattern deleted in §3 (compare), left standing for search.
- **Fix:** delete `search.py`, drop the fallback, fix the annotation to `DatabaseBackend`.

### 4.4 🟡 Orphaned typed-diff layer in `models.py` (~80 lines)
`PageDiff`, `GeometryDiff`, `TextAdd`, `TextMoved`, `TextDiff`, `RasterGridConfig` and the helpers `page_diff_from_dict` / `page_diff_to_overlay_dict` / `diffs_to_overlay_dicts` are defined and exported but referenced nowhere — their only consumer (`compare.diff_documents_typed`) was deleted in §3. Dead code.
- **Fix:** delete them, or actually adopt typed diffs in `compare_new`. (The ingest models `VectorMap`/`TextRun`/etc. are live and stay.)

### 4.5 🟡 `bbox` stored as hand-built JSON text, parsed with a fragile fallback
`TextRow.bbox` is `Column(Text)` (comment claims "JSONB for PostgreSQL", but it's plain Text). `upsert_vectormap` builds it with an f-string `f"[{x0},{y0},{x1},{y1}]"` rather than `json.dumps`, and **three** read sites parse it as `json.loads(...)` with a `bbox_str.strip("[]").split(",")` fallback — the fallback itself signals known fragility. NaN/inf coords or any format drift breaks parsing. `GeometryRow` already stores `x0..y1` as `Float` columns; text bbox could do the same (or real JSONB) for typing + indexing.

### 4.6 🟡 Replica read-after-write inconsistency
Writes always hit the primary, but `load_page_geoms` / `load_page_texts` / `search_text` read from a **random** replica (`read_session`), while `get_vectormap` and `list_documents` deliberately use the primary "to guarantee read-after-write." So a just-ingested document can be diffed (which uses `load_page_*`) against stale replica data. Harmless single-node (the pool falls back to primary), but a latent correctness bug in the scaled deployment the repo advertises.

### 4.7 🟢 Redundant local `import json`
`export_document_text` re-imports `json` (line 425) though it's already imported at module top (line 8).

### 4.8 🟢 `store_new.py` is a thin pass-through wrapper
Most functions just delegate 1:1 to `DatabaseBackend` methods. It adds an indirection layer without behavior; several wrappers (`delete_all_documents`, `export_document_text`) have no callers. Minor redundancy — consider importing the backend directly.

### 3.1 🔴 The page-alignment feature is SQLite-only and dead on the supported PostgreSQL backend
`page_alignment.py` and `compare.diff_documents_aligned` are built entirely on raw `sqlite3` SQL (`SELECT ... FROM text_rows`, `geometry.x0..y1`) against the **removed** SQLite schema. The canonical `compare_new.py` (used by CLI + UI) has **no** aligned-diff path and no alignment hook. Grep confirms `align_pages` / `diff_documents_aligned` are referenced *only inside those two legacy files* — nothing on the live path calls them. So "intelligent page alignment" (handling inserted/deleted/reordered pages) — a documented feature with its own `docs/PAGE_ALIGNMENT_GUIDE.md` — **cannot run** in the shipped PostgreSQL-only configuration. It's either dead code or it throws if invoked.
- **Fix (decision needed):** port the alignment logic onto `DatabaseBackend`, or delete it + its guide as dead code.

### 3.2 🟡 `compare_new` legacy fallback can never succeed
`diff_documents`/`diff_pages` fall back to `compare.legacy_*` when the arg isn't a `DatabaseBackend` — but that fallback uses `sqlite3`, and SQLite support was removed, so the only supported input *is* `DatabaseBackend`. The else-branch is unreachable-by-design dead code that imports the legacy module.

### 3.3 🟡 The `_new` migration silently dropped features
`compare_new.py` lost `diff_documents_typed` (typed `PageDiff` models) and `diff_documents_aligned` that exist in `compare.py`. Anyone relying on typed diffs or aligned diffs via the canonical module no longer has them. (Ties to 1.2 — incomplete migration.)

### 3.4 🟡 Broken type annotation: `Union[DatabaseBackend, any]`
`compare_new.py` (lines 74, 108) annotates `conn_or_backend: Union[DatabaseBackend, any]`. Lowercase `any` is the **builtin function**, not `typing.Any` — `Union[X, any]` is meaningless/incorrect. Should be `DatabaseBackend` (if the fallback is dropped) or `Union[DatabaseBackend, Any]`.

### 3.5 🟡 Text diff is O(n_a × n_b) and matches on exact string equality only
`diff_pages` nests a full loop over `b_txt` for every `a_txt` entry, matching only when `t2 == t` exactly, then picking nearest center. For drawings/logs with hundreds of text runs this is slow, and any minor OCR variation ("Valve" vs "Va1ve") turns one logical edit into a remove+add pair. Fragile for the OCR direction; consider spatial indexing + fuzzy match (you already depend on `rapidfuzz`).

### 3.6 🟢 Geometry "changed" is never populated
Both modules always return `geometry.changed = []` (comment: "reserved for raster modes"). The overlay legend advertises a "Changed" category that the vector path never fills — only raster grid does. Acceptable by design, but worth documenting so it doesn't read as a bug.

### 3.7 🟡 `_merge_adjacent_boxes` is an incomplete interval merge
It sorts boxes then merges each only against `merged[-1]`. Boxes that overlap a merged region earlier in the list (not just the immediate predecessor) are left unmerged, so the output can still contain overlapping rectangles. A proper sweep or union-find merge is needed for correctness.

### 3.8 🟡 Overlay rasterization fallback changes document structure
The final fallback in `write_overlay` builds a fresh doc with **one page per diff entry**, so pages without diffs vanish and page numbering shifts versus the primary path (which annotates all original pages). Output shape silently depends on which code path succeeded.

### 3.9 🟢 Unused imports
`compare_new.py`: `json` (line 7) and `Tuple` (line 6) are unused. Minor lint.

### 3.10 🟢 `print(file=sys.stderr)` in `overlay.create_searchable_pdf`
Same logging inconsistency as 2.8 — uses `print` to stderr instead of `logging`.

### 2.1 🟡 `pdf_extract.py` and `pdf_extract_server.py` are ~90% duplicated
`_hash_file`, `_cubic_sample`, `_adaptive_bezier_samples`, and the entire `_drawings_to_geoms` (≈80 lines) are **byte-identical** across both files; the two public functions differ only in OCR support, logging style, and env-var config. The duplication has **already drifted** (see 2.2), which is exactly the failure mode duplication invites.
- **Fix:** extract the shared helpers (`_hash_file`, bezier sampling, `_drawings_to_geoms`) into a `_extract_core.py` and have both modules import them. Or collapse to one module with an `ocr=`/`config=` switch.

### 2.2 🔴 Stroke extraction silently drops rectangle and quad path ops
`_drawings_to_geoms` handles only `op == "l"` (line) and `op == "c"` (cubic). PyMuPDF `get_drawings()` also emits `"re"` (rectangle) and `"qu"` (quad) draw items. Any *stroked* rectangle or quad (borders, boxes, title-block frames, table cell outlines) is **never extracted** — they're skipped entirely. For engineering diagrams and table-heavy maintenance logs this is a meaningful loss of geometry.
- **Fix:** add `"re"` and `"qu"` branches that emit the rectangle/quad as a `LineString`/closed ring stroke (and respect `min_segment_len`).

### 2.3 🟡 Extracted `source` ("native"/"ocr") tag is computed then discarded
In `pdf_extract.py`, `_extract_text` tags every run with `"source": "native"` or `"ocr"`, but the dict→dataclass conversion (`TextRun(text=, bbox=, font=, size=)`) never passes `source` through. The provenance info — which would let downstream distinguish OCR'd text from native text — is silently dropped. Either wire it into `TextRun` or stop computing it.

### 2.4 🟡 Two different `pdf_to_vectormap` symbols with divergent behavior
`pdf_extract.py` defines the real `pdf_to_vectormap` (supports OCR). `pdf_extract_server.py` ends with `pdf_to_vectormap = pdf_to_vectormap_server` (a no-OCR alias). A caller importing `pdf_to_vectormap` from the server module gets silently different behavior (no OCR, no Streamlit handling). Misleading "backwards compatibility" alias.
- **Fix:** drop the alias, or make the server function honor the same OCR contract.

### 2.5 🟡 `worker_pool` advertises auto-retry that doesn't exist
The module docstring claims *"Automatic retry with reduced workers on crashes."* No retry logic exists — `submit_throttled` raises `RuntimeError` on the first worker exception and aborts. Documentation describes a feature that isn't implemented.
- **Fix:** implement the retry (catch `BrokenProcessPool`, halve workers, re-run) or remove the claim.

### 2.6 🟡 `submit_throttled` can loop forever on a hung worker
The `wait(..., timeout=300)` branch only logs a warning and `continue`s. If a worker genuinely hangs (e.g. Tesseract/EasyOCR stuck on a huge page), there's no max-timeout or abort — the loop spins every 300s indefinitely. Add a bounded number of empty waits before raising.

### 2.7 🟡 OCR path hardcodes EasyOCR + GPU with no fallback
In `_extract_text`, both OCR branches hardcode `engine="easyocr", use_gpu=True`. On a box without CUDA (or without `easyocr` installed) this fails or silently underperforms, with the error only swallowed into a stderr traceback. Magic constants (`TESSERACT_PIXEL_LIMIT=29000`, `psm=11`, `min_conf=50`, `overlap_pct=0.35`) are buried inline. Given the new handwriting-OCR direction, this path needs to be configurable (engine, GPU, DPI, confidence) rather than hardcoded.

### 2.8 🟢 Inconsistent logging: `print(file=sys.stderr)` vs `logging`
`pdf_extract.py` uses `print(..., file=sys.stderr)` for ~10 OCR debug lines (with `import sys` repeated inside functions), while `pdf_extract_server.py` and `worker_pool.py` use the `logging` module. The print spam can't be silenced by log level and clutters stderr. Standardize on `logging`.

### 2.9 🟢 Contradictory OCR-DPI documentation
`pdf_to_vectormap`'s docstring says `ocr_dpi` "will be auto-capped"; the inline comment at the call site says "no capping here - tiled OCR handles it." One is wrong.

### 2.10 🟢 Unused imports
`pdf_extract.py`: `asdict` (line 4), `Polygon` and `unary_union` (lines 21–22), `BBox` (line 26) are imported but never used. `pdf_extract_server.py`: `BBox` (line 47) unused. Minor lint.
`build.bat` / `build.ps1` / `build.sh` plus `docker-build.ps1` / `docker-build.sh` / `docker-build-test.ps1` / `docker_test.sh`. Likely overlapping responsibilities. Reviewed in Section 9.
