"""PDF Revision Diff - Local Streamlit UI (all local)

Usage
-----
1) Ensure the `vectormap` package from our previous step is on PYTHONPATH (or installed as an editable package).
   Repo layout example:

   pdf_compare/
     __init__.py
     models.py
     pdf_extract.py
     store.py
     search.py
     compare.py
     overlay.py
   ui/
     streamlit_app.py  (this file)

2) Create and activate a virtual environment, then install requirements:

   pip install -r requirements.txt

   # If you don't have a requirements.txt yet, use the inline list below:
   pip install streamlit pymupdf shapely rtree numpy pillow typer

3) Run the UI:

   streamlit run streamlit_app.py

4) Interact in the browser:
   - Drag & drop PDFs into the uploader and click **Ingest**.
   - Use **Search** to find text across ingested PDFs.
   - Pick **Old** and **New** documents, click **Compare**, then **Create overlay** to export a diff PDF.

Notes
-----
- Database location is configured via `DATABASE_URL` (PostgreSQL).
- Overlays are generated locally; no external calls.
- Large PDFs: ingestion time scales with page count; progress bars included.
- Geometry registration (alignment) is not enabled here; add a module later if needed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from streamlit_session_manager import SessionManager, init_session

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    HAS_PSUTIL = False

# Suppress Streamlit ScriptRunContext warnings from multiprocessing workers
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

try:
    from pdf_compare.pdf_extract import pdf_to_vectormap
    from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
    from pdf_compare.db_backend import DatabaseBackend
    from pdf_compare.store_new import (
        delete_all_documents,
        delete_document,
        export_document_text,
        get_document_text_with_coords,
        get_vectormap,
        list_documents,
        upsert_vectormap,
    )
    from pdf_compare.overlay import create_searchable_pdf
    from pdf_compare.search_new import search_text as vm_search_text
    from pdf_compare.compare_new import diff_documents
    from pdf_compare.overlay import write_overlay
    from pdf_compare.rag_simple import SimplePDFChat
    from pdf_compare.analyzers.table_extractor import TableExtractor, TableExtractionConfig
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    st.error(
        "Couldn't import the 'pdf_compare' package. Ensure the repository root is on PYTHONPATH or install it as an editable package (pip install -e .).\n"
        f"Underlying import error: {exc}"
    )
    st.stop()

st.set_page_config(page_title="PDF Revision Diff (Local)", layout="wide")

init_session()
SessionManager.initialize_defaults({
    "rag_chat_cache": {},
    "rag_histories": {},
    "extracted_tables": [],
    "last_diffs": [],
    "last_pair": None,
})

# Sidebar configuration
st.sidebar.header("Configuration")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error(
        "DATABASE_URL not set.\n\n"
        "This application requires PostgreSQL. Set the DATABASE_URL environment variable, for example:\n\n"
        "    DATABASE_URL=postgresql://user:password@host:5432/pdfcompare"
    )
    st.stop()

st.sidebar.success("PostgreSQL connected")
if "@" in DATABASE_URL:
    display_url = DATABASE_URL.split("@", maxsplit=1)[1]
    st.sidebar.caption(f"Host: `{display_url}`")

READ_REPLICA_URLS: List[str] = []
single_replica = os.getenv("DATABASE_READ_URL")
if single_replica:
    READ_REPLICA_URLS.append(single_replica)

idx = 1
while True:
    env_key = f"DATABASE_READ_URL_{idx}"
    replica_url = os.getenv(env_key)
    if not replica_url:
        break
    READ_REPLICA_URLS.append(replica_url)
    idx += 1

default_data_root = Path.cwd() / "data"
data_root = Path(os.getenv("APP_DATA_DIR", default_data_root))
data_root.mkdir(parents=True, exist_ok=True)
outputs_dir = data_root / "outputs"
uploads_dir = data_root / "uploads"
outputs_dir.mkdir(parents=True, exist_ok=True)
uploads_dir.mkdir(parents=True, exist_ok=True)

st.sidebar.subheader("Performance")
cpu_count = os.cpu_count() or 4
default_workers = max(1, cpu_count - 1)
num_workers = st.sidebar.slider(
    "Worker Processes",
    min_value=1,
    max_value=cpu_count,
    value=default_workers,
    help=f"Number of parallel workers for PDF extraction. System cores: {cpu_count}. Recommended: {default_workers}"
)
st.sidebar.caption(f"Using {num_workers} worker(s) for parallel extraction")

if HAS_PSUTIL:
    st.sidebar.caption(f"CPU load: {psutil.cpu_percent(interval=0.1)}%")


@st.cache_resource(show_spinner=False)
def get_db_backend() -> DatabaseBackend:
    return DatabaseBackend(DATABASE_URL, read_replica_urls=READ_REPLICA_URLS)


def get_conn() -> DatabaseBackend:
    return get_db_backend()


def display_pdf(pdf_path: Path, *, height: int = 720) -> None:
    if not pdf_path.exists():
        st.error(f"PDF not found: {pdf_path}")
        return

    try:
        pdf_bytes = pdf_path.read_bytes()
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to read PDF {pdf_path}: {exc}")
        return

    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe = (
        f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
        f'width="100%" height="{height}" type="application/pdf"></iframe>'
    )
    components.html(iframe, height=height + 16, scrolling=False)


def get_rag_chat(doc_id: str, doc_path: Path, *, force_refresh: bool = False) -> "SimplePDFChat":
    cache = SessionManager.get_user_state("rag_chat_cache")
    if not isinstance(cache, dict):
        cache = {}
        SessionManager.set_user_state("rag_chat_cache", cache)

    if not force_refresh:
        entry = cache.get(doc_id)
        if entry and entry.get("path") == str(doc_path):
            stored = entry.get("chat")
            if isinstance(stored, SimplePDFChat):
                return stored

    chat = SimplePDFChat(str(doc_path))
    cache[doc_id] = {"chat": chat, "path": str(doc_path)}
    SessionManager.set_user_state("rag_chat_cache", cache)
    return chat


def reset_rag_chat(doc_id: str) -> None:
    cache = SessionManager.get_user_state("rag_chat_cache")
    if isinstance(cache, dict) and doc_id in cache:
        cache.pop(doc_id, None)
        SessionManager.set_user_state("rag_chat_cache", cache)


st.title("PDF Revision Diff - Local UI")
st.caption("Drop PDFs, index & search content, and generate visual diff overlays. All local.")

backend = get_conn()
workspace_tab, viewer_tab, chat_tab = st.tabs(["Workspace", "PDF Viewer", "Ollama Chat"])

with workspace_tab:
    st.subheader("1) Upload & Ingest PDFs")
    with st.expander("Upload PDFs", expanded=True):
        uploaded = st.file_uploader(
            "Drop one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are stored locally under ./uploads and indexed into PostgreSQL."
        )

        col_debug, col_ocr = st.columns(2)
        with col_debug:
            debug_mode = st.checkbox("Enable debug logging", value=False, help="Show detailed extraction progress")
        with col_ocr:
            enable_ocr = st.checkbox("Enable OCR", value=True, help="Run OCR when embedded text is sparse")

        ingest_btn = st.button("Ingest selected PDFs", type="primary", disabled=not uploaded)

        if ingest_btn and uploaded:
            overall_progress = st.progress(0, text="Overall Progress")
            page_progress_bar = st.progress(0, text="Page Progress")
            status_container = st.container()

            debug_messages: List[str] = []
            debug_feed = st.expander("Debug Feed", expanded=True) if debug_mode else None

            total_files = len(uploaded)

            for file_idx, uploaded_file in enumerate(uploaded, start=1):
                with status_container:
                    st.info(f"Processing file {file_idx}/{total_files}: {uploaded_file.name}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                target_name = f"{timestamp}_{Path(uploaded_file.name).stem}.pdf"
                target_path = uploads_dir / target_name
                start_time = time.time()

                if debug_mode:
                    debug_messages.append(f"[{datetime.now():%H:%M:%S}] Saving to {target_path}")

                with open(target_path, "wb") as handle:
                    handle.write(uploaded_file.getbuffer())

                if debug_mode:
                    debug_messages.append(f"[{datetime.now():%H:%M:%S}] Saved {uploaded_file.name}")

                try:
                    doc = fitz.open(str(target_path))
                    page_count = doc.page_count
                    doc.close()

                    if debug_mode:
                        debug_messages.append(f"[{datetime.now():%H:%M:%S}] Page count: {page_count}")
                        actual_workers = min(num_workers, page_count)
                        debug_messages.append(
                            f"[{datetime.now():%H:%M:%S}] Using {actual_workers} worker process(es)"
                        )

                    def update_progress(completed_pages: int, total_pages: int) -> None:
                        progress_pct = completed_pages / total_pages
                        page_progress_bar.progress(progress_pct, text=f"Extracting page {completed_pages}/{total_pages}")
                        if debug_mode and debug_feed is not None:
                            debug_messages.append(
                                f"[{datetime.now():%H:%M:%S}] Extracted page {completed_pages}/{total_pages}"
                            )
                            debug_feed.text_area(
                                "Debug Log",
                                value="\n".join(debug_messages[-50:]),
                                height=200,
                                key=f"debug_{file_idx}_{completed_pages}"
                            )

                    # Unified extraction with fallback logic (applies to both OCR and non-OCR paths)
                    # Try with requested workers, fallback to fewer workers if process pool crashes
                    workers_to_try = [num_workers, max(1, num_workers // 2), 1]
                    vectormap = None

                    for attempt, worker_count in enumerate(workers_to_try, 1):
                        try:
                            if debug_mode and attempt > 1:
                                debug_messages.append(f"[{datetime.now():%H:%M:%S}] Retry {attempt} with {worker_count} worker(s)")

                            # Choose extraction function based on OCR setting
                            if enable_ocr:
                                vectormap = pdf_to_vectormap(
                                    str(target_path),
                                    workers=worker_count,
                                    enable_ocr=True,
                                )
                            else:
                                vectormap = pdf_to_vectormap_server(
                                    str(target_path),
                                    workers=worker_count,
                                    progress_callback=update_progress,
                                )

                            if debug_mode:
                                ocr_status = "OCR-enabled" if enable_ocr else "Standard"
                                debug_messages.append(f"[{datetime.now():%H:%M:%S}] {ocr_status} extraction complete with {worker_count} worker(s)")
                            page_progress_bar.progress(1.0, text="Extraction complete")
                            break  # Success!

                        except Exception as worker_exc:
                            if debug_mode:
                                debug_messages.append(f"[{datetime.now():%H:%M:%S}] Extraction failed with {worker_count} workers: {worker_exc}")

                            # If this was the last attempt, re-raise
                            if attempt == len(workers_to_try):
                                raise worker_exc

                            # Otherwise, warn and try again with fewer workers
                            with status_container:
                                st.warning(f"Process pool crashed with {worker_count} workers. Retrying with {workers_to_try[attempt]}...")

                    if debug_mode:
                        debug_messages.append(f"[{datetime.now():%H:%M:%S}] Writing to database")

                    if vectormap is None:
                        raise RuntimeError("Failed to extract vectormap from PDF")

                    upsert_vectormap(backend, vectormap)

                    elapsed = time.time() - start_time
                    with status_container:
                        st.success(
                            f"Ingested {uploaded_file.name} as {vectormap.meta.doc_id} "
                            f"({vectormap.meta.page_count} pages in {elapsed:.1f}s)"
                        )

                except Exception as exc:
                    with status_container:
                        st.error(f"Failed to ingest {uploaded_file.name}: {exc}")
                    if debug_mode and debug_feed is not None:
                        debug_messages.append(f"[{datetime.now():%H:%M:%S}] ERROR: {exc}")
                        debug_feed.text_area(
                            "Debug Log",
                            value="\n".join(debug_messages[-50:]),
                            height=200,
                            key=f"debug_{file_idx}_error"
                        )

                overall_progress.progress(file_idx / total_files, text=f"Overall Progress: {file_idx}/{total_files} files")

            if debug_mode and debug_feed is not None:
                debug_feed.text_area(
                    "Final Debug Log",
                    value="\n".join(debug_messages),
                    height=300,
                    key="debug_final"
                )

            with status_container:
                st.success(f"All done! Processed {total_files} file(s)")

    docs: List[Tuple[str, str, int]] = list_documents(backend)

    st.subheader("2) Indexed Documents")
    if not docs:
        st.info("No documents ingested yet.")
    else:
        st.dataframe(
            {"doc_id": [d[0] for d in docs], "path": [d[1] for d in docs], "pages": [d[2] for d in docs]},
            use_container_width=True,
        )

        with st.expander("Document Management", expanded=False):
            st.caption("Warning: these operations cannot be undone.")
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Delete Individual Document**")
                delete_options = [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs]
                selected_doc = st.selectbox(
                    "Select document to delete",
                    options=delete_options,
                    key="delete_doc_select",
                )
                if st.button("Delete Selected Document", type="secondary", key="delete_single"):
                    doc_id = selected_doc.split(" - ", maxsplit=1)[0]
                    if delete_document(backend, doc_id):
                        st.success(f"Deleted document: {doc_id}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete document: {doc_id}")

            with col_right:
                st.markdown("**Clear All Documents**")
                st.caption(f"This will delete all {len(docs)} document(s)")
                confirm_clear = st.checkbox("I understand this will delete all documents", key="confirm_clear")
                if st.button("Delete All Documents", type="secondary", disabled=not confirm_clear, key="delete_all"):
                    deleted = delete_all_documents(backend)
                    st.success(f"Deleted {deleted} document(s)")
                    st.rerun()

        with st.expander("🔍 Debug: Export Extracted Text", expanded=False):
            st.caption("Download extracted text to verify OCR results")

            col_export_doc, col_export_fmt = st.columns(2)

            with col_export_doc:
                export_options = [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs]
                selected_export = st.selectbox(
                    "Select document",
                    options=export_options,
                    key="export_doc_select",
                )
            with col_export_fmt:
                export_format = st.radio("Format", ["txt", "json"], horizontal=True, key="export_format")

            if st.button("Generate Export", type="primary", key="export_btn"):
                export_doc_id = selected_export.split(" - ", maxsplit=1)[0]
                filename = Path(selected_export.split(" - ")[1]).stem

                export_content = export_document_text(backend, export_doc_id, format=export_format)

                # Show preview
                st.text_area("Preview (first 1000 chars)", export_content[:1000], height=200)

                # Download button
                st.download_button(
                    label=f"Download {export_format.upper()}",
                    data=export_content,
                    file_name=f"{filename}_extracted_text.{export_format}",
                    mime="application/json" if export_format == "json" else "text/plain",
                    key="download_export"
                )

        with st.expander("📄 Create Searchable PDF", expanded=False):
            st.caption("Embed invisible OCR text at precise coordinates to make scanned PDFs searchable")

            searchable_doc = st.selectbox(
                "Select document to make searchable",
                options=[f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs],
                key="searchable_doc_select"
            )

            if st.button("Generate Searchable PDF", type="primary", key="generate_searchable"):
                searchable_doc_id = searchable_doc.split(" - ", maxsplit=1)[0]

                # Get document info
                doc_info = next((d for d in docs if d[0] == searchable_doc_id), None)
                if not doc_info:
                    st.error("Document not found")
                else:
                    doc_id, source_path, page_count = doc_info

                    with st.spinner("Creating searchable PDF..."):
                        # Get all text with coordinates
                        text_data = get_document_text_with_coords(backend, searchable_doc_id)

                        if not text_data:
                            st.warning("No text found in document. OCR may not have been run.")
                        else:
                            # Generate output path
                            output_filename = Path(source_path).stem + "_searchable.pdf"
                            output_path = f"/app/outputs/{output_filename}"

                            # Create searchable PDF
                            create_searchable_pdf(source_path, text_data, output_path)

                            # Offer download
                            with open(output_path, "rb") as f:
                                pdf_bytes = f.read()

                            st.success(f"✅ Created searchable PDF with {len(text_data)} text overlays")
                            st.download_button(
                                label="Download Searchable PDF",
                                data=pdf_bytes,
                                file_name=output_filename,
                                mime="application/pdf",
                                key="download_searchable"
                            )

    st.subheader("3) Search Text (FTS5)")
    with st.expander("Text search", expanded=True):
        col_query, col_doc, col_page = st.columns([3, 2, 1])
        with col_query:
            search_query = st.text_input("Query (FTS syntax supported)", placeholder="e.g., valve OR pump*")
        with col_doc:
            doc_select_options = ["(any)"] + [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs]
            chosen_doc = st.selectbox("Limit to document", doc_select_options)
        with col_page:
            page_number = st.number_input("Page (optional)", min_value=0, value=0, help="0 = any page")

        if st.button("Run search", disabled=not search_query):
            doc_id_filter = None if chosen_doc == "(any)" else chosen_doc.split(" - ", maxsplit=1)[0]
            page_filter = None if page_number == 0 else int(page_number)
            try:
                rows = vm_search_text(backend, search_query, doc_id=doc_id_filter, page=page_filter, limit=500)
                if rows:
                    st.success(f"{len(rows)} hit(s)")
                    st.dataframe(
                        {
                            "doc_id": [r[0] for r in rows],
                            "page": [r[1] for r in rows],
                            "text": [r[2] for r in rows],
                            "bbox": [r[3] for r in rows],
                            "font": [r[4] for r in rows],
                            "size": [r[5] for r in rows],
                        },
                        use_container_width=True,
                    )
                else:
                    st.warning("No results.")
            except Exception as exc:
                st.error(f"Search error: {exc}")

    st.subheader("4) Quick BOM Extract")
    with st.expander("📋 Extract Parts List / BOM to CSV", expanded=False):
        st.caption("Quickly extract Bill of Materials from your drawing and download as CSV")

        if not docs:
            st.info("No documents available. Please ingest documents first.")
        else:
            bom_doc = st.selectbox(
                "Select drawing",
                options=[f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs],
                key="bom_doc_select"
            )

            col_bom_settings = st.columns(2)
            with col_bom_settings[0]:
                bom_dpi = st.select_slider("Quality", options=[200, 300, 400, 500], value=400, key="bom_dpi")
            with col_bom_settings[1]:
                bom_pages = st.text_input("Pages (e.g., 1 or 1,3,5)", value="1", key="bom_pages")

            if st.button("Extract BOM to CSV", type="primary", key="quick_bom_btn"):
                bom_doc_id = bom_doc.split(" - ", maxsplit=1)[0]
                doc_info = next((d for d in docs if d[0] == bom_doc_id), None)

                if not doc_info:
                    st.error("Document not found")
                else:
                    _, doc_path, page_count = doc_info

                    # Parse page numbers
                    try:
                        if bom_pages.strip():
                            page_indices = [int(p.strip()) - 1 for p in bom_pages.split(",")]
                        else:
                            page_indices = [0]
                    except ValueError:
                        st.error("Invalid page numbers. Use format: 1 or 1,2,3")
                        page_indices = None

                    if page_indices is not None:
                        invalid_pages = [idx + 1 for idx in page_indices if idx < 0 or idx >= page_count]
                        if invalid_pages:
                            st.error(
                                "Page number(s) out of range: "
                                + ", ".join(str(p) for p in invalid_pages)
                            )
                        else:
                            with st.spinner("Extracting BOM from drawing..."):
                                try:
                                    config = TableExtractionConfig(
                                        dpi=bom_dpi,
                                        ocr_min_conf=40,
                                        enable_line_detection=True,
                                        enable_whitespace_detection=True,
                                        bom_keywords=[
                                            "PARTS LIST", "BILL OF MATERIALS", "BOM", "MATERIAL LIST",
                                            "ITEM", "QTY", "QUANTITY", "PART NUMBER", "DESCRIPTION", "PART NO",
                                            "PART#", "P/N", "PN", "MATL", "MATERIAL"
                                        ]
                                    )

                                    page_numbers = sorted({idx + 1 for idx in page_indices})

                                    vectormap = get_vectormap(backend, bom_doc_id, page_numbers=page_numbers)

                                    if vectormap is None or not vectormap.pages:
                                        vectormap = pdf_to_vectormap(
                                            doc_path,
                                            doc_id=bom_doc_id,
                                            workers=1,
                                            enable_ocr=True,
                                        )
                                        try:
                                            upsert_vectormap(backend, vectormap)
                                        except Exception as upsert_exc:
                                            st.warning(
                                                f"Re-ingested vectormap locally but failed to persist: {upsert_exc}"
                                            )

                                    extractor = TableExtractor(config)
                                    tables = extractor.extract_all_tables(
                                        doc_path,
                                        page_indices=page_indices,
                                        vector_map=vectormap,
                                    )

                                    if not tables:
                                        st.warning("No tables found on the selected page(s).")
                                    else:
                                        bom_tables = [t for t in tables if t.table_type == "bom"]
                                        if not bom_tables:
                                            st.warning(
                                                f"Found {len(tables)} table(s) but none identified as BOM. Showing all tables."
                                            )
                                            bom_tables = tables

                                        st.success(f"Found {len(bom_tables)} BOM table(s)")

                                        try:
                                            import pandas as pd
                                            import io

                                            all_bom_rows = []
                                            for table in bom_tables:
                                                df = table.to_dataframe()
                                                if not df.empty:
                                                    df.insert(0, "Page", table.page)
                                                    all_bom_rows.append(df)

                                            if all_bom_rows:
                                                combined_bom = pd.concat(all_bom_rows, ignore_index=True)

                                                st.dataframe(combined_bom, use_container_width=True, height=300)

                                                csv_buffer = io.StringIO()
                                                combined_bom.to_csv(csv_buffer, index=False)
                                                csv_data = csv_buffer.getvalue()

                                                filename = Path(doc_path).stem
                                                output_csv = outputs_dir / f"{filename}_BOM.csv"
                                                output_csv.write_text(csv_data, encoding="utf-8")

                                                st.download_button(
                                                    label="?? Download BOM CSV",
                                                    data=csv_data,
                                                    file_name=f"{filename}_BOM.csv",
                                                    mime="text/csv",
                                                    type="primary",
                                                    key="download_bom_csv"
                                                )

                                                st.success(f"? BOM extracted: {len(combined_bom)} rows")
                                            else:
                                                st.error("No data extracted from tables")

                                        except ImportError:
                                            st.error("pandas is required for CSV export. Please rebuild Docker container.")
                                        except Exception as e:
                                            st.error(f"CSV generation failed: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())

                                except Exception as exc:
                                    st.error(f"BOM extraction failed: {exc}")
                                    import traceback
                                    st.code(traceback.format_exc())
    st.subheader("5) Advanced Table Extraction")
    with st.expander("Table extraction from drawings", expanded=False):
        st.caption("Extract Bills of Materials, parts lists, symbol legends, and other tabular data from engineering drawings")

        if not docs:
            st.info("No documents available. Please ingest documents first.")
        else:
            col_table_doc, col_table_page = st.columns([3, 1])

            with col_table_doc:
                table_doc_options = [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs]
                selected_table_doc = st.selectbox(
                    "Select document for table extraction",
                    options=table_doc_options,
                    key="table_doc_select"
                )

            with col_table_page:
                extract_all_pages = st.checkbox("All pages", value=True, key="extract_all_pages")
                if not extract_all_pages:
                    page_for_table = st.number_input("Page number", min_value=1, value=1, key="table_page_num")

            # Configuration options
            with st.expander("Advanced settings", expanded=False):
                col_dpi, col_thresh = st.columns(2)
                with col_dpi:
                    table_dpi = st.slider("Rendering DPI", 200, 600, 400, 50, help="Higher DPI improves OCR accuracy but takes longer")
                with col_thresh:
                    ocr_conf = st.slider("Minimum OCR confidence", 0, 100, 50, 5, help="Filter out low-confidence OCR results")

            if st.button("Extract Tables", type="primary", key="extract_tables_btn"):
                table_doc_id = selected_table_doc.split(" - ", maxsplit=1)[0]
                doc_info = next((d for d in docs if d[0] == table_doc_id), None)

                if not doc_info:
                    st.error("Document not found")
                else:
                    _, doc_path, page_count = doc_info

                    with st.spinner("Extracting tables from PDF..."):
                        # Create config
                        config = TableExtractionConfig(
                            dpi=table_dpi,
                            ocr_min_conf=ocr_conf
                        )

                        # Create extractor
                        extractor = TableExtractor(config)

                        # Extract tables
                        page_indices = None if extract_all_pages else [page_for_table - 1]

                        try:
                            page_numbers = None if page_indices is None else [idx + 1 for idx in page_indices]

                            vectormap = get_vectormap(backend, table_doc_id, page_numbers=page_numbers)

                            if vectormap is None or not vectormap.pages:
                                vectormap = pdf_to_vectormap(
                                    doc_path,
                                    doc_id=table_doc_id,
                                    workers=1,
                                    enable_ocr=True,
                                )
                                try:
                                    upsert_vectormap(backend, vectormap)
                                except Exception as upsert_exc:
                                    st.warning(
                                        f"Re-ingested vectormap locally but failed to persist: {upsert_exc}"
                                    )

                            tables = extractor.extract_all_tables(
                                doc_path,
                                page_indices=page_indices,
                                vector_map=vectormap,
                            )

                            if not tables:
                                st.warning("No tables detected in the selected pages.")
                            else:
                                st.success(f"Extracted {len(tables)} table(s)")

                                # Store in per-session state
                                SessionManager.set_user_state("extracted_tables", tables)
                                SessionManager.set_user_state("table_doc_path", str(doc_path))
                                SessionManager.set_user_state("table_doc_id", table_doc_id)

                                # Display summary
                                st.dataframe({
                                    "Table ID": [t.table_id for t in tables],
                                    "Type": [t.table_type for t in tables],
                                    "Page": [t.page for t in tables],
                                    "Rows": [len(t.rows) for t in tables],
                                    "Columns": [len(t.headers) for t in tables],
                                    "Detection": [t.metadata.get("detection_method", "unknown") for t in tables]
                                }, use_container_width=True)

                        except Exception as exc:
                            st.error(f"Table extraction failed: {exc}")
                            import traceback
                            st.code(traceback.format_exc())

            # Display extracted tables
            tables = SessionManager.get_user_state("extracted_tables")
            if tables:
                st.markdown("---")
                st.markdown("**Extracted Tables**")

                table_doc_id = SessionManager.get_user_state("table_doc_id") or "document"
                safe_doc_id = str(table_doc_id)

                # Filter by table type
                col_filter, col_export = st.columns([2, 1])
                with col_filter:
                    table_type_filter = st.selectbox(
                        "Filter by type",
                        options=["All", "BOM", "Symbols", "Line Types", "Generic"],
                        key="table_type_filter"
                    )

                with col_export:
                    export_format = st.radio("Export format", ["JSON", "CSV"], horizontal=True, key="table_export_format")

                # Filter tables
                filtered_tables = tables
                if table_type_filter != "All":
                    filter_map = {
                        "BOM": "bom",
                        "Symbols": "symbols",
                        "Line Types": "line_types",
                        "Generic": "generic"
                    }
                    filtered_tables = [t for t in tables if t.table_type == filter_map[table_type_filter]]

                if not filtered_tables:
                    st.info(f"No tables of type '{table_type_filter}' found.")
                else:
                    # Export buttons
                    col_export_json, col_export_csv = st.columns(2)

                    with col_export_json:
                        if export_format == "JSON":
                            json_data = json.dumps([t.to_dict() for t in filtered_tables], indent=2)
                            st.download_button(
                                "Download JSON",
                                data=json_data,
                                file_name=f"tables_{safe_doc_id}.json",
                                mime="application/json",
                                key="download_tables_json"
                            )

                    with col_export_csv:
                        if export_format == "CSV":
                            try:
                                import pandas as pd
                                # Create a combined CSV with all tables
                                all_rows = []
                                for table in filtered_tables:
                                    df = table.to_dataframe()
                                    df.insert(0, "Table ID", table.table_id)
                                    df.insert(1, "Table Type", table.table_type)
                                    df.insert(2, "Page", table.page)
                                    all_rows.append(df)

                                if all_rows:
                                    combined_df = pd.concat(all_rows, ignore_index=True)
                                    csv_data = combined_df.to_csv(index=False)
                                    st.download_button(
                                        "Download CSV",
                                        data=csv_data,
                                        file_name=f"tables_{safe_doc_id}.csv",
                                        mime="text/csv",
                                        key="download_tables_csv"
                                    )
                            except ImportError:
                                st.warning("pandas not installed. CSV export unavailable.")

                    # Display each table
                    for table in filtered_tables:
                        with st.expander(f"{table.table_id} - {table.table_type.upper()} (Page {table.page})", expanded=True):
                            st.caption(f"Detection method: {table.metadata.get('detection_method', 'unknown')}")

                            # Display headers
                            if table.headers:
                                st.markdown("**Headers:**")
                                st.write(", ".join(table.headers))

                            # Display table content as dataframe
                            if table.rows:
                                try:
                                    df = table.to_dataframe()
                                    st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 38))
                                except ImportError:
                                    # Fallback: display as dict
                                    st.json(table.to_dict())
                            else:
                                st.info("No rows in table")

                            # Show raw cell data
                            with st.expander("View raw cell data", expanded=False):
                                for row in table.rows[:5]:  # Show first 5 rows
                                    st.write(f"**Row {row.row_index}:**")
                                    for cell in row.cells:
                                        st.text(f"  Col {cell.col}: '{cell.text}' (conf: {cell.confidence:.1f}%)")

st.subheader("6) Compare & Create Overlay")
if len(docs) >= 2:
    col_old, col_new = st.columns(2)
    with col_old:
        old_choice = st.selectbox("Old document (baseline)", [f"{d[0]} - {Path(d[1]).name}" for d in docs])
    with col_new:
        new_choice = st.selectbox(
            "New document (revised)",
            [f"{d[0]} - {Path(d[1]).name}" for d in docs],
            index=1 if len(docs) > 1 else 0,
        )

    if st.button("Compare documents", type="secondary"):
        old_id = old_choice.split(" - ", maxsplit=1)[0]
        new_id = new_choice.split(" - ", maxsplit=1)[0]
        try:
            diffs = diff_documents(backend, old_id, new_id)
            st.success(f"Computed diffs for {len(diffs)} page(s)")
            st.dataframe(
                {
                    "page": [d["page"] for d in diffs],
                    "added geom": [len(d["geometry"]["added"]) for d in diffs],
                    "removed geom": [len(d["geometry"]["removed"]) for d in diffs],
                    "added text": [len(d["text"]["added"]) for d in diffs],
                    "removed text": [len(d["text"]["removed"]) for d in diffs],
                    "moved text": [len(d["text"]["moved"]) for d in diffs],
                },
                use_container_width=True,
            )
            SessionManager.set_user_state("last_diffs", diffs)
            SessionManager.set_user_state("last_pair", (old_id, new_id))
        except Exception as exc:
            st.error(f"Compare failed: {exc}")

    last_diffs = SessionManager.get_user_state("last_diffs")
    last_pair = SessionManager.get_user_state("last_pair")

    if last_diffs and last_pair:
        old_id, new_id = last_pair
        doc_map = {doc_id: (path, pages) for doc_id, path, pages in docs}
        base_pdf_path = doc_map.get(new_id, (None,))[0]
        overlay_name = f"diff_overlay_{old_id[:6]}_{new_id[:6]}.pdf"
        overlay_path = outputs_dir / overlay_name

        if st.button("Create overlay PDF", type="primary"):
            try:
                if base_pdf_path is None:
                    raise RuntimeError("Unable to resolve revised document path")
                write_overlay(base_pdf_path, last_diffs, str(overlay_path))
                st.success(f"Overlay written to {overlay_path}")
                with open(overlay_path, "rb") as handle:
                    st.download_button(
                        "Download overlay PDF",
                        data=handle,
                        file_name=overlay_name,
                        mime="application/pdf",
                    )
            except Exception as exc:
                st.error(f"Overlay generation failed: {exc}")
else:
    st.info("Ingest at least two documents to enable comparison.")

with viewer_tab:
    st.subheader("PDF Viewer")
    docs_for_viewer = list_documents(backend)
    overlay_files = sorted([p for p in outputs_dir.glob("*.pdf") if p.is_file()], key=lambda p: p.name.lower())

    if not docs_for_viewer and not overlay_files:
        st.info("No PDFs available yet. Ingest documents or generate overlays to preview them here.")
    else:
        source_options = []
        if docs_for_viewer:
            source_options.append("Ingested Documents")
        if overlay_files:
            source_options.append("Generated Overlays")

        source_choice: Optional[str] = source_options[0] if source_options else None
        if len(source_options) > 1:
            source_choice = st.radio("Source", source_options, horizontal=True)
        elif source_choice:
            st.caption(f"Source: {source_choice}")

        selected_path: Optional[Path] = None

        if source_choice == "Ingested Documents":
            doc_labels = [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs_for_viewer]
            selected_label = st.selectbox("Select document", doc_labels)
            if selected_label:
                selected_id = selected_label.split(" - ", maxsplit=1)[0]
                doc_map = {doc_id: (path, pages) for doc_id, path, pages in docs_for_viewer}
                if selected_id in doc_map:
                    doc_path, pages = doc_map[selected_id]
                    selected_path = Path(doc_path)
                    st.markdown(f"**Pages:** {pages}")
        elif source_choice == "Generated Overlays":
            overlay_labels = [p.name for p in overlay_files]
            selected_label = st.selectbox("Select overlay", overlay_labels)
            if selected_label:
                selected_path = next((p for p in overlay_files if p.name == selected_label), None)

        if selected_path:
            if selected_path.exists():
                with st.expander("File details", expanded=False):
                    st.write(f"Path: `{selected_path}`")
                    st.write(f"Size: {selected_path.stat().st_size / 1024:.1f} KiB")

                display_pdf(selected_path)

                with open(selected_path, "rb") as handle:
                    st.download_button(
                        "Download PDF",
                        data=handle,
                        file_name=selected_path.name,
                        mime="application/pdf",
                    )
            else:
                st.error(f"Selected file not found on disk: {selected_path}")

with chat_tab:
    st.subheader("Chat with Ollama")
    docs_for_chat = list_documents(backend)
    ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    st.caption(f"Ollama endpoint: `{ollama_host}`")

    if not docs_for_chat:
        st.info("Ingest a document first to enable chat.")
    else:
        doc_labels = [f"{doc_id} - {Path(path).name}" for doc_id, path, _ in docs_for_chat]
        selected_label = st.selectbox("Select document to chat about", doc_labels, key="rag_doc_select")
        selected_id = selected_label.split(" - ", maxsplit=1)[0] if selected_label else None

        doc_map = {doc_id: (path, pages) for doc_id, path, pages in docs_for_chat}
        chat_histories = SessionManager.get_user_state("rag_histories")
        if not isinstance(chat_histories, dict):
            chat_histories = {}
            SessionManager.set_user_state("rag_histories", chat_histories)

        if selected_id and selected_id in doc_map:
            doc_path_str, page_count = doc_map[selected_id]
            doc_path = Path(doc_path_str)

            if not doc_path.exists():
                st.error(f"Document file not found on disk: {doc_path}")
            else:
                st.caption(f"Document has {page_count} page(s). First response will build embeddings using Ollama.")

                doc_history = chat_histories.setdefault(selected_id, [])
                SessionManager.set_user_state("rag_histories", chat_histories)
                col_init, col_reset = st.columns(2)

                if col_init.button("Build / Refresh embeddings", key=f"rag_build_{selected_id}"):
                    try:
                        with st.spinner("Building embeddings via Ollama..."):
                            get_rag_chat(selected_id, doc_path, force_refresh=True)
                        st.success("Embeddings ready. Ask your question below.")
                    except Exception as exc:
                        reset_rag_chat(selected_id)
                        st.error(f"Failed to prepare embeddings: {exc}")

                if col_reset.button("Reset chat", key=f"rag_reset_{selected_id}"):
                    chat_histories[selected_id] = []
                    SessionManager.set_user_state("rag_histories", chat_histories)
                    reset_rag_chat(selected_id)
                    st.rerun()

                for role, message in doc_history:
                    st.chat_message(role).write(message)

                prompt = st.chat_input(
                    "Ask a question about this document",
                    key=f"rag_chat_input_{selected_id}"
                )

                if prompt:
                    doc_history.append(("user", prompt))
                    try:
                        with st.spinner("Consulting Ollama..."):
                            chat = get_rag_chat(selected_id, doc_path)
                            answer = chat.ask(prompt)
                    except Exception as exc:
                        answer = f"Error while contacting Ollama: {exc}"
                        reset_rag_chat(selected_id)
                    doc_history.append(("assistant", answer))
                    SessionManager.set_user_state("rag_histories", chat_histories)
                    st.rerun()
        else:
            st.info("Select a document to begin chatting.")

# --- Footer ---
st.caption("Local-only toolchain. Vector extraction via PyMuPDF; geometry diff via shapely; text search via PostgreSQL FTS.")
