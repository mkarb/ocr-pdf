"""
PDF Revision Diff – Local Streamlit UI (all local)

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
     streamlit_app.py  ← (this file)

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
- The SQLite database file path is configurable in the sidebar (default: vectormap.sqlite).
- Overlays are generated locally; no external calls.
- Large PDFs: ingestion time scales with page count; progress bars included.
- Geometry registration (alignment) is not enabled here; add a module later if needed.
"""

import io
import os
import logging
import time
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Optional CPU monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import streamlit as st

# Suppress Streamlit ScriptRunContext warnings from multiprocessing workers
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Try to import the vectormap modules. Provide a clear error if missing.
try:
    from pdf_compare.pdf_extract import pdf_to_vectormap
    from pdf_compare.pdf_extract_server import pdf_to_vectormap_server
    from pdf_compare.store import open_db, upsert_vectormap, list_documents
    from pdf_compare.search import search_text as vm_search_text
    from pdf_compare.compare import diff_documents
    from pdf_compare.overlay import write_overlay
    import fitz  # PyMuPDF for getting page count
except Exception as e:
    st.error(
        "Couldn't import the 'pdf_compare' package. Ensure the modules from our previous step are available.\n"
        "Add the repository root to PYTHONPATH or install it as an editable package (pip install -e .).\n"
        f"Underlying import error: {e}"
    )
    st.stop()

st.set_page_config(page_title="PDF Revision Diff (Local)", layout="wide")

# Sidebar configuration
st.sidebar.header("Configuration")

def_path = Path.cwd() / "vectormap.sqlite"
db_path = st.sidebar.text_input("SQLite DB path", value=str(def_path))
outputs_dir = Path.cwd() / "outputs"
uploads_dir = Path.cwd() / "uploads"
outputs_dir.mkdir(exist_ok=True)
uploads_dir.mkdir(exist_ok=True)

# Performance settings
st.sidebar.subheader("Performance")
cpu_count = os.cpu_count() or 4
default_workers = max(1, cpu_count - 1)
num_workers = st.sidebar.slider(
    "Worker Processes",
    min_value=1,
    max_value=cpu_count,
    value=default_workers,
    help=f"Number of parallel workers for PDF extraction. Your system has {cpu_count} cores. Recommended: {default_workers}"
)
st.sidebar.caption(f"💡 Using {num_workers} worker(s) for parallel extraction")

# Open (and initialize) DB connection lazily per action

def get_conn():
    return open_db(db_path)

st.title("PDF Revision Diff – Local UI")
st.caption("Drop PDFs, index & search content, and generate visual diff overlays. All local.")

# --- Section: Upload & Ingest ---
st.subheader("1) Upload & Ingest PDFs")
with st.expander("Upload PDFs", expanded=True):
    uploaded = st.file_uploader(
        "Drop one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are stored locally under ./uploads and indexed into SQLite."
    )

    # Add debug mode toggle
    debug_mode = st.checkbox("Enable debug logging", value=False, help="Show detailed extraction progress")

    ingest_btn = st.button("Ingest selected PDFs", type="primary", disabled=not uploaded)

    if ingest_btn and uploaded:
        conn = get_conn()

        # Create containers for progress visualization
        overall_progress = st.progress(0, text="Overall Progress")
        file_progress_container = st.container()
        page_progress_bar = st.progress(0, text="Page Progress")
        status_container = st.container()

        # Debug feed container
        if debug_mode:
            debug_feed = st.expander("Debug Feed", expanded=True)
            debug_messages = []

        total_files = len(uploaded)

        for file_idx, uf in enumerate(uploaded, start=1):
            with status_container:
                st.info(f"📄 Processing file {file_idx}/{total_files}: **{uf.name}**")

            # Persist the uploaded bytes to disk
            target = uploads_dir / uf.name
            start_time = time.time()

            if debug_mode:
                debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Saving {uf.name} to disk...")

            with open(target, "wb") as f:
                f.write(uf.getbuffer())

            if debug_mode:
                debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] File saved to {target}")

            try:
                # Get page count first for better progress tracking
                doc = fitz.open(str(target))
                page_count = doc.page_count
                doc.close()

                if debug_mode:
                    debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] PDF has {page_count} pages")
                    actual_workers = min(num_workers, page_count)  # Can't use more workers than pages
                    debug_messages.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Using {actual_workers} worker process(es) (configured: {num_workers}, available cores: {cpu_count})"
                    )
                    debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting vector extraction...")

                # Progress callback for page-level tracking
                def update_progress(completed_pages, total_pages):
                    progress_pct = completed_pages / total_pages
                    page_progress_bar.progress(
                        progress_pct,
                        text=f"Extracting page {completed_pages}/{total_pages}"
                    )
                    if debug_mode:
                        debug_messages.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Extracted page {completed_pages}/{total_pages} "
                            f"({progress_pct*100:.1f}%)"
                        )
                        with debug_feed:
                            st.text_area(
                                "Debug Log",
                                value="\n".join(debug_messages[-50:]),  # Show last 50 messages
                                height=200,
                                key=f"debug_{file_idx}_{completed_pages}"
                            )

                # Use server mode with progress callback and configured workers
                vm = pdf_to_vectormap_server(
                    str(target),
                    workers=num_workers,
                    progress_callback=update_progress
                )

                if debug_mode:
                    debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Vector extraction complete")
                    debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Storing in database...")

                upsert_vectormap(conn, vm)

                elapsed = time.time() - start_time

                if debug_mode:
                    debug_messages.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Database storage complete (total time: {elapsed:.2f}s)"
                    )

                with status_container:
                    st.success(
                        f"✅ Ingested **{uf.name}** as `{vm.meta.doc_id}` "
                        f"({vm.meta.page_count} pages in {elapsed:.1f}s, "
                        f"{vm.meta.page_count/elapsed:.1f} pages/sec)"
                    )

            except Exception as ex:
                if debug_mode:
                    debug_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(ex)}")
                    with debug_feed:
                        st.text_area(
                            "Debug Log",
                            value="\n".join(debug_messages[-50:]),
                            height=200,
                            key=f"debug_{file_idx}_error"
                        )
                with status_container:
                    st.error(f"Failed to ingest {uf.name}: {ex}")

            # Update overall progress
            overall_progress.progress(
                file_idx / total_files,
                text=f"Overall Progress: {file_idx}/{total_files} files"
            )

        # Final debug log
        if debug_mode:
            with debug_feed:
                st.text_area(
                    "Final Debug Log",
                    value="\n".join(debug_messages),
                    height=300,
                    key="debug_final"
                )

        with status_container:
            st.success(f"🎉 All done! Processed {total_files} file(s)")

# --- Section: Inventory ---
st.subheader("2) Indexed Documents")
conn = get_conn()
docs = list_documents(conn)
if not docs:
    st.info("No documents ingested yet.")
else:
    # docs: List of tuples (doc_id, path, page_count)
    st.dataframe(
        {"doc_id": [d[0] for d in docs], "path": [d[1] for d in docs], "pages": [d[2] for d in docs]},
        use_container_width=True,
    )

# --- Section: Search ---
st.subheader("3) Search Text (FTS5)")
with st.expander("Text search", expanded=True):
    colq, cold, colp = st.columns([3, 2, 1])
    with colq:
        query = st.text_input("Query (FTS5 syntax supported)", placeholder="e.g., valve OR pump*")
    with cold:
        doc_options = ["(any)"] + [f"{d[0]}  —  {Path(d[1]).name}" for d in docs]
        chosen_doc = st.selectbox("Limit to document", doc_options)
    with colp:
        page_no = st.number_input("Page (optional)", min_value=0, value=0, help="0 = any page")

    run_search = st.button("Run search", disabled=not query)

    if run_search and query:
        doc_id_filter = None if chosen_doc == "(any)" else chosen_doc.split()[0]
        page_filter = None if page_no == 0 else int(page_no)
        try:
            rows = vm_search_text(conn, query, doc_id=doc_id_filter, page=page_filter, limit=500)
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
        except Exception as ex:
            st.error(f"Search error: {ex}")

# --- Section: Compare & Overlay ---
st.subheader("4) Compare & Create Overlay")
if len(docs) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        old_choice = st.selectbox("Old document (baseline)", [f"{d[0]}  —  {Path(d[1]).name}" for d in docs])
    with col2:
        new_choice = st.selectbox("New document (revised)", [f"{d[0]}  —  {Path(d[1]).name}" for d in docs], index=1 if len(docs) > 1 else 0)

    do_compare = st.button("Compare documents", type="secondary")

    if do_compare:
        old_id = old_choice.split()[0]
        new_id = new_choice.split()[0]
        try:
            diffs = diff_documents(conn, old_id, new_id)
            st.success(f"Computed diffs for {len(diffs)} page(s)")
            # quick summary table by page
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
            st.session_state["_last_diffs"] = diffs
            st.session_state["_last_pair"] = (old_id, new_id)
        except Exception as ex:
            st.error(f"Compare failed: {ex}")

    if "_last_diffs" in st.session_state and "_last_pair" in st.session_state:
        old_id, new_id = st.session_state["_last_pair"]
        # Need the base PDF path for overlay export; grab from the documents table
        doc_map = {d[0]: d for d in docs}
        base_pdf_path = doc_map[new_id][1]
        overlay_name = f"diff_overlay_{old_id[:6]}_{new_id[:6]}.pdf"
        overlay_path = outputs_dir / overlay_name

        if st.button("Create overlay PDF", type="primary"):
            try:
                write_overlay(base_pdf_path, st.session_state["_last_diffs"], str(overlay_path))
                st.success(f"Overlay written → {overlay_path}")
                # Offer download
                with open(overlay_path, "rb") as f:
                    st.download_button(
                        "Download overlay PDF",
                        data=f,
                        file_name=overlay_name,
                        mime="application/pdf",
                    )
            except Exception as ex:
                st.error(f"Overlay generation failed: {ex}")
else:
    st.info("Ingest at least two documents to enable comparison.")

# --- Footer ---
st.caption("© Local-only toolchain. Vector extraction via PyMuPDF; geometry diff via shapely; text search via SQLite FTS5.")
