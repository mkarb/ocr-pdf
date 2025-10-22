"""
OCR Confidence Visualization Page

Shows extracted text with confidence scores, allows filtering by confidence,
and displays visual debug output.
"""

import streamlit as st
from pathlib import Path
import json

st.set_page_config(page_title="OCR Confidence", page_icon="search", layout="wide")

try:
    from pdf_compare.db_backend import DatabaseBackend
    import os
except Exception as exc:
    st.error(f"Failed to import required modules: {exc}")
    st.stop()

st.title("OCR Confidence Visualization")

# Get database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("DATABASE_URL not set. Please configure the database connection.")
    st.stop()

db = DatabaseBackend(DATABASE_URL)

# Sidebar controls
st.sidebar.header("Filters")

# Document selector
docs = db.list_documents()
if not docs:
    st.info("No documents ingested yet. Upload and process a PDF first.")
    st.stop()

doc_options = {f"{doc_id}: {Path(path).name}": doc_id for doc_id, path, _ in docs}
selected_doc_name = st.sidebar.selectbox("Select Document", list(doc_options.keys()))
selected_doc_id = doc_options[selected_doc_name]

# Get document details
vm = db.get_vectormap(selected_doc_id)
if not vm:
    st.error(f"Failed to load document: {selected_doc_id}")
    st.stop()

# Page selector
page_numbers = list(range(1, vm.meta.page_count + 1))
selected_page = st.sidebar.selectbox("Select Page", page_numbers)

# Confidence filter
st.sidebar.subheader("Confidence Filter")
min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0,
    max_value=100,
    value=0,
    help="Show only text with confidence >= this value"
)

show_native = st.sidebar.checkbox("Show Native Text", value=True,
                                   help="Include text extracted directly from PDF (no OCR)")
show_ocr = st.sidebar.checkbox("Show OCR Text", value=True,
                                 help="Include text from OCR")

# Get page data
selected_page_data = next((p for p in vm.pages if p.page_number == selected_page), None)
if not selected_page_data:
    st.error(f"Page {selected_page} not found")
    st.stop()

# Filter text runs
filtered_texts = []
for text_run in selected_page_data.texts:
    # Filter by source
    is_native = text_run.confidence is None
    if is_native and not show_native:
        continue
    if not is_native and not show_ocr:
        continue

    # Filter by confidence
    if text_run.confidence is not None and text_run.confidence < min_confidence:
        continue

    filtered_texts.append(text_run)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Page {selected_page} - Text Confidence")

    if not filtered_texts:
        st.warning("No text matches the current filters.")
    else:
        # Statistics
        ocr_texts = [t for t in filtered_texts if t.confidence is not None]
        native_texts = [t for t in filtered_texts if t.confidence is None]

        st.metric("Total Text Spans", len(filtered_texts))

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Native Text", len(native_texts))
        with col_stat2:
            st.metric("OCR Text", len(ocr_texts))
        with col_stat3:
            if ocr_texts:
                avg_conf = sum(t.confidence for t in ocr_texts) / len(ocr_texts)
                st.metric("Avg OCR Confidence", f"{avg_conf:.1f}%")
            else:
                st.metric("Avg OCR Confidence", "N/A")

        # Display text table
        st.subheader("Extracted Text")

        # Prepare table data
        table_data = []
        for i, text_run in enumerate(filtered_texts):
            conf = text_run.confidence if text_run.confidence is not None else "Native"
            source = text_run.source if text_run.source else ("native" if text_run.confidence is None else "ocr")

            # Confidence indicator
            if text_run.confidence is not None:
                if text_run.confidence >= 90:
                    conf_indicator = "HIGH"
                elif text_run.confidence >= 70:
                    conf_indicator = "MEDIUM"
                else:
                    conf_indicator = "LOW"
            else:
                conf_indicator = "NATIVE"

            table_data.append({
                "#": i + 1,
                "Quality": conf_indicator,
                "Text": text_run.text[:80] + "..." if len(text_run.text) > 80 else text_run.text,
                "Confidence": conf,
                "Source": source,
                "Bbox": f"({text_run.bbox[0]:.1f}, {text_run.bbox[1]:.1f}, {text_run.bbox[2]:.1f}, {text_run.bbox[3]:.1f})"
            })

        st.dataframe(
            table_data,
            use_container_width=True,
            height=400
        )

        # Confidence distribution (OCR only)
        if ocr_texts:
            st.subheader("OCR Confidence Distribution")

            confidences = [t.confidence for t in ocr_texts]
            high_conf = sum(1 for c in confidences if c >= 90)
            medium_conf = sum(1 for c in confidences if 70 <= c < 90)
            low_conf = sum(1 for c in confidences if c < 70)

            col_dist1, col_dist2, col_dist3 = st.columns(3)
            with col_dist1:
                st.metric("High (>=90%)", f"{high_conf} ({high_conf/len(ocr_texts)*100:.1f}%)")
            with col_dist2:
                st.metric("Medium (70-89%)", f"{medium_conf} ({medium_conf/len(ocr_texts)*100:.1f}%)")
            with col_dist3:
                st.metric("Low (<70%)", f"{low_conf} ({low_conf/len(ocr_texts)*100:.1f}%)")

            # Show low confidence items
            if low_conf > 0:
                with st.expander(f"Low Confidence Text ({low_conf} items)"):
                    low_conf_items = [t for t in ocr_texts if t.confidence < 70]
                    for t in low_conf_items[:20]:  # Limit to first 20
                        st.write(f"- [{t.confidence}%] \"{t.text}\"")
                    if len(low_conf_items) > 20:
                        st.write(f"... and {len(low_conf_items) - 20} more")

with col2:
    st.subheader("Debug Output")

    # Check for debug images
    debug_dir = Path("./debug/ocr")
    if debug_dir.exists():
        page_prefix = f"page_{selected_page:03d}_"
        debug_images = sorted([
            f for f in debug_dir.glob(f"{page_prefix}*.png")
        ])

        if debug_images:
            st.success(f"Found {len(debug_images)} debug images")

            # Image selector
            image_names = {
                f.name.split('_', 3)[-1].replace('.png', '').replace('_', ' ').title(): str(f)
                for f in debug_images
            }

            selected_image_name = st.selectbox(
                "Select Debug Stage",
                list(image_names.keys())
            )

            if selected_image_name:
                image_path = image_names[selected_image_name]
                st.image(image_path, caption=selected_image_name, use_column_width=True)

                # Download button for selected image
                with open(image_path, "rb") as file:
                    st.download_button(
                        label="Download This Image",
                        data=file,
                        file_name=Path(image_path).name,
                        mime="image/png"
                    )

            # Download all debug images for this page
            st.divider()
            if st.button("Download All Debug Images (ZIP)"):
                import zipfile
                import io

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for img_path in debug_images:
                        zip_file.write(img_path, img_path.name)

                    # Include summary if exists
                    summary_path = debug_dir / f"{page_prefix}08_summary.txt"
                    if summary_path.exists():
                        zip_file.write(summary_path, summary_path.name)

                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"debug_page_{selected_page}.zip",
                    mime="application/zip"
                )

            # Show summary report if exists
            summary_path = debug_dir / f"{page_prefix}08_summary.txt"
            if summary_path.exists():
                with st.expander("Summary Report"):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        st.text(f.read())
        else:
            st.info(f"No debug images found for page {selected_page}.")
            st.caption("Enable 'OCR Debug Mode' in the main page to generate debug output.")
    else:
        st.info("Debug output directory not found.")
        st.caption("Enable 'OCR Debug Mode' in the main page to generate debug output.")

# Download options
st.divider()
st.subheader("Export")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if st.button("Download Text Data (JSON)"):
        # Export text data
        export_data = {
            "doc_id": selected_doc_id,
            "page": selected_page,
            "texts": [
                {
                    "text": t.text,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "source": t.source
                }
                for t in filtered_texts
            ]
        }

        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"page_{selected_page}_confidence.json",
            mime="application/json"
        )

with col_export2:
    if st.button("Download Low Confidence Report (TXT)"):
        ocr_texts = [t for t in filtered_texts if t.confidence is not None]
        low_conf_items = [t for t in ocr_texts if t.confidence < 70]

        report = f"""Low Confidence OCR Report
Document: {selected_doc_id}
Page: {selected_page}
{'='*60}

Total OCR Text: {len(ocr_texts)}
Low Confidence (<70%): {len(low_conf_items)}

Low Confidence Items:
"""
        for t in low_conf_items:
            report += f"\n[{t.confidence}%] {t.text}\n  Bbox: {t.bbox}\n"

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"page_{selected_page}_low_confidence.txt",
            mime="text/plain"
        )

# Help section
with st.expander("Help"):
    st.markdown("""
    ### Confidence Score Guide

    - **HIGH (90-100%)**: Text is clearly readable, high accuracy
    - **MEDIUM (70-89%)**: Acceptable quality, may need review
    - **LOW (<70%)**: Poor quality, likely misread, needs manual review
    - **NATIVE**: Text extracted directly from PDF (100% accurate)

    ### Debug Output Stages

    1. **Original**: Raw PDF render at target DPI
    2. **Grayscale**: Converted to grayscale for OCR
    3. **Preprocessed**: After bilateral filter, thresholding, morphology
    4. **Detections**: Bounding boxes color-coded by confidence
    5. **Tiles**: Tile grid (if tiled OCR was used)
    6. **Final With Text**: Text content overlaid with confidence scores
    7. **Confidence Heatmap**: Color-coded confidence distribution across page
    8. **Summary**: Statistics and low-confidence text list

    ### Tips

    - Use confidence filtering to identify problematic text
    - Review debug output to tune OCR parameters
    - Low confidence often indicates:
      - Low resolution / blurry text
      - Unusual fonts
      - Rotated text
      - Background noise
      - Very small text size
    """)
