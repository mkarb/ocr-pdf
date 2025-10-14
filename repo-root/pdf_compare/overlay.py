from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import fitz

def _try_strip_layers_with_pikepdf(src_pdf: str) -> Optional[str]:
    try:
        import pikepdf
        out_path = os.path.splitext(src_pdf)[0] + ".nolayers.pdf"
        with pikepdf.open(src_pdf) as pdf:
            if "/OCProperties" in pdf.root:
                del pdf.root["/OCProperties"]
            pdf.save(out_path, linearize=False)
        return out_path
    except Exception:
        return None

def _draw_overlay_rects(page: fitz.Page, diff: Dict):
    # geometry overlays (vector compare)
    for x0, y0, x1, y1 in diff["geometry"]["added"]:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1),
                       color=(0, 1, 0), fill=(0, 1, 0, 0.15), width=0.5)
    for x0, y0, x1, y1 in diff["geometry"]["removed"]:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1),
                       color=(1, 0, 0), fill=(1, 0, 0, 0.15), width=0.5)

    # changed regions (raster grid / raster fallback)
    for x0, y0, x1, y1 in diff["geometry"].get("changed", []):
        page.draw_rect(fitz.Rect(x0, y0, x1, y1),
                       color=(0.5, 0.0, 0.8),
                       fill=(0.5, 0.0, 0.8, 0.15),
                       width=0.6)

    # text overlays
    for t in diff["text"]["added"]:
        x0, y0, x1, y1 = t["bbox"]
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 1, 0), width=0.6)
    for t in diff["text"]["removed"]:
        x0, y0, x1, y1 = t["bbox"]
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 0, 0), width=0.6)
    for t in diff["text"]["moved"]:
        x0, y0, x1, y1 = t["from"]
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 0.5, 0), width=0.6)
        x0, y0, x1, y1 = t["to"]
        page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, 0.5, 0), width=1.0, dashes=[2, 2])

def _draw_legend(page: fitz.Page):
    x, y, w, h = 36, 36, 200, 66
    r = fitz.Rect(x, y, x+w, y+h)
    page.draw_rect(r, color=(0,0,0), fill=(1,1,1,0.9), width=0.5)
    page.insert_text((x+10, y+16), "Legend", fontsize=9, color=(0,0,0))
    # swatches
    def box(cx, cy, col):
        page.draw_rect(fitz.Rect(cx, cy, cx+10, cy+10), fill=col, color=col, width=0.2)
    box(x+10,  y+24, (0,1,0));        page.insert_text((x+26, y+33), "Added",   fontsize=8, color=(0,0,0))
    box(x+70,  y+24, (1,0,0));        page.insert_text((x+86, y+33), "Removed", fontsize=8, color=(0,0,0))
    box(x+130, y+24, (0.5,0.0,0.8));  page.insert_text((x+146,y+33), "Changed", fontsize=8, color=(0,0,0))

def write_overlay(base_pdf_path: str, diffs: List[Dict], out_pdf_path: str):
    # First attempt: draw directly on the original
    try:
        doc = fitz.open(base_pdf_path)
        if diffs:
            _draw_legend(doc[diffs[0]["page"] - 1])
        for d in diffs:
            page = doc[d["page"] - 1]
            _draw_overlay_rects(page, d)
        doc.save(out_pdf_path, deflate=True, clean=True)
        doc.close()
        return
    except RuntimeError:
        # Close any partially opened doc
        try: doc.close()
        except Exception: pass

        # Second attempt: strip OCGs then draw
        stripped = _try_strip_layers_with_pikepdf(base_pdf_path)
        if stripped:
            doc2 = fitz.open(stripped)
            if diffs:
                _draw_legend(doc2[diffs[0]["page"] - 1])
            for d in diffs:
                page = doc2[d["page"] - 1]
                _draw_overlay_rects(page, d)
            doc2.save(out_pdf_path, deflate=True, clean=True)
            doc2.close()
            return

        # Final fallback: rasterize background pages, then draw overlays
        base = fitz.open(base_pdf_path)
        out = fitz.open()
        for idx, d in enumerate(diffs):
            src_pg = base[d["page"] - 1]
            # crisper fallback background (~216 DPI)
            mat = fitz.Matrix(3, 3)
            pix = src_pg.get_pixmap(matrix=mat, alpha=False)

            # new page same size as source
            new_pg = out.new_page(width=src_pg.rect.width, height=src_pg.rect.height)
            new_pg.insert_image(src_pg.rect, pixmap=pix)

            # optional legend on first output page
            if idx == 0:
                _draw_legend(new_pg)

            _draw_overlay_rects(new_pg, d)

        out.save(out_pdf_path, deflate=True)
        out.close()
        base.close()
        return


def create_searchable_pdf(
    source_pdf_path: str,
    text_data: List[Tuple[int, str, Tuple[float, float, float, float], str]],
    output_pdf_path: str
) -> None:
    """
    Create a searchable PDF by overlaying invisible OCR text at precise coordinates.

    Args:
        source_pdf_path: Path to the source PDF (typically a scanned image or drawing)
        text_data: List of (page_number, text, (x0, y0, x1, y1), source) tuples
        output_pdf_path: Path where searchable PDF will be saved

    The function overlays invisible text layers at the exact coordinates where OCR
    detected text, making the PDF fully searchable while keeping the original appearance.
    """
    doc = fitz.open(source_pdf_path)

    # Group text by page for efficient processing
    text_by_page: Dict[int, List] = {}
    for page_num, text, bbox, source in text_data:
        if page_num not in text_by_page:
            text_by_page[page_num] = []
        text_by_page[page_num].append((text, bbox, source))

    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        texts = text_by_page.get(page_num + 1, [])  # page_num is 0-indexed, DB is 1-indexed

        if not texts:
            continue

        for text, bbox, source in texts:
            x0, y0, x1, y1 = bbox

            # Calculate text size to fit the bounding box
            bbox_width = x1 - x0
            bbox_height = y1 - y0

            if bbox_width <= 0 or bbox_height <= 0:
                continue

            # Estimate font size based on bbox height
            # OCR bboxes are usually tight around text, so use ~80% of height
            fontsize = bbox_height * 0.8

            # Clamp font size to reasonable values
            fontsize = max(4, min(fontsize, 72))

            try:
                # Insert invisible text at OCR coordinates
                # render_mode=3 makes text invisible but searchable
                # The text is there for PDF search but not visible to users
                page.insert_text(
                    point=(x0, y1),  # Bottom-left of bbox (PDF text baseline)
                    text=text,
                    fontsize=fontsize,
                    render_mode=3,  # 3 = invisible (neither fill nor stroke)
                    color=(0, 0, 0)  # Color doesn't matter for invisible text
                )
            except Exception as e:
                # If text insertion fails (e.g., special characters), skip it
                import sys
                print(f"Warning: Failed to insert text '{text}' at page {page_num+1}: {e}", file=sys.stderr)
                continue

    # Ensure output directory exists
    import os
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the searchable PDF
    doc.save(output_pdf_path, deflate=True, clean=True, garbage=4)
    doc.close()

    print(f"Created searchable PDF with {len(text_data)} text overlays", file=sys.stderr)
