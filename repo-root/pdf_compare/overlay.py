# overlay.py
from __future__ import annotations
from typing import List, Dict, Optional
import os
import fitz

def _try_strip_layers_with_pikepdf(src_pdf: str) -> Optional[str]:
    """
    If the PDF has broken /OCProperties, removing them avoids MuPDF's
    'No default Layer config' error. Requires optional 'pikepdf'.
    Returns path to a sanitized temp copy, or None if not available/failed.
    """
    try:
        import pikepdf
        out_path = os.path.splitext(src_pdf)[0] + ".nolayers.pdf"
        with pikepdf.open(src_pdf) as pdf:
            if "/OCProperties" in pdf.root:
                del pdf.root["/OCProperties"]
            # Scrub minor issues too
            pdf.save(out_path, linearize=False)
        return out_path
    except Exception:
        return None

def _draw_overlay_rects(page: fitz.Page, diff: Dict):
    # geometry overlays
    for x0, y0, x1, y1 in diff["geometry"]["added"]:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1),
                       color=(0, 1, 0), fill=(0, 1, 0, 0.15), width=0.5)
    for x0, y0, x1, y1 in diff["geometry"]["removed"]:
        page.draw_rect(fitz.Rect(x0, y0, x1, y1),
                       color=(1, 0, 0), fill=(1, 0, 0, 0.15), width=0.5)
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

def write_overlay(base_pdf_path: str, diffs: List[Dict], out_pdf_path: str):
    """
    Attempt overlay directly. On MuPDF 'No default Layer config' failure,
    sanitize layers with pikepdf; if that still fails, rasterize pages and draw overlays.
    """
    # First attempt: direct overlay on original
    try:
        doc = fitz.open(base_pdf_path)
        for d in diffs:
            p = d["page"]
            page = doc[p - 1]
            _draw_overlay_rects(page, d)
        # Full save (not incremental), with cleaning
        doc.save(out_pdf_path, deflate=True, clean=True)
        doc.close()
        return
    except RuntimeError as e:
        # Typical message contains 'No default Layer config'
        err_msg = str(e)
        # Close any partially opened doc
        try:
            doc.close()
        except Exception:
            pass

        # Second attempt: strip OCGs then draw
        stripped = _try_strip_layers_with_pikepdf(base_pdf_path)
        if stripped:
            doc2 = fitz.open(stripped)
            for d in diffs:
                p = d["page"]
                page = doc2[p - 1]
                _draw_overlay_rects(page, d)
            doc2.save(out_pdf_path, deflate=True, clean=True)
            doc2.close()
            return

        # Final fallback: build a new PDF by rasterizing each base page then drawing overlays
        base = fitz.open(base_pdf_path)
        out = fitz.open()
        for d in diffs:
            p = d["page"]
            src_pg = base[p - 1]
            # render background
            mat = fitz.Matrix(2, 2)  # ~144 DPI; increase for higher quality
            pix = src_pg.get_pixmap(matrix=mat, alpha=False)
            # create new page same size as source
            new_pg = out.new_page(width=src_pg.rect.width, height=src_pg.rect.height)
            # place rasterized background
            rect = src_pg.rect
            new_pg.insert_image(rect, pixmap=pix)
            # draw overlays
            _draw_overlay_rects(new_pg, d)
        out.save(out_pdf_path, deflate=True)
        out.close()
        base.close()
        return
