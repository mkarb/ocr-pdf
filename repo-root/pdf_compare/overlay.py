from __future__ import annotations
from typing import List, Dict, Optional
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
