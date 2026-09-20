from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import sys
import tempfile
import fitz

def _try_strip_layers_with_pikepdf(src_pdf: str) -> Optional[str]:
    out_path = None
    try:
        import pikepdf
        fd, out_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with pikepdf.open(src_pdf) as pdf:
            if "/OCProperties" in pdf.Root:
                del pdf.Root["/OCProperties"]
            pdf.save(out_path, linearize=False)
        return out_path
    except Exception:
        if out_path is not None:
            os.unlink(out_path)
        return None

def _draw_overlay_rects(page: fitz.Page, diff: Dict):
    # Raster and layout coordinates describe the rotated, displayed page;
    # PyMuPDF drawing operations always take unrotated page coordinates.
    def rect(box):
        result = fitz.Rect(box)
        if diff.get("coordinate_space") == "display":
            result = result * page.derotation_matrix
        return result

    for kind, color in (
        ("added", (0, 1, 0)),
        ("removed", (1, 0, 0)),
        ("changed", (0.5, 0, 0.8)),
    ):
        for box in diff.get("geometry", {}).get(kind, []):
            # Four-component fill tuples mean CMYK, not RGBA.
            page.draw_rect(rect(box), color=color, fill=color,
                           fill_opacity=0.15, width=0.6)

    text = diff.get("text", {})
    for kind, color in (("added", (0, 1, 0)), ("removed", (1, 0, 0))):
        for item in text.get(kind, []):
            page.draw_rect(rect(item["bbox"]), color=color, width=0.6)
    for item in text.get("moved", []):
        page.draw_rect(rect(item["from"]), color=(1, 0.5, 0), width=0.6)
        page.draw_rect(rect(item["to"]), color=(1, 0.5, 0),
                       width=1.0, dashes="[2 2] 0")

    layout = diff.get("layout", {})
    for kind, color in (("added", (0, 1, 0)), ("removed", (1, 0, 0))):
        for item in layout.get(kind, []):
            page.draw_rect(rect(item["bbox"]), color=color, fill=color,
                           fill_opacity=0.1, width=0.8)
    for kind, color, width in (
        ("moved", (1, 0.5, 0), 1.8),
        ("resized", (0.5, 0, 0.8), 0.8),
        ("changed", (0.5, 0, 0.8), 0.8),
    ):
        for item in layout.get(kind, []):
            # A block may move and resize together. The wider orange stroke
            # stays visible around the narrower purple stroke in that case.
            page.draw_rect(rect(item["from"]), color=color, width=width)
            page.draw_rect(rect(item["to"]), color=color, width=width,
                           dashes="[2 2] 0")

    if diff.get("layout_page_changed"):
        bounds = page.rect + (2, 2, -2, -2)
        page.draw_rect(bounds * page.derotation_matrix,
                       color=(0.5, 0, 0.8), width=2)

    status = diff.get("page_status")
    if status in ("added", "removed"):
        color = (0, 0.7, 0) if status == "added" else (1, 0, 0)
        bounds = page.rect + (2, 2, -2, -2)
        page.draw_rect(bounds * page.derotation_matrix, color=color, width=2)
        page.insert_text(fitz.Point(12, page.rect.height - 12) * page.derotation_matrix,
                         f"Page {status}", fontsize=10, color=color, rotate=page.rotation)

def _legend_position(page: fitz.Page, width: float, height: float, margin: float):
    """Find a blank displayed corner without obscuring content or highlights."""
    zoom = min(1.0, 1200 / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY,
                         alpha=False)
    pixels = pix.samples_mv
    for x, y in (
        (margin, margin),
        (page.rect.width - margin - width, margin),
        (page.rect.width - margin - width, page.rect.height - margin - height),
        (margin, page.rect.height - margin - height),
    ):
        # Include the border and a little padding when checking for content.
        bounds = fitz.Rect(x - 1, y - 1, x + width + 1, y + height + 1)
        bounds = (bounds * fitz.Matrix(zoom, zoom)).irect
        bounds &= fitz.IRect(0, 0, pix.width, pix.height)
        if all(min(pixels[row * pix.stride + bounds.x0:row * pix.stride + bounds.x1],
                   default=255) >= 250 for row in range(bounds.y0, bounds.y1)):
            return x, y
    return None


def _draw_legend(page: fitz.Page, *, layout: bool = False):
    purple_label = "Resized" if layout else "Changed"
    height = 94 if layout else 66
    margin = min(36, page.rect.width * 0.05, page.rect.height * 0.05)
    scale = min(1.0, (page.rect.width - 2 * margin) / 200,
                (page.rect.height - 2 * margin) / height)
    origin = _legend_position(page, 200 * scale, height * scale, margin)
    if origin is None:
        # Dense drawings may have no room for a legend. Keep the explanation
        # in a PDF comment instead of painting a large panel over the drawing.
        display_rect = fitz.Rect(page.rect.width - 22, 2, page.rect.width - 4, 20)
        note_rect = display_rect * page.derotation_matrix
        description = ("Legend\nGreen: Added\nRed: Removed\n"
                       f"Purple: {purple_label}\nOrange: Moved")
        if layout:
            description += ("\nSolid: Previous position/size\nDashed: New position/size"
                            "\nPurple page border: Page size or rotation changed")
        note = page.add_text_annot(note_rect.tl, description,
                                  icon="Comment")
        note.set_flags(note.flags & ~fitz.PDF_ANNOT_IS_NO_ROTATE)
        note.set_rect(note_rect)
        note.set_info(title="PDF comparison legend")
        note.update()
        return
    origin_x, origin_y = origin

    def point(x, y):
        return fitz.Point(origin_x + x * scale, origin_y + y * scale) * page.derotation_matrix

    def rectangle(x0, y0, x1, y1):
        return fitz.Rect(origin_x + x0 * scale, origin_y + y0 * scale,
                         origin_x + x1 * scale, origin_y + y1 * scale) * page.derotation_matrix

    page.draw_rect(rectangle(0, 0, 200, height), color=(0, 0, 0),
                   fill=(1, 1, 1), fill_opacity=0.9, width=0.5)
    page.insert_text(point(10, 16), "Legend", fontsize=9 * scale,
                     color=(0, 0, 0), rotate=page.rotation)
    for x, y, label, color in (
        (10, 24, "Added", (0, 1, 0)),
        (104, 24, "Removed", (1, 0, 0)),
        (10, 44, purple_label, (0.5, 0, 0.8)),
        (104, 44, "Moved", (1, 0.5, 0)),
    ):
        page.draw_rect(rectangle(x, y, x + 10, y + 10),
                       fill=color, color=color, width=0.2)
        page.insert_text(point(x + 16, y + 9), label, fontsize=8 * scale,
                         color=(0, 0, 0), rotate=page.rotation)
    if layout:
        for y, label in (
            (70, "Solid: before   Dashed: after"),
            (84, "Purple border: page size/rotation"),
        ):
            page.insert_text(point(10, y), label, fontsize=8 * scale,
                             color=(0, 0, 0), rotate=page.rotation)


def _append_missing_pages(doc: fitz.Document, diffs: List[Dict], alternate_pdf_path: Optional[str]):
    last_page = max((d["page"] for d in diffs), default=0)
    if last_page <= len(doc):
        return
    if alternate_pdf_path is None:
        raise ValueError(f"Page {last_page} is missing from the base PDF; provide alternate_pdf_path")
    with fitz.open(alternate_pdf_path) as alternate:
        if last_page > len(alternate):
            raise ValueError(f"Page {last_page} is missing from both comparison PDFs")
        doc.insert_pdf(alternate, from_page=len(doc), to_page=last_page - 1)


def _apply_overlays(doc: fitz.Document, diffs: List[Dict]):
    for diff in diffs:
        _draw_overlay_rects(doc[diff["page"] - 1], diff)
    if diffs:
        _draw_legend(doc[diffs[0]["page"] - 1],
                     layout=any("layout" in diff for diff in diffs))


def write_overlay(base_pdf_path: str, diffs: List[Dict], out_pdf_path: str,
                  *, alternate_pdf_path: Optional[str] = None):
    """Preserve the base PDF and highlight differences at their page numbers.

    Missing trailing pages are copied from ``alternate_pdf_path``. Vector boxes
    use unrotated coordinates; raster/layout diffs must set ``coordinate_space`` to
    ``"display"``. Entire-page changes can set ``page_status`` to added/removed.
    """
    for diff in diffs:
        if not isinstance(diff.get("page"), int) or diff["page"] < 1:
            raise ValueError("Overlay page numbers must be positive integers")
    output_path = os.path.normcase(os.path.realpath(out_pdf_path))
    for source in (base_pdf_path, alternate_pdf_path):
        if source and os.path.normcase(os.path.realpath(source)) == output_path:
            raise ValueError("Overlay output must be different from the source PDFs")
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf_path)), exist_ok=True)

    def write_vector(path):
        with fitz.open(path) as doc:
            _append_missing_pages(doc, diffs, alternate_pdf_path)
            _apply_overlays(doc, diffs)
            doc.save(out_pdf_path, deflate=True, clean=True)

    try:
        write_vector(base_pdf_path)
        return
    except RuntimeError:
        pass

    stripped = _try_strip_layers_with_pikepdf(base_pdf_path)
    if stripped:
        try:
            write_vector(stripped)
            return
        except RuntimeError:
            pass
        finally:
            os.unlink(stripped)

    # Preserve every page and its rotation even when a vector save fails.
    with fitz.open(base_pdf_path) as base, fitz.open() as out:
        _append_missing_pages(base, diffs, alternate_pdf_path)
        for src_pg in base:
            rotation = src_pg.rotation
            src_pg.set_rotation(0)
            bounds = src_pg.rect
            zoom = min(3.0, 4000 / max(bounds.width, bounds.height))
            pix = src_pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            new_pg = out.new_page(width=bounds.width, height=bounds.height)
            new_pg.insert_image(new_pg.rect, pixmap=pix)
            new_pg.set_rotation(rotation)
        _apply_overlays(out, diffs)
        out.save(out_pdf_path, deflate=True)


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
                print(f"Warning: Failed to insert text '{text}' at page {page_num+1}: {e}", file=sys.stderr)
                continue

    # Ensure output directory exists
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the searchable PDF
    doc.save(output_pdf_path, deflate=True, clean=True, garbage=4)
    doc.close()

    print(f"Created searchable PDF with {len(text_data)} text overlays", file=sys.stderr)
