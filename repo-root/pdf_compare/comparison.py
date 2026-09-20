"""File-based visual comparison shared by the CLI and Streamlit UI."""

from pathlib import Path

import fitz

from .raster_grid import raster_grid_changed_boxes


def resolve_overlay_sources(old_pdf: str, new_pdf: str, base_pdf: str | None = None) -> tuple[str, str]:
    """Keep overlay coordinates and source protection tied to the two revisions."""
    base = base_pdf or new_pdf
    if Path(base).resolve() == Path(new_pdf).resolve():
        return base, old_pdf
    if Path(base).resolve() == Path(old_pdf).resolve():
        return base, new_pdf
    raise ValueError("The overlay base must be the old or new comparison PDF")


def diff_pdf_files(
    old_pdf: str,
    new_pdf: str,
    *,
    base_pdf: str | None = None,
    dpi: int = 400,
    rows: int = 12,
    cols: int = 16,
    cell_change_ratio: float = 0.0,
) -> list[dict]:
    """Compare corresponding pages, including added and removed trailing pages.

    Raster boxes use the selected base page's displayed coordinates. Page order
    is positional; inserted pages are not automatically matched to other indices.
    """
    base, _ = resolve_overlay_sources(old_pdf, new_pdf, base_pdf)
    base_is_old = Path(base).resolve() == Path(old_pdf).resolve()
    diffs = []
    with fitz.open(old_pdf) as old, fitz.open(new_pdf) as new:
        for index in range(max(len(old), len(new))):
            diff = {
                "page": index + 1,
                "coordinate_space": "display",
                "geometry": {"added": [], "removed": [], "changed": []},
                "text": {"added": [], "removed": [], "moved": []},
            }
            if index >= min(len(old), len(new)):
                added = index >= len(old)
                page = new[index] if added else old[index]
                diff["page_status"] = "added" if added else "removed"
                diff["page_size"] = (page.rect.width, page.rect.height)
                diff["geometry"]["added" if added else "removed"] = [tuple(page.rect)]
            else:
                source, target = (new_pdf, old_pdf) if base_is_old else (old_pdf, new_pdf)
                diff["geometry"]["changed"] = raster_grid_changed_boxes(
                    source, target, index, dpi=dpi, rows=rows, cols=cols,
                    method="hybrid", cell_change_ratio=cell_change_ratio,
                    merge_adjacent=True,
                )
            diffs.append(diff)
    return diffs
