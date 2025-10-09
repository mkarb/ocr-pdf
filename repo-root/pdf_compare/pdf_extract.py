from __future__ import annotations
from typing import List, Tuple
import hashlib
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.wkb import dumps as wkb_dumps
from .models import VectorMap, DocMeta, PageVectors, VectorGeom, GeoKind, TextRun, BBox

# Tunables
MIN_LEN = 0.5         # drop tiny strokes
MIN_AREA = 0.5        # drop tiny fills
BEZIER_SAMPLES = 24   # curve sampling for cubic bezier
DEFAULT_ROTATIONS = {0, 90, 180, 270}

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def _cubic_sample(p0, p1, p2, p3, n=BEZIER_SAMPLES):
    t = np.linspace(0, 1, n)
    xs = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
    ys = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
    return list(zip(xs.tolist(), ys.tolist()))

def _drawings_to_geoms(page) -> List[VectorGeom]:
    out: List[VectorGeom] = []
    for d in page.get_drawings():
        # strokes (path segments)
        for item in d["items"]:
            op = item[0]
            if op == "l":  # line (p0, p1)
                _, p0, p1 = item
                if p0 != p1:
                    ls = LineString([p0, p1])
                    if ls.length >= MIN_LEN:
                        out.append(VectorGeom(GeoKind.STROKE, wkb_dumps(ls), ls.bounds))
            elif op == "c":  # cubic bezier (p0,p1,p2,p3)
                _, p0, p1, p2, p3 = item
                pts = _cubic_sample(p0, p1, p2, p3)
                ls = LineString(pts)
                if ls.length >= MIN_LEN:
                    out.append(VectorGeom(GeoKind.STROKE, wkb_dumps(ls), ls.bounds))
        # simple rect fill (PyMuPDF exposes as "rect" sometimes)
        if d.get("fill") and d.get("rect"):
            x0, y0, x1, y1 = d["rect"]
            poly = box(x0, y0, x1, y1)
            if poly.area >= MIN_AREA:
                out.append(VectorGeom(GeoKind.FILL, wkb_dumps(poly), poly.bounds))
    return out

def _extract_text(page) -> List[TextRun]:
    runs: List[TextRun] = []
    raw = page.get_text("rawdict") or {}
    for blk in raw.get("blocks", []):
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                bbox = tuple(span["bbox"])  # type: ignore
                runs.append(TextRun(txt, bbox, span.get("font"), span.get("size")))
    return runs

def pdf_to_vectormap(path: str, doc_id: str | None = None) -> VectorMap:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    doc = fitz.open(path)
    if doc_id is None:
        doc_id = _hash_file(p)
    pages: List[PageVectors] = []
    for i in range(doc.page_count):
        pg = doc[i]
        rotation = pg.rotation
        if rotation not in DEFAULT_ROTATIONS:
            rotation = 0
        geoms = _drawings_to_geoms(pg)
        texts = _extract_text(pg)
        pages.append(PageVectors(
            page_number=i+1,
            width=pg.rect.width, height=pg.rect.height, rotation=rotation,
            geoms=geoms, texts=texts
        ))
    meta = DocMeta(doc_id=doc_id, path=str(p.resolve()), page_count=doc.page_count)
    return VectorMap(meta=meta, pages=pages)
