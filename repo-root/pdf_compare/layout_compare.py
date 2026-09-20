"""Compare the placement and extent of PDF content without comparing wording.

Native pages retain text, image and drawing blocks. Pages dominated by a scan
are segmented into visible regions, without an OCR or database dependency. All
reported bounds use displayed page coordinates, including page rotation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import fitz
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree


@dataclass(frozen=True)
class _Block:
    kind: str
    bbox: tuple[float, float, float, float]
    identity: str
    text: str = ""


def _size(bbox):
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def _display_bbox(page, bbox):
    rect = fitz.Rect(bbox) * page.rotation_matrix
    return tuple(float(value) for value in rect)


def _drawing_identity(drawing):
    """Identify path structure independently of its position, scale or color."""
    rect = drawing["rect"]
    width, height = max(rect.width, 1e-6), max(rect.height, 1e-6)

    def normalize(value):
        if isinstance(value, fitz.Point):
            return (round((value.x - rect.x0) / width, 3),
                    round((value.y - rect.y0) / height, 3))
        if isinstance(value, fitz.Rect):
            return (normalize(value.tl), normalize(value.br))
        if isinstance(value, fitz.Quad):
            return tuple(normalize(point) for point in value)
        if isinstance(value, (tuple, list)):
            return tuple(normalize(item) for item in value)
        return value

    return repr(normalize(drawing["items"]))


def _is_scan(page):
    if page is None or page.rect.width * page.rect.height <= 0:
        return False
    threshold = page.rect.width * page.rect.height * 0.75
    rectangles = []
    for info in page.get_image_info():
        overlap = fitz.Rect(info["bbox"]) * page.rotation_matrix & page.rect
        if overlap.is_empty:
            continue
        if overlap.width * overlap.height >= threshold:
            return True
        rectangles.append(box(*overlap))
    # Exporters can split one scanned sheet into tiles. Union coverage avoids
    # counting overlaid logos or repeated image layers as a full-page scan.
    return (sum(rect.area for rect in rectangles) >= threshold
            and unary_union(rectangles).area >= threshold)


def _native_blocks(page):
    blocks = []
    # TEXTFLAGS_TEXT avoids decoding embedded image payloads with text extraction.
    content = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    visibility = {}
    # Trace rendering mode catches invisible OCR text on otherwise native pages.
    # Union visibility at each origin keeps visible overlapping text intact.
    for trace in page.get_texttrace():
        if trace["chars"]:
            origin = tuple(round(value, 3) for value in trace["chars"][0][2])
            visible = trace["type"] != 3 and trace.get("opacity", 1.0) > 0
            visibility[origin] = visibility.get(origin, False) or visible
    for block in content["blocks"]:
        if block.get("type") != 0:
            continue
        spans = [span for line in block.get("lines", [])
                 for span in line.get("spans", [])
                 if (span["alpha"] > 0 if "alpha" in span else visibility.get(
                     tuple(round(value, 3) for value in span["origin"]), True))]
        text = " ".join(span["text"] for span in spans)
        text = " ".join(text.split())
        if text:
            bounds = fitz.Rect(spans[0]["bbox"])
            for span in spans[1:]:
                bounds |= fitz.Rect(span["bbox"])
            blocks.append(_Block("text", _display_bbox(page, bounds), text, text))
    for info in page.get_image_info(hashes=True):
        blocks.append(_Block("image", _display_bbox(page, info["bbox"]),
                             info["digest"].hex()))
    for drawing in page.get_drawings():
        if not (("s" in drawing["type"] and drawing.get("stroke_opacity", 1.0) > 0)
                or ("f" in drawing["type"] and drawing.get("fill_opacity", 1.0) > 0)):
            continue
        bounds = fitz.Rect(drawing["rect"])
        # Strokes have visible extent even when their centerline rect is empty.
        padding = max(float(drawing.get("width") or 1.0) / 2, 0.25)
        if bounds.width == 0:
            bounds.x0 -= padding
            bounds.x1 += padding
        if bounds.height == 0:
            bounds.y0 -= padding
            bounds.y1 += padding
        blocks.append(_Block("drawing", _display_bbox(page, bounds),
                             _drawing_identity(drawing)))
    return blocks


def _raster_blocks(page):
    """Group ink into visible regions; internal pixel edits are not differences."""
    import cv2

    # Preserve thin engineering strokes on large sheets: rendering below one
    # pixel per point can drop native strokes that survive a scanned version.
    scale = min(2.0, 4000.0 / max(page.rect.width, page.rect.height))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                             colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # A white or uniformly colored blank page has no foreground layout.
    if gray.max() - gray.min() < 8:
        return []
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, round(8 * scale)), max(1, round(5 * scale))))
    grouped = cv2.dilate(ink, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    blocks = []
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        mask = (labels[y:y + height, x:x + width] == label) & (
            ink[y:y + height, x:x + width] != 0)
        rows, cols = np.nonzero(mask)
        if len(rows) < max(3, round(2 * scale * scale)):
            continue
        left, top = x + int(cols.min()), y + int(rows.min())
        right, bottom = x + int(cols.max()) + 1, y + int(rows.max()) + 1
        crop = ink[top:bottom, left:right]
        signature = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA) > 127
        digest = sha256(signature.tobytes()).hexdigest()
        bounds = (left / scale, top / scale, right / scale, bottom / scale)
        blocks.append(_Block("region", bounds, digest))
    return blocks


def _blocks(page, raster):
    if page is None:
        return []
    blocks = _raster_blocks(page) if raster else _native_blocks(page)
    visible = []
    for block in blocks:
        if not all(isfinite(value) for value in block.bbox):
            continue
        bounds = fitz.Rect(block.bbox) & page.rect
        if not bounds.is_empty:
            visible.append(_Block(block.kind, tuple(bounds), block.identity, block.text))
    return sorted(visible, key=lambda block: (block.kind, block.bbox, block.identity))


def _nearby_candidates(old, new, old_indices, new_indices, position_tol, size_tol):
    """Find unchanged bounds through a small spatial index, including jitter."""
    cell = max(position_tol, 1.0)
    buckets = defaultdict(list)
    for j in new_indices:
        block = new[j]
        buckets[(block.kind, int(block.bbox[0] // cell),
                 int(block.bbox[1] // cell))].append(j)
    candidates = []
    for i in old_indices:
        block = old[i]
        x, y = int(block.bbox[0] // cell), int(block.bbox[1] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets[(block.kind, x + dx, y + dy)]:
                    target = new[j].bbox
                    if (max(abs(block.bbox[k] - target[k]) for k in (0, 1)) <= position_tol
                            and max(abs(a - b) for a, b in zip(
                                _size(block.bbox), _size(target))) <= size_tol):
                        candidates.append((_distance(block.bbox, target), i, j))
    return candidates


def _spatial_score(a, b, position_tol):
    """Allow local block edits, but avoid pairing unrelated distant additions."""
    aw, ah = _size(a)
    bw, bh = _size(b)
    size_ratio = min((min(x, y) + 1) / (max(x, y) + 1)
                     for x, y in ((aw, bw), (ah, bh)))
    anchor = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    if anchor <= max(12.0, 2 * position_tol) and size_ratio >= 0.1:
        return _distance(a, b)
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    if size_ratio >= 0.25 and overlap / max(1.0, min(aw * ah, bw * bh)) >= 0.5:
        return _distance(a, b)
    return None


def _match(old, new, position_tol, size_tol):
    matches, used = {}, set()

    def accept(candidates):
        for _, i, j in sorted(candidates):
            if i not in matches and j not in used:
                matches[i] = j
                used.add(j)

    # Reserve fixed placements first, even if their wording or colors changed.
    exact = defaultdict(deque)
    for j, block in enumerate(new):
        exact[(block.kind, block.bbox)].append(j)
    for i, block in enumerate(old):
        queue = exact[(block.kind, block.bbox)]
        if queue:
            j = queue.popleft()
            matches[i] = j
            used.add(j)
    accept(_nearby_candidates(old, new,
                             [i for i in range(len(old)) if i not in matches],
                             [j for j in range(len(new)) if j not in used],
                             position_tol, size_tol))

    # Identity provides evidence for movement across the entire page. Repeated
    # labels use nearest pairs, after all unchanged occurrences are reserved.
    old_groups, new_groups = defaultdict(list), defaultdict(list)
    for i, block in enumerate(old):
        if i not in matches:
            old_groups[(block.kind, block.identity)].append(i)
    for j, block in enumerate(new):
        if j not in used:
            new_groups[(block.kind, block.identity)].append(j)
    for key, old_indices in old_groups.items():
        new_indices = new_groups[key]
        if len(old_indices) * len(new_indices) <= 100_000:
            accept((_distance(old[i].bbox, new[j].bbox), i, j)
                   for i in old_indices for j in new_indices)
        else:
            # Bound candidate memory for drawings with thousands of identical
            # paths. Selecting nearest remaining candidates is deterministic.
            bounds = np.asarray([new[j].bbox for j in new_indices])
            available = np.ones(len(new_indices), dtype=bool)
            for i in old_indices:
                if not available.any():
                    break
                distances = np.abs(bounds - np.asarray(old[i].bbox)).sum(axis=1)
                distances[~available] = np.inf
                target = int(distances.argmin())
                j = new_indices[target]
                matches[i] = j
                used.add(j)
                available[target] = False

    # A nearby replacement can retain its layout without retaining its text or
    # image bytes. Disjoint replacements with no identity remain added/removed.
    candidates = []
    spatial_groups = defaultdict(list)
    for j, block in enumerate(new):
        if j not in used:
            spatial_groups[block.kind].append(j)
    trees = {kind: STRtree([box(*new[j].bbox) for j in indices])
             for kind, indices in spatial_groups.items()}
    padding = max(12.0, 2 * position_tol)
    for i, a in enumerate(old):
        if i in matches or a.kind not in trees:
            continue
        search = box(a.bbox[0] - padding, a.bbox[1] - padding,
                     a.bbox[2] + padding, a.bbox[3] + padding)
        for index in trees[a.kind].query(search):
            j = spatial_groups[a.kind][int(index)]
            b = new[j]
            score = _spatial_score(a.bbox, b.bbox, position_tol)
            if score is not None:
                candidates.append((score, i, j))
    accept(candidates)
    return matches, used


def _item(block):
    item = {"kind": block.kind, "bbox": block.bbox}
    if block.text:
        item["text"] = block.text
    return item


def diff_layout_files(old_pdf, new_pdf, *, position_tolerance=3.0, size_tolerance=3.0):
    """Return block layout differences for corresponding PDF page numbers.

    Tolerances are finite, nonnegative PDF points (72 points per inch). Movement
    measures the block's top-left position; resizing measures width and height.
    A block can be both moved and resized. Text and image content only help match
    blocks: content edits at unchanged bounds are never reported. For scans,
    region segmentation compares occupied bounds rather than recognizing words.
    """
    for name, value in (("position_tolerance", position_tolerance),
                        ("size_tolerance", size_tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")

    diffs = []
    with fitz.open(old_pdf) as old_doc, fitz.open(new_pdf) as new_doc:
        for index in range(max(len(old_doc), len(new_doc))):
            old_page = old_doc[index] if index < len(old_doc) else None
            new_page = new_doc[index] if index < len(new_doc) else None
            raster = _is_scan(old_page) or _is_scan(new_page)
            old, new = _blocks(old_page, raster), _blocks(new_page, raster)
            layout = {"added": [], "removed": [], "moved": [], "resized": [], "changed": []}
            diff = {"page": index + 1, "coordinate_space": "display",
                    "geometry": {"added": [], "removed": [], "changed": []},
                    "text": {"added": [], "removed": [], "moved": []},
                    "layout": layout, "layout_source": "raster" if raster else "native"}
            if old_page is None or new_page is None:
                page = new_page if new_page is not None else old_page
                diff["page_status"] = "added" if old_page is None else "removed"
                diff["page_size"] = (page.rect.width, page.rect.height)
            else:
                old_size = (old_page.rect.width, old_page.rect.height)
                new_size = (new_page.rect.width, new_page.rect.height)
                if (max(abs(a - b) for a, b in zip(old_size, new_size)) > size_tolerance
                        or old_page.rotation != new_page.rotation):
                    diff.update(layout_page_changed=True, old_page_size=old_size,
                                new_page_size=new_size, page_size=new_size,
                                old_rotation=old_page.rotation, new_rotation=new_page.rotation)
            matches, used = _match(old, new, position_tolerance, size_tolerance)
            for i, block in enumerate(old):
                if i not in matches:
                    layout["removed"].append(_item(block))
                    continue
                target = new[matches[i]]
                item = {"kind": block.kind, "from": block.bbox, "to": target.bbox}
                if block.text:
                    item["text"] = block.text
                if max(abs(block.bbox[k] - target.bbox[k]) for k in (0, 1)) > position_tolerance:
                    layout["moved"].append(item)
                if max(abs(a - b) for a, b in zip(_size(block.bbox), _size(target.bbox))) > size_tolerance:
                    layout["resized"].append(item)
            layout["added"] = [_item(block) for j, block in enumerate(new) if j not in used]
            diffs.append(diff)
    return diffs
