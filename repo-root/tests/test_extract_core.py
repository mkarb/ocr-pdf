"""Unit tests for pdf_compare._extract_core.

Uses lightweight fake page objects so the tests are deterministic and do not
depend on a specific PyMuPDF version's drawing decomposition. The point is to
verify that drawings_to_geoms emits geometry for every PyMuPDF path op,
especially the "re" (rectangle) and "qu" (quad) ops that the previous
implementation silently dropped.
"""
from types import SimpleNamespace

from shapely.wkb import loads as wkb_loads

from pdf_compare._extract_core import drawings_to_geoms


class _FakePage:
    """Minimal stand-in for fitz.Page exposing only get_drawings()."""

    def __init__(self, drawings):
        self._drawings = drawings

    def get_drawings(self):
        return self._drawings


def _run(items):
    page = _FakePage([{"items": items, "fill": None, "rect": None}])
    return drawings_to_geoms(
        page,
        min_segment_len=0.5,
        min_fill_area=0.5,
        bezier_samples=24,
        simplify_tolerance=None,
    )


def test_line_op_extracted():
    geoms = _run([("l", (0.0, 0.0), (100.0, 0.0))])
    assert [g["kind"] for g in geoms] == ["STROKE"]


def test_rectangle_op_extracted():
    # Regression: "re" path ops used to be dropped entirely.
    geoms = _run([("re", (10.0, 10.0, 110.0, 60.0), 1)])
    assert len(geoms) == 1
    assert geoms[0]["kind"] == "STROKE"
    # Closed ring spanning the rectangle corners.
    assert geoms[0]["bbox"] == (10.0, 10.0, 110.0, 60.0)
    ring = wkb_loads(geoms[0]["wkb"])
    assert ring.coords[0] == ring.coords[-1]  # closed


def test_quad_op_extracted():
    # Regression: "qu" path ops used to be dropped entirely.
    quad = SimpleNamespace(
        ul=(0.0, 0.0), ur=(50.0, 0.0), lr=(50.0, 30.0), ll=(0.0, 30.0)
    )
    geoms = _run([("qu", quad)])
    assert len(geoms) == 1
    assert geoms[0]["kind"] == "STROKE"
    assert geoms[0]["bbox"] == (0.0, 0.0, 50.0, 30.0)


def test_rect_fill_extracted():
    page = _FakePage([{"items": [], "fill": (0, 0, 0), "rect": (0.0, 0.0, 20.0, 20.0)}])
    geoms = drawings_to_geoms(page, 0.5, 0.5, 24, None)
    assert [g["kind"] for g in geoms] == ["FILL"]


def test_degenerate_shapes_filtered():
    # Zero-length line and a sub-threshold fill are both dropped.
    page = _FakePage([
        {"items": [("l", (5.0, 5.0), (5.0, 5.0))], "fill": (0, 0, 0), "rect": (0.0, 0.0, 0.1, 0.1)},
    ])
    assert drawings_to_geoms(page, 0.5, 0.5, 24, None) == []
