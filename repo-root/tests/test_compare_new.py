"""Vector/text comparison regressions using the database backend contract."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from shapely.geometry import LineString, Polygon, box

from pdf_compare.compare_new import diff_documents, diff_pages
from pdf_compare.db_backend import DatabaseBackend
from pdf_compare.db_models import Document, Page


class MemoryBackend(DatabaseBackend):
    """Exercise comparison without requiring a running PostgreSQL service."""

    def __init__(self, old_pages=1, new_pages=1):
        self.documents = {
            "old": Document(doc_id="old", path="old.pdf", page_count=old_pages),
            "new": Document(doc_id="new", path="new.pdf", page_count=new_pages),
        }
        self.geometries = {}
        self.texts = {}

    @contextmanager
    def SessionLocal(self):
        def get(model, key):
            if model is Document:
                return self.documents.get(key)
            if model is Page:
                doc_id, page = key
                if 1 <= page <= self.documents[doc_id].page_count:
                    return Page(doc_id=doc_id, page_number=page, width=612, height=792)
            return None

        yield SimpleNamespace(get=get)

    def load_page_geoms(self, doc_id, page):
        return [geom.wkb for geom in self.geometries.get((doc_id, page), [])]

    def load_page_texts(self, doc_id, page):
        return self.texts.get((doc_id, page), [])


def compare_geometry(old, new):
    backend = MemoryBackend()
    backend.geometries = {("old", 1): old, ("new", 1): new}
    return diff_pages(backend, "old", "new", 1)["geometry"]


@pytest.mark.parametrize(
    "old,new",
    [
        (LineString([(0, 0), (100, 0)]), LineString([(0, 0), (50, 0)])),
        (box(0, 0, 100, 100), box(10, 10, 90, 90)),
        (box(0, 0, 100, 100), Polygon(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            holes=[[(20, 20), (80, 20), (80, 80), (20, 80)]],
        )),
    ],
)
def test_contained_geometry_is_a_change(old, new):
    result = compare_geometry([old], [new])
    assert result["removed"] == [old.bounds]
    assert result["added"] == [new.bounds]


@pytest.mark.parametrize("offset,changed", [(0, False), (0.1, False), (0.2, True)])
def test_geometry_tolerance_is_a_distance(offset, changed):
    old = LineString([(0, 0), (100, 0)])
    new = LineString([(0, offset), (100, offset)])
    result = compare_geometry([old], [new])
    assert bool(result["added"]) is changed
    assert bool(result["removed"]) is changed


def test_geometry_additions_and_removals_keep_original_bounds():
    old = LineString([(0, 0), (10, 0)])
    new = LineString([(20, 20), (30, 20)])
    assert compare_geometry([old], [new]) == {
        "added": [new.bounds], "removed": [old.bounds], "changed": []
    }


def test_empty_geometry_does_not_emit_nan_bounds():
    assert compare_geometry([LineString()], []) == {
        "added": [], "removed": [], "changed": []
    }


def text_at(x, *, width=4):
    return ("Valve", (x, 0, x + width, 10))


@pytest.mark.parametrize("old_order", [[0, 10], [10, 0]])
def test_duplicate_text_removal_preserves_unchanged_occurrence(old_order):
    backend = MemoryBackend()
    backend.texts = {
        ("old", 1): [text_at(x) for x in old_order],
        ("new", 1): [text_at(10)],
    }
    result = diff_pages(backend, "old", "new", 1)["text"]
    assert result == {
        "removed": [{"text": "Valve", "bbox": text_at(0)[1]}],
        "added": [], "moved": [],
    }


def test_duplicate_text_near_unchanged_match_has_priority():
    backend = MemoryBackend()
    backend.texts = {
        ("old", 1): [text_at(0), text_at(10)],
        ("new", 1): [text_at(10.1)],
    }
    result = diff_pages(backend, "old", "new", 1)["text"]
    assert result["removed"] == [{"text": "Valve", "bbox": text_at(0)[1]}]
    assert result["moved"] == []


def test_duplicate_text_movement_does_not_steal_unchanged_occurrence():
    backend = MemoryBackend()
    backend.texts = {
        ("old", 1): [text_at(0), text_at(10)],
        ("new", 1): [text_at(10), text_at(30)],
    }
    result = diff_pages(backend, "old", "new", 1)["text"]
    assert result == {
        "removed": [], "added": [],
        "moved": [{"text": "Valve", "from": text_at(0)[1], "to": text_at(30)[1]}],
    }


def test_text_bounds_change_with_same_center_is_reported():
    backend = MemoryBackend()
    old, new = text_at(0), text_at(-2, width=8)
    backend.texts = {("old", 1): [old], ("new", 1): [new]}
    result = diff_pages(backend, "old", "new", 1)["text"]
    assert result["moved"] == [{"text": "Valve", "from": old[1], "to": new[1]}]


def test_replacement_text_is_added_and_removed():
    backend = MemoryBackend()
    bbox = (0, 0, 10, 10)
    backend.texts = {("old", 1): [("OLD", bbox)], ("new", 1): [("NEW", bbox)]}
    result = diff_pages(backend, "old", "new", 1)["text"]
    assert result == {
        "removed": [{"text": "OLD", "bbox": bbox}],
        "added": [{"text": "NEW", "bbox": bbox}], "moved": [],
    }


@pytest.mark.parametrize("counts,status", [((1, 2), "added"), ((2, 1), "removed")])
def test_extra_blank_page_is_reported(counts, status):
    backend = MemoryBackend(*counts)
    result = diff_documents(backend, "old", "new")
    assert [page["page"] for page in result] == [1, 2]
    assert result[1]["page_status"] == status
    assert result[1]["page_size"] == (612, 792)


def test_extra_page_content_is_reported():
    backend = MemoryBackend(1, 2)
    geom = box(0, 0, 10, 10)
    backend.geometries[("new", 2)] = [geom]
    backend.texts[("new", 2)] = [text_at(0)]
    result = diff_pages(backend, "old", "new", 2)
    assert result["page_status"] == "added"
    assert result["geometry"]["added"] == [geom.bounds]
    assert result["text"]["added"] == [{"text": "Valve", "bbox": text_at(0)[1]}]


@pytest.mark.parametrize("page", [0, -1, 2, 1.5, True])
def test_invalid_page_selection_is_rejected(page):
    with pytest.raises(ValueError, match="page"):
        diff_pages(MemoryBackend(), "old", "new", page)


def test_unknown_document_is_rejected_by_single_page_api():
    with pytest.raises(ValueError, match="Unknown doc_id"):
        diff_pages(MemoryBackend(), "missing", "new", 1)


def test_explicit_empty_page_selection_is_empty():
    assert diff_documents(MemoryBackend(), "old", "new", pages=[]) == []
