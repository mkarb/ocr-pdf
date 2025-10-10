from __future__ import annotations
import os
import sqlite3
import threading
from typing import List, Tuple
from .models import VectorMap, GeoKind
_INIT_LOCK = threading.Lock()

SCHEMA_BASE = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  path   TEXT NOT NULL,
  page_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
  doc_id TEXT NOT NULL,
  page_number INTEGER NOT NULL,
  width REAL, height REAL, rotation INTEGER,
  PRIMARY KEY (doc_id, page_number)
);
CREATE TABLE IF NOT EXISTS geometry (
  doc_id TEXT NOT NULL,
  page_number INTEGER NOT NULL,
  kind INTEGER NOT NULL,
  x0 REAL, y0 REAL, x1 REAL, y1 REAL,
  wkb BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_geom_doc_page ON geometry(doc_id, page_number);
CREATE INDEX IF NOT EXISTS idx_geom_bbox ON geometry(x0, y0, x1, y1);

CREATE TABLE IF NOT EXISTS text_rows (
  doc_id TEXT NOT NULL,
  page_number INTEGER NOT NULL,
  text TEXT NOT NULL,
  bbox TEXT,
  font TEXT, size REAL
);

-- meta must ALWAYS exist before we write flags
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(
  doc_id, page_number, text, bbox, font, size, content=''
);
CREATE TRIGGER IF NOT EXISTS text_fts_ai AFTER INSERT ON text_rows BEGIN
  INSERT INTO text_fts (doc_id, page_number, text, bbox, font, size)
  VALUES (new.doc_id, new.page_number, new.text, new.bbox, new.font, new.size);
END;
CREATE TRIGGER IF NOT EXISTS text_fts_ad AFTER DELETE ON text_rows BEGIN
  INSERT INTO text_fts (text_fts, rowid, doc_id, page_number, text, bbox, font, size)
  VALUES ('delete', old.rowid, old.doc_id, old.page_number, old.text, old.bbox, old.font, old.size);
END;
"""

def _ensure_dir(db_path: str):
    d = os.path.dirname(os.path.abspath(db_path)) or "."
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def open_db(db_path: str) -> sqlite3.Connection:
    _ensure_dir(db_path)
    # allow use across Streamlit threads; we set a busy timeout too
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)  # autocommit
    conn.execute("PRAGMA busy_timeout=5000;")
    # Serialize first-time schema init under a lock to avoid races
    with _INIT_LOCK:
        conn.executescript(SCHEMA_BASE)
        # Probe FTS5 support (don’t crash if absent)
        fts_enabled = "0"
        try:
            conn.executescript(SCHEMA_FTS)
            fts_enabled = "1"
        except sqlite3.OperationalError as e:
            if "fts5" not in str(e).lower():
                raise
        # Defensive: ensure meta exists *again* before we write the flag
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        # Upsert the flag (works on all SQLite builds)
        try:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('fts_enabled', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (fts_enabled,),
            )
        except sqlite3.OperationalError:
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('fts_enabled', ?)", (fts_enabled,))
    return conn

def upsert_vectormap(conn: sqlite3.Connection, vm: VectorMap) -> None:
    """Idempotently store a VectorMap into SQLite."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id, path, page_count) VALUES (?,?,?)",
            (vm.meta.doc_id, vm.meta.path, vm.meta.page_count),
        )
        # Clear any previous content for this doc_id
        conn.execute("DELETE FROM pages WHERE doc_id=?", (vm.meta.doc_id,))
        conn.execute("DELETE FROM geometry WHERE doc_id=?", (vm.meta.doc_id,))
        conn.execute("DELETE FROM text_rows WHERE doc_id=?", (vm.meta.doc_id,))

        # Insert pages + payload
        for pg in vm.pages:
            conn.execute(
                "INSERT INTO pages(doc_id,page_number,width,height,rotation) VALUES (?,?,?,?,?)",
                (vm.meta.doc_id, pg.page_number, pg.width, pg.height, pg.rotation),
            )

            # Geometry rows
            if pg.geoms:
                g_rows = []
                for g in pg.geoms:
                    x0, y0, x1, y1 = g.bbox
                    g_rows.append(
                        (vm.meta.doc_id, pg.page_number, int(g.kind.value), x0, y0, x1, y1, sqlite3.Binary(g.wkb))
                    )
                conn.executemany(
                    "INSERT INTO geometry(doc_id,page_number,kind,x0,y0,x1,y1,wkb) VALUES (?,?,?,?,?,?,?,?)",
                    g_rows,
                )

            # Text rows (mirrored to FTS via triggers if FTS5 is enabled)
            if pg.texts:
                t_rows = []
                for t in pg.texts:
                    x0, y0, x1, y1 = t.bbox
                    bbox_json = f"[{x0},{y0},{x1},{y1}]"
                    t_rows.append((vm.meta.doc_id, pg.page_number, t.text, bbox_json, t.font, t.size))
                conn.executemany(
                    "INSERT INTO text_rows(doc_id,page_number,text,bbox,font,size) VALUES (?,?,?,?,?,?)",
                    t_rows,
                )

def list_documents(conn: sqlite3.Connection):
    cur = conn.execute("SELECT doc_id, path, page_count FROM documents ORDER BY rowid DESC")
    return cur.fetchall()
