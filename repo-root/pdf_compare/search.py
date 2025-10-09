import sqlite3
from typing import Optional, Tuple, List

BBox = Tuple[float,float,float,float]

def _fts_enabled(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM meta WHERE key='fts_enabled'").fetchone()
    return bool(row and row[0] == '1')

def search_text(conn: sqlite3.Connection, q: str, doc_id: Optional[str]=None, page: Optional[int]=None, limit=100):
    params: List[object] = []
    if _fts_enabled(conn):
        sql = "SELECT doc_id, page_number, text, bbox, font, size FROM text_fts WHERE text_fts MATCH ?"
        params.append(q)
        if doc_id:
            sql += " AND doc_id = ?"; params.append(doc_id)
        if page:
            sql += " AND page_number = ?"; params.append(page)
        sql += " LIMIT ?"; params.append(limit)
        return conn.execute(sql, params).fetchall()
    else:
        # basic LIKE fallback (supports % and _; treat * as %)
        q_like = q.replace("*", "%")
        sql = "SELECT doc_id, page_number, text, bbox, font, size FROM text_rows WHERE text LIKE ?"
        params = [q_like]
        if doc_id:
            sql += " AND doc_id = ?"; params.append(doc_id)
        if page:
            sql += " AND page_number = ?"; params.append(page)
        sql += " LIMIT ?"; params.append(limit)
        return conn.execute(sql, params).fetchall()

def search_geometry_bbox(conn: sqlite3.Connection, bbox: BBox, doc_id: Optional[str]=None, page: Optional[int]=None, limit=1000):
    x0,y0,x1,y1 = bbox
    sql = """
      SELECT doc_id, page_number, kind, x0,y0,x1,y1, wkb
      FROM geometry
      WHERE NOT (x1 < ? OR x0 > ? OR y1 < ? OR y0 > ?)
    """
    params: List[object] = [x0, x1, y0, y1]
    if doc_id:
        sql += " AND doc_id = ?"; params.append(doc_id)
    if page:
        sql += " AND page_number = ?"; params.append(page)
    sql += " LIMIT ?"; params.append(limit)
    return conn.execute(sql, params).fetchall()
