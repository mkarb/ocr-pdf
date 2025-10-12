"""
Database backend abstraction layer.
Supports both SQLite and PostgreSQL via SQLAlchemy.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from sqlalchemy import create_engine, text, select, delete
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, NullPool

from .db_models import Base, Document, Page, GeometryRow, TextRow, Meta
from .models import VectorMap, GeoKind


class DatabaseBackend:
    """Unified database backend supporting SQLite and PostgreSQL."""

    def __init__(self, database_url: str):
        """
        Initialize database backend.

        Args:
            database_url: Database connection string
                - SQLite: "sqlite:///path/to/file.db"
                - PostgreSQL: "postgresql://user:pass@host:port/dbname"
        """
        self.database_url = database_url
        self.is_sqlite = database_url.startswith("sqlite")
        self.is_postgres = database_url.startswith("postgresql")

        # Configure engine based on database type
        if self.is_sqlite:
            # SQLite configuration
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
            # Enable WAL mode for better concurrency
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA foreign_keys=ON"))
                conn.commit()
        else:
            # PostgreSQL configuration
            self.engine = create_engine(
                database_url,
                poolclass=NullPool,
                echo=False
            )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Create database schema if it doesn't exist."""
        Base.metadata.create_all(self.engine)

        # Set FTS enabled flag for SQLite
        with self.SessionLocal() as session:
            if self.is_sqlite:
                # Check if FTS5 is available
                try:
                    session.execute(text("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(
                            doc_id, page_number, text, bbox, font, size, content=''
                        )
                    """))
                    session.execute(text("""
                        CREATE TRIGGER IF NOT EXISTS text_fts_ai AFTER INSERT ON text_rows BEGIN
                            INSERT INTO text_fts (doc_id, page_number, text, bbox, font, size)
                            VALUES (new.doc_id, new.page_number, new.text, new.bbox, new.font, new.size);
                        END
                    """))
                    session.execute(text("""
                        CREATE TRIGGER IF NOT EXISTS text_fts_ad AFTER DELETE ON text_rows BEGIN
                            INSERT INTO text_fts (text_fts, rowid, doc_id, page_number, text, bbox, font, size)
                            VALUES ('delete', old.rowid, old.doc_id, old.page_number, old.text, old.bbox, old.font, old.size);
                        END
                    """))
                    fts_enabled = "1"
                except Exception:
                    fts_enabled = "0"
            else:
                # PostgreSQL uses native full-text search
                fts_enabled = "1"
                # Create GIN index for full-text search
                try:
                    session.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_text_fts
                        ON text_rows USING gin(to_tsvector('english', text))
                    """))
                except Exception:
                    pass

            # Upsert meta flag
            meta = session.get(Meta, "fts_enabled")
            if meta:
                meta.value = fts_enabled
            else:
                session.add(Meta(key="fts_enabled", value=fts_enabled))

            session.commit()

    def upsert_vectormap(self, vm: VectorMap) -> None:
        """Store a VectorMap into the database."""
        with self.SessionLocal() as session:
            # Upsert document
            doc = session.get(Document, vm.meta.doc_id)
            if doc:
                doc.path = vm.meta.path
                doc.page_count = vm.meta.page_count
            else:
                doc = Document(
                    doc_id=vm.meta.doc_id,
                    path=vm.meta.path,
                    page_count=vm.meta.page_count
                )
                session.add(doc)

            # Delete existing content for this doc_id
            session.execute(delete(Page).where(Page.doc_id == vm.meta.doc_id))
            session.execute(delete(GeometryRow).where(GeometryRow.doc_id == vm.meta.doc_id))
            session.execute(delete(TextRow).where(TextRow.doc_id == vm.meta.doc_id))

            # Insert pages + payload
            for pg in vm.pages:
                # Insert page metadata
                session.add(Page(
                    doc_id=vm.meta.doc_id,
                    page_number=pg.page_number,
                    width=pg.width,
                    height=pg.height,
                    rotation=pg.rotation
                ))

                # Bulk insert geometry rows
                if pg.geoms:
                    geom_objs = []
                    for g in pg.geoms:
                        x0, y0, x1, y1 = g.bbox
                        geom_objs.append(GeometryRow(
                            doc_id=vm.meta.doc_id,
                            page_number=pg.page_number,
                            kind=int(g.kind.value),
                            x0=x0, y0=y0, x1=x1, y1=y1,
                            wkb=g.wkb
                        ))
                    session.bulk_save_objects(geom_objs)

                # Bulk insert text rows
                if pg.texts:
                    text_objs = []
                    for t in pg.texts:
                        x0, y0, x1, y1 = t.bbox
                        bbox_json = f"[{x0},{y0},{x1},{y1}]"
                        text_objs.append(TextRow(
                            doc_id=vm.meta.doc_id,
                            page_number=pg.page_number,
                            text=t.text,
                            bbox=bbox_json,
                            font=t.font,
                            size=t.size,
                            source="native"
                        ))
                    session.bulk_save_objects(text_objs)

            session.commit()

    def list_documents(self) -> List[Tuple[str, str, int]]:
        """List all documents in the database."""
        with self.SessionLocal() as session:
            docs = session.query(Document).order_by(Document.doc_id.desc()).all()
            return [(d.doc_id, d.path, d.page_count) for d in docs]

    def search_text(
        self,
        query: str,
        doc_id: Optional[str] = None,
        page: Optional[int] = None,
        limit: int = 100
    ) -> List[Tuple]:
        """
        Full-text search across text content.

        Returns: List of (doc_id, page_number, text, bbox, font, size)
        """
        with self.SessionLocal() as session:
            # Check if FTS is enabled
            meta = session.get(Meta, "fts_enabled")
            fts_enabled = meta and meta.value == "1"

            if self.is_sqlite and fts_enabled:
                # Use FTS5 for SQLite
                sql = text("""
                    SELECT doc_id, page_number, text, bbox, font, size
                    FROM text_fts
                    WHERE text_fts MATCH :query
                """)
                params = {"query": query}

                if doc_id:
                    sql = text(str(sql) + " AND doc_id = :doc_id")
                    params["doc_id"] = doc_id
                if page:
                    sql = text(str(sql) + " AND page_number = :page")
                    params["page"] = page

                sql = text(str(sql) + f" LIMIT {limit}")
                result = session.execute(sql, params)

            elif self.is_postgres:
                # Use PostgreSQL full-text search
                sql = text("""
                    SELECT doc_id, page_number, text, bbox, font, size
                    FROM text_rows
                    WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)
                """)
                params = {"query": query}

                if doc_id:
                    sql = text(str(sql) + " AND doc_id = :doc_id")
                    params["doc_id"] = doc_id
                if page:
                    sql = text(str(sql) + " AND page_number = :page")
                    params["page"] = page

                sql = text(str(sql) + f" LIMIT {limit}")
                result = session.execute(sql, params)

            else:
                # Fallback to LIKE search
                query_like = query.replace("*", "%")
                stmt = select(
                    TextRow.doc_id,
                    TextRow.page_number,
                    TextRow.text,
                    TextRow.bbox,
                    TextRow.font,
                    TextRow.size
                ).where(TextRow.text.like(f"%{query_like}%"))

                if doc_id:
                    stmt = stmt.where(TextRow.doc_id == doc_id)
                if page:
                    stmt = stmt.where(TextRow.page_number == page)

                stmt = stmt.limit(limit)
                result = session.execute(stmt)

            return result.fetchall()

    def load_page_geoms(self, doc_id: str, page: int) -> List[bytes]:
        """Load geometry WKB data for a page."""
        with self.SessionLocal() as session:
            geoms = session.query(GeometryRow.wkb).filter(
                GeometryRow.doc_id == doc_id,
                GeometryRow.page_number == page
            ).all()
            return [g[0] for g in geoms]

    def load_page_texts(self, doc_id: str, page: int) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Load text data for a page."""
        with self.SessionLocal() as session:
            texts = session.query(TextRow.text, TextRow.bbox).filter(
                TextRow.doc_id == doc_id,
                TextRow.page_number == page
            ).all()

            result = []
            for text, bbox_str in texts:
                try:
                    bbox_data = json.loads(bbox_str)
                    x0, y0, x1, y1 = map(float, bbox_data)
                except Exception:
                    x0, y0, x1, y1 = map(float, bbox_str.strip("[]").split(","))
                result.append((text, (x0, y0, x1, y1)))

            return result

    def close(self):
        """Close database connections."""
        self.engine.dispose()


def create_backend(database_url: str = "sqlite:///vectormap.sqlite") -> DatabaseBackend:
    """
    Factory function to create a database backend.

    Args:
        database_url: Database connection string
            - SQLite: "sqlite:///path/to/file.db" (default)
            - PostgreSQL: "postgresql://user:pass@host:port/dbname"
    """
    return DatabaseBackend(database_url)
