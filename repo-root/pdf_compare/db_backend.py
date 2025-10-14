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
        self.postgres_supports_websearch = False

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
            # Detect websearch_to_tsquery availability for richer search syntax
            with self.engine.connect() as conn:
                try:
                    conn.execute(text("SELECT websearch_to_tsquery('english', 'test')"))
                    self.postgres_supports_websearch = True
                except Exception:
                    self.postgres_supports_websearch = False

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

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its associated data from the database.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document was deleted, False if not found
        """
        with self.SessionLocal() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return False

            # Delete document (CASCADE will handle related rows)
            session.delete(doc)
            session.commit()
            return True

    def delete_all_documents(self) -> int:
        """
        Delete all documents from the database.

        Returns:
            Number of documents deleted
        """
        with self.SessionLocal() as session:
            count = session.query(Document).count()
            session.execute(delete(Document))
            session.commit()
            return count

    def get_document_text_with_coords(self, doc_id: str) -> List[Tuple[int, str, Tuple[float, float, float, float], str]]:
        """
        Get all text with coordinates for creating searchable PDFs.

        Args:
            doc_id: Document ID

        Returns:
            List of (page_number, text, (x0,y0,x1,y1), source) tuples
        """
        with self.SessionLocal() as session:
            texts = session.query(
                TextRow.page_number,
                TextRow.text,
                TextRow.bbox,
                TextRow.source
            ).filter(
                TextRow.doc_id == doc_id
            ).order_by(TextRow.page_number, TextRow.id).all()

            result = []
            for page_num, text, bbox_str, source in texts:
                try:
                    bbox_data = json.loads(bbox_str)
                    x0, y0, x1, y1 = map(float, bbox_data)
                except Exception:
                    x0, y0, x1, y1 = map(float, bbox_str.strip("[]").split(","))
                result.append((page_num, text, (x0, y0, x1, y1), source or "native"))

            return result

    def export_document_text(self, doc_id: str, format: str = "txt") -> str:
        """
        Export all text content from a document for debugging.

        Args:
            doc_id: Document ID to export
            format: Output format ("txt" or "json")

        Returns:
            Formatted text content
        """
        with self.SessionLocal() as session:
            # Get document info
            doc = session.get(Document, doc_id)
            if not doc:
                return f"Error: Document {doc_id} not found"

            # Get all text rows ordered by page and position
            texts = session.query(TextRow).filter(
                TextRow.doc_id == doc_id
            ).order_by(TextRow.page_number, TextRow.id).all()

            if format == "json":
                import json
                data = {
                    "doc_id": doc_id,
                    "path": doc.path,
                    "page_count": doc.page_count,
                    "total_text_items": len(texts),
                    "pages": {}
                }
                for text_row in texts:
                    page_key = f"page_{text_row.page_number}"
                    if page_key not in data["pages"]:
                        data["pages"][page_key] = []
                    data["pages"][page_key].append({
                        "text": text_row.text,
                        "bbox": text_row.bbox,
                        "font": text_row.font,
                        "size": text_row.size,
                        "source": text_row.source
                    })
                return json.dumps(data, indent=2)
            else:  # txt format
                output = []
                output.append(f"Document: {doc_id}")
                output.append(f"Path: {doc.path}")
                output.append(f"Total Pages: {doc.page_count}")
                output.append(f"Total Text Items: {len(texts)}")
                output.append("=" * 80)
                output.append("")

                current_page = None
                for text_row in texts:
                    if text_row.page_number != current_page:
                        current_page = text_row.page_number
                        output.append(f"\n{'='*80}")
                        output.append(f"PAGE {current_page}")
                        output.append(f"{'='*80}\n")

                    source_tag = f"[{text_row.source.upper()}]" if text_row.source else "[NATIVE]"
                    font_info = f" (font: {text_row.font}, size: {text_row.size:.1f})" if text_row.font else ""
                    output.append(f"{source_tag}{font_info}: {text_row.text}")

                output.append(f"\n{'='*80}")
                output.append(f"END OF DOCUMENT - Total {len(texts)} text items across {doc.page_count} pages")
                return "\n".join(output)

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
                tsquery_fn = "websearch_to_tsquery" if self.postgres_supports_websearch else "plainto_tsquery"
                sql = text(f"""
                    SELECT doc_id, page_number, text, bbox, font, size
                    FROM text_rows
                    WHERE to_tsvector('english', text) @@ {tsquery_fn}('english', :query)
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
