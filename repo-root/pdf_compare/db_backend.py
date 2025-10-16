"""
PostgreSQL database backend using SQLAlchemy.
Provides ORM-based storage for PDF vector data with full-text search support.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import json
import os
import random
from contextlib import contextmanager

from sqlalchemy import create_engine, text, select, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from .db_models import Base, Document, Page, GeometryRow, TextRow, Meta
from .models import VectorMap, GeoKind


class DatabaseBackend:
    """PostgreSQL database backend with SQLAlchemy ORM."""

    def __init__(
        self,
        database_url: str,
        *,
        read_replica_urls: Optional[List[str]] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
    ):
        """
        Initialize database backend.

        Args:
            database_url: Database connection string
                - PostgreSQL: "postgresql://user:pass@host:port/dbname"
                - SQLite: "sqlite:///path/to/file.db" (for local dev/testing)
            read_replica_urls: Optional list of PostgreSQL read replicas
            pool_size: Connection pool size for primary engine
            max_overflow: Overflow connections for pool
            pool_pre_ping: Enable SQLAlchemy connection pre-ping

        Raises:
            ValueError: If database_url is not supported
        """
        self.database_url = database_url
        self.read_replica_urls = list(read_replica_urls or [])
        self.is_postgres = database_url.startswith("postgresql")
        self.is_sqlite = database_url.startswith("sqlite")

        if not (self.is_postgres or self.is_sqlite):
            raise ValueError(
                f"Unsupported database URL. Expected PostgreSQL or SQLite, got: {database_url[:32]}..."
            )

        # Primary engine (read/write)
        if self.is_sqlite:
            self.write_engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
            # Enable WAL for better concurrency
            with self.write_engine.connect() as conn:
                try:
                    conn.execute(text("PRAGMA journal_mode=WAL"))
                    conn.execute(text("PRAGMA foreign_keys=ON"))
                    conn.commit()
                except Exception:
                    # Some SQLite builds may not support WAL commits
                    pass
        else:
            self.write_engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=pool_pre_ping,
                pool_recycle=3600,
                echo=False,
            )

        # Backwards compatibility attribute
        self.engine = self.write_engine
        self.postgres_supports_websearch = False

        # Detect websearch_to_tsquery availability for richer search syntax
        if self.is_postgres:
            with self.write_engine.connect() as conn:
                try:
                    conn.execute(text("SELECT websearch_to_tsquery('english', 'test')"))
                    self.postgres_supports_websearch = True
                except Exception:
                    self.postgres_supports_websearch = False

        # Configure read replicas (PostgreSQL only)
        self.read_engines = []
        if self.is_postgres:
            for replica_url in self.read_replica_urls:
                if not replica_url or replica_url == database_url:
                    continue
                engine = create_engine(
                    replica_url,
                    poolclass=QueuePool,
                    pool_size=max(1, pool_size // 2),
                    max_overflow=max(0, max_overflow // 2),
                    pool_pre_ping=pool_pre_ping,
                    pool_recycle=3600,
                    echo=False,
                )
                self.read_engines.append(engine)
        else:
            self.read_engines = []

        # Session factories
        self.SessionLocal = sessionmaker(bind=self.write_engine)
        self._read_sessionmakers = [sessionmaker(bind=engine) for engine in self.read_engines]
        # Always include primary as fallback for reads
        self._read_sessionmakers.append(self.SessionLocal)

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Create database schema if it doesn't exist."""
        Base.metadata.create_all(self.engine)

        # Set up PostgreSQL full-text search
        with self.SessionLocal() as session:
            # Create GIN index for full-text search on text content
            try:
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_text_fts
                    ON text_rows USING gin(to_tsvector('english', text))
                """))
            except Exception:
                # Index may already exist or insufficient permissions
                pass

            # Set FTS enabled flag
            meta = session.get(Meta, "fts_enabled")
            if meta:
                meta.value = "1"
            else:
                session.add(Meta(key="fts_enabled", value="1"))

            session.commit()

    def _get_read_sessionmaker(self):
        """Select a read session factory (load-balanced across replicas)."""
        if not self._read_sessionmakers:
            return self.SessionLocal
        if len(self._read_sessionmakers) == 1:
            return self._read_sessionmakers[0]
        return random.choice(self._read_sessionmakers)

    @contextmanager
    def read_session(self):
        """Context manager that yields a session bound to a read engine."""
        sessionmaker_cls = self._get_read_sessionmaker()
        session = sessionmaker_cls()
        try:
            yield session
        finally:
            session.close()

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
        with self.read_session() as session:
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
        with self.read_session() as session:
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
        with self.read_session() as session:
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
        Full-text search across text content using PostgreSQL full-text search.

        Args:
            query: Search query string (supports web search syntax if available)
            doc_id: Optional document ID to filter results
            page: Optional page number to filter results
            limit: Maximum number of results to return

        Returns:
            List of (doc_id, page_number, text, bbox, font, size) tuples
        """
        with self.read_session() as session:
            if self.is_postgres:
                tsquery_fn = "websearch_to_tsquery" if self.postgres_supports_websearch else "plainto_tsquery"
                clauses = [
                    "SELECT doc_id, page_number, text, bbox, font, size",
                    "FROM text_rows",
                    f"WHERE to_tsvector('english', text) @@ {tsquery_fn}('english', :query)"
                ]
                params = {"query": query, "limit": limit}

                if doc_id:
                    clauses.append("AND doc_id = :doc_id")
                    params["doc_id"] = doc_id
                if page:
                    clauses.append("AND page_number = :page")
                    params["page"] = page

                clauses.append("ORDER BY page_number LIMIT :limit")
                sql = text(" ".join(clauses))
                result = session.execute(sql, params)
            else:
                # Fallback simple LIKE search (SQLite or unsupported engines)
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

                stmt = stmt.order_by(TextRow.page_number).limit(limit)
                result = session.execute(stmt)

            return result.fetchall()

    def load_page_geoms(self, doc_id: str, page: int) -> List[bytes]:
        """Load geometry WKB data for a page."""
        with self.read_session() as session:
            geoms = session.query(GeometryRow.wkb).filter(
                GeometryRow.doc_id == doc_id,
                GeometryRow.page_number == page
            ).all()
            return [g[0] for g in geoms]

    def load_page_texts(self, doc_id: str, page: int) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Load text data for a page."""
        with self.read_session() as session:
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
        if hasattr(self, "write_engine"):
            self.write_engine.dispose()
        for engine in getattr(self, "read_engines", []):
            engine.dispose()


def create_backend(
    database_url: str,
    *,
    read_replica_urls: Optional[List[str]] = None,
    pool_size: int = 20,
    max_overflow: int = 10,
    pool_pre_ping: bool = True,
) -> DatabaseBackend:
    """
    Factory function to create a PostgreSQL database backend.

    Args:
        database_url: PostgreSQL connection string
            Format: "postgresql://user:pass@host:port/dbname"
        read_replica_urls: Optional list of read replica URLs
        pool_size: Connection pool size for primary engine
        max_overflow: Overflow connection count
        pool_pre_ping: Enable SQLAlchemy connection pre-ping

    Returns:
        DatabaseBackend instance

    Raises:
        ValueError: If database_url is not a PostgreSQL URL
    """
    if read_replica_urls is None:
        read_replica_urls = []
        # Support both DATABASE_READ_URL and DATABASE_READ_URL_n env vars
        single_replica = os.getenv("DATABASE_READ_URL")
        if single_replica:
            read_replica_urls.append(single_replica)

        idx = 1
        while True:
            env_key = f"DATABASE_READ_URL_{idx}"
            replica_url = os.getenv(env_key)
            if not replica_url:
                break
            read_replica_urls.append(replica_url)
            idx += 1

    return DatabaseBackend(
        database_url,
        read_replica_urls=read_replica_urls,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
    )
