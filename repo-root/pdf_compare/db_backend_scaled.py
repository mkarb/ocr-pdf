"""
Enhanced database backend with read/write splitting and connection pooling.
Supports PostgreSQL read replicas for horizontal scaling.
"""

from __future__ import annotations
import os
import random
from typing import List, Tuple, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .db_backend import DatabaseBackend


class ScaledDatabaseBackend(DatabaseBackend):
    """
    Enhanced database backend with read replica support.

    Features:
    - Read/write splitting
    - Connection pooling via PgBouncer or direct
    - Automatic read replica load balancing
    - Fallback to primary if replicas unavailable
    """

    def __init__(
        self,
        database_url: str,
        read_replica_urls: Optional[List[str]] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_pre_ping: bool = True
    ):
        """
        Initialize scaled database backend.

        Args:
            database_url: Primary database URL (read/write)
            read_replica_urls: List of read replica URLs (optional)
            pool_size: Connection pool size per engine
            max_overflow: Max overflow connections
            pool_pre_ping: Enable connection health checks
        """
        self.database_url = database_url
        self.read_replica_urls = read_replica_urls or []
        self.is_sqlite = database_url.startswith("sqlite")
        self.is_postgres = database_url.startswith("postgresql")

        # Primary engine (read/write)
        if self.is_sqlite:
            # SQLite doesn't benefit from connection pooling the same way
            from sqlalchemy.pool import StaticPool
            self.write_engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
            # Enable WAL mode
            with self.write_engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA foreign_keys=ON"))
                conn.commit()
        else:
            # PostgreSQL with connection pooling
            self.write_engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=pool_pre_ping,
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False
            )

        # Read replica engines
        self.read_engines = []
        for replica_url in self.read_replica_urls:
            if replica_url and replica_url != database_url:
                engine = create_engine(
                    replica_url,
                    poolclass=QueuePool,
                    pool_size=pool_size // 2,  # Smaller pool for replicas
                    max_overflow=max_overflow // 2,
                    pool_pre_ping=pool_pre_ping,
                    pool_recycle=3600,
                    echo=False
                )
                self.read_engines.append(engine)

        # Use write engine as default
        self.engine = self.write_engine

        # Detect PostgreSQL features
        self.postgres_supports_websearch = False
        if self.is_postgres:
            with self.write_engine.connect() as conn:
                try:
                    conn.execute(text("SELECT websearch_to_tsquery('english', 'test')"))
                    self.postgres_supports_websearch = True
                except Exception:
                    self.postgres_supports_websearch = False

        # Session factories
        self.WriteSession = sessionmaker(bind=self.write_engine)
        self.ReadSession = sessionmaker(
            bind=self.read_engines[0] if self.read_engines else self.write_engine
        )
        self.SessionLocal = self.WriteSession  # Default for compatibility

        # Initialize schema on primary
        self._init_schema()

    def get_read_engine(self):
        """
        Get a read engine with load balancing.
        Falls back to primary if no replicas available.
        """
        if not self.read_engines:
            return self.write_engine

        # Random selection for simple load balancing
        return random.choice(self.read_engines)

    @contextmanager
    def read_session(self):
        """Context manager for read-only operations."""
        engine = self.get_read_engine()
        SessionFactory = sessionmaker(bind=engine)
        session = SessionFactory()
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def write_session(self):
        """Context manager for write operations."""
        session = self.WriteSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Override read methods to use read replicas
    def list_documents(self) -> List[Tuple[str, str, int]]:
        """List all documents using read replica."""
        with self.read_session() as session:
            from .db_models import Document
            docs = session.query(Document).order_by(Document.doc_id.desc()).all()
            return [(d.doc_id, d.path, d.page_count) for d in docs]

    def search_text(
        self,
        query: str,
        doc_id: Optional[str] = None,
        page: Optional[int] = None,
        limit: int = 100
    ) -> List[Tuple]:
        """Full-text search using read replica."""
        with self.read_session() as session:
            from .db_models import TextRow, Meta

            # Check if FTS is enabled
            meta = session.get(Meta, "fts_enabled")
            fts_enabled = meta and meta.value == "1"

            if self.is_sqlite and fts_enabled:
                # SQLite FTS5
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
                # PostgreSQL FTS
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
                # Fallback LIKE search
                from sqlalchemy import select
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
        """Load geometry WKB data using read replica."""
        with self.read_session() as session:
            from .db_models import GeometryRow
            geoms = session.query(GeometryRow.wkb).filter(
                GeometryRow.doc_id == doc_id,
                GeometryRow.page_number == page
            ).all()
            return [g[0] for g in geoms]

    def load_page_texts(self, doc_id: str, page: int) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Load text data using read replica."""
        with self.read_session() as session:
            from .db_models import TextRow
            import json

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

    def get_document_text_with_coords(self, doc_id: str) -> List[Tuple[int, str, Tuple[float, float, float, float], str]]:
        """Get text with coordinates using read replica."""
        with self.read_session() as session:
            from .db_models import TextRow
            import json

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

    def close(self):
        """Close all database connections."""
        self.write_engine.dispose()
        for engine in self.read_engines:
            engine.dispose()


def create_scaled_backend(
    primary_url: Optional[str] = None,
    replica_urls: Optional[List[str]] = None
) -> ScaledDatabaseBackend:
    """
    Factory function to create a scaled database backend.

    Args:
        primary_url: Primary database URL (from env if not provided)
        replica_urls: List of replica URLs (from env if not provided)

    Environment variables:
        DATABASE_URL: Primary database URL
        DATABASE_READ_URL_1: First read replica
        DATABASE_READ_URL_2: Second read replica
        (etc.)
    """
    if primary_url is None:
        primary_url = os.getenv("DATABASE_URL", "sqlite:///vectormap.sqlite")

    if replica_urls is None:
        replica_urls = []
        i = 1
        while True:
            replica_url = os.getenv(f"DATABASE_READ_URL_{i}")
            if not replica_url:
                break
            replica_urls.append(replica_url)
            i += 1

    return ScaledDatabaseBackend(primary_url, replica_urls)
