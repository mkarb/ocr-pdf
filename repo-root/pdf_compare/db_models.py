"""
SQLAlchemy models for PDF comparison database.
Supports both SQLite and PostgreSQL backends.
"""

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, Index, LargeBinary
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON
from geoalchemy2 import Geometry

Base = declarative_base()


class Document(Base):
    """Document metadata table."""
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    path = Column(Text, nullable=False)
    page_count = Column(Integer, nullable=False)

    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    geometries = relationship("GeometryRow", back_populates="document", cascade="all, delete-orphan")
    text_rows = relationship("TextRow", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    """Page metadata table."""
    __tablename__ = "pages"

    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True)
    page_number = Column(Integer, primary_key=True)
    width = Column(Float)
    height = Column(Float)
    rotation = Column(Integer)

    # Relationships
    document = relationship("Document", back_populates="pages")


class GeometryRow(Base):
    """Geometry (vector) data table with spatial indexing."""
    __tablename__ = "pdf_geometry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    kind = Column(Integer, nullable=False)  # GeoKind enum value
    x0 = Column(Float)
    y0 = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    wkb = Column(LargeBinary, nullable=False)  # Well-Known Binary format

    # Relationships
    document = relationship("Document", back_populates="geometries")

    # Indexes
    __table_args__ = (
        Index("idx_geom_doc_page", "doc_id", "page_number"),
        Index("idx_geom_bbox", "x0", "y0", "x1", "y1"),
    )


class TextRow(Base):
    """Text content table with full-text search support."""
    __tablename__ = "text_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    bbox = Column(Text)  # JSON string for SQLite, JSONB for PostgreSQL
    font = Column(Text)
    size = Column(Float)
    source = Column(String, default="native")  # 'native' or 'ocr'

    # Relationships
    document = relationship("Document", back_populates="text_rows")

    # Indexes
    __table_args__ = (
        Index("idx_text_rows_doc_page", "doc_id", "page_number"),
        Index("idx_text_rows_source", "source"),
    )


class Meta(Base):
    """Metadata key-value store."""
    __tablename__ = "meta"

    key = Column(String, primary_key=True)
    value = Column(Text)
