"""
Database Models
===============

This module defines SQLAlchemy ORM models for the database tables.
These models support both SQLite and PostgreSQL backends.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Paper(Base):
    """
    Paper model representing a research paper.

    This uses the lightweight schema from LightweightPaper model.

    Attributes
    ----------
    uid : str
        Unique identifier (hash-based, primary key).
    original_id : str, optional
        Original ID from the source (e.g., OpenReview ID).
    title : str
        Paper title.
    authors : str, optional
        Semicolon-separated list of author names.
    abstract : str, optional
        Paper abstract.
    session : str, optional
        Conference session name.
    poster_position : str, optional
        Poster position identifier.
    paper_pdf_url : str, optional
        URL to paper PDF.
    poster_image_url : str, optional
        URL to poster image.
    url : str, optional
        General URL for the paper.
    room_name : str, optional
        Room name for presentation.
    keywords : str, optional
        Comma-separated keywords.
    starttime : str, optional
        Start time of presentation.
    endtime : str, optional
        End time of presentation.
    award : str, optional
        Award received (e.g., "Best Paper").
    year : int, optional
        Publication year.
    conference : str, optional
        Conference name (e.g., "NeurIPS", "ICLR").
    created_at : datetime
        Timestamp when record was created.
    """

    __tablename__ = "papers"

    uid = Column(String(16), primary_key=True, index=True)
    original_id = Column(String, nullable=True, index=True)
    title = Column(Text, nullable=False, index=True)
    authors = Column(Text, nullable=True)
    abstract = Column(Text, nullable=True)
    session = Column(String, nullable=True, index=True)
    poster_position = Column(String, nullable=True)
    paper_pdf_url = Column(String, nullable=True)
    poster_image_url = Column(String, nullable=True)
    url = Column(String, nullable=True)
    room_name = Column(String, nullable=True)
    keywords = Column(Text, nullable=True)
    starttime = Column(String, nullable=True)
    endtime = Column(String, nullable=True)
    award = Column(String, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    conference = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, server_default=func.now())

    def __repr__(self) -> str:
        """String representation of Paper."""
        return f"<Paper(uid='{self.uid}', title='{self.title[:50]}...')>"


class EmbeddingsMetadata(Base):
    """
    Embeddings metadata model.

    Tracks which embedding model was used for the vector embeddings.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    embedding_model : str
        Name of the embedding model used.
    created_at : datetime
        Timestamp when record was created.
    updated_at : datetime
        Timestamp when record was last updated.
    """

    __tablename__ = "embeddings_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, server_default=func.now())

    def __repr__(self) -> str:
        """String representation of EmbeddingsMetadata."""
        return f"<EmbeddingsMetadata(id={self.id}, model='{self.embedding_model}')>"
