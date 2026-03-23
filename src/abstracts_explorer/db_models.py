"""
Database Models
===============

This module defines SQLAlchemy ORM models for the database tables.
These models support both SQLite and PostgreSQL backends.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Float,
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
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

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
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EmbeddingsMetadata."""
        return f"<EmbeddingsMetadata(id={self.id}, model='{self.embedding_model}')>"


class ClusteringCache(Base):
    """
    Clustering cache model.

    Stores cached clustering results with parameters and metadata.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    embedding_model : str
        Name of the embedding model used.
    reduction_method : str
        Dimensionality reduction method used (e.g., 'pca', 'tsne').
    n_components : int
        Number of dimensions after reduction.
    clustering_method : str
        Clustering algorithm used (e.g., 'kmeans', 'dbscan').
    n_clusters : int, optional
        Number of clusters (for kmeans/agglomerative).
    clustering_params : str
        JSON string of additional clustering parameters.
    results_json : str
        JSON string containing full clustering results.
    created_at : datetime
        Timestamp when cache was created.
    """

    __tablename__ = "clustering_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model = Column(String, nullable=False, index=True)
    reduction_method = Column(String, nullable=False)
    n_components = Column(Integer, nullable=False)
    clustering_method = Column(String, nullable=False, index=True)
    n_clusters = Column(Integer, nullable=True)
    clustering_params = Column(Text, nullable=True)
    results_json = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ClusteringCache."""
        return f"<ClusteringCache(id={self.id}, method='{self.clustering_method}', n_clusters={self.n_clusters})>"


class ValidationData(Base):
    """
    Validation data model.

    Stores anonymized user-donated data about interesting papers
    for validation and service improvement purposes.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    paper_uid : str
        Paper UID reference (anonymized - no direct user identification).
    priority : int
        User-assigned priority/rating (1-5).
    search_term : str, optional
        Search term or context associated with this paper.
    donated_at : datetime
        Timestamp when data was donated.
    """

    __tablename__ = "validation_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_uid = Column(String(16), nullable=False, index=True)
    priority = Column(Integer, nullable=False)
    search_term = Column(String, nullable=True)
    donated_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ValidationData."""
        return f"<ValidationData(id={self.id}, paper_uid='{self.paper_uid}', priority={self.priority})>"


class EvalQAPair(Base):
    """
    Evaluation query/answer pair.

    Stores queries and their expected answers for automatic evaluation of the
    RAG system. Supports multi-turn conversations via ``conversation_id`` and
    ``turn_number``.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    conversation_id : str
        Groups related queries in a conversation. All turns in the same
        conversation share this ID.
    turn_number : int
        Position within the conversation (0 = initial query, 1+ = follow-ups).
    query : str
        The user query text.
    expected_answer : str
        The expected/reference answer.
    tool_name : str, optional
        The MCP tool expected to be invoked for this query.
    verified : int
        Verification status: 0 = unverified, 1 = verified/approved,
        -1 = rejected/deleted.
    source_info : str, optional
        JSON string with metadata about how the pair was generated
        (e.g. paper UIDs used, generation model).
    created_at : datetime
        Timestamp when the pair was created.
    updated_at : datetime
        Timestamp when the pair was last modified.
    """

    __tablename__ = "eval_qa_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, nullable=False, index=True)
    turn_number = Column(Integer, nullable=False, default=0)
    query = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    tool_name = Column(String, nullable=True, index=True)
    verified = Column(Integer, nullable=False, default=0)
    source_info = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EvalQAPair."""
        return (
            f"<EvalQAPair(id={self.id}, conv='{self.conversation_id}', "
            f"turn={self.turn_number}, tool='{self.tool_name}')>"
        )


class EvalResult(Base):
    """
    Evaluation run result for a single Q/A pair.

    Stores the actual output from the RAG system when evaluated against a
    stored :class:`EvalQAPair`, together with scoring metrics.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    run_id : str
        Identifier grouping results from the same evaluation run.
    qa_pair_id : int
        ID of the :class:`EvalQAPair` that was evaluated.
    actual_answer : str, optional
        The answer produced by the RAG system.
    actual_tool_name : str, optional
        The MCP tool actually invoked by the RAG system.
    answer_score : float, optional
        LLM-judged quality score (1–5 scale).
    tool_correct : int, optional
        Whether the correct tool was used (1 = yes, 0 = no).
    latency_ms : int, optional
        Wall-clock time for the query in milliseconds.
    error : str, optional
        Error message if the query failed.
    judge_reasoning : str, optional
        The LLM judge's reasoning for the assigned score.
    created_at : datetime
        Timestamp when the result was recorded.
    """

    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    qa_pair_id = Column(Integer, nullable=False, index=True)
    actual_answer = Column(Text, nullable=True)
    actual_tool_name = Column(String, nullable=True)
    answer_score = Column(Float, nullable=True)
    tool_correct = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    judge_reasoning = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EvalResult."""
        return (
            f"<EvalResult(id={self.id}, run='{self.run_id}', "
            f"qa_pair={self.qa_pair_id}, score={self.answer_score})>"
        )
