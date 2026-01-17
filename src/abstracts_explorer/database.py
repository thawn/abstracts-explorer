"""
Database Module
===============

This module provides functionality to load JSON data into a SQL database.
Supports both SQLite and PostgreSQL backends via SQLAlchemy.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, select, func, or_, and_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError, IntegrityError

# Import Pydantic models from plugin framework
from abstracts_explorer.plugin import LightweightPaper

# Import SQLAlchemy models
from abstracts_explorer.db_models import Base, Paper, EmbeddingsMetadata, ClusteringCache

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class DatabaseManager:
    """
    Manager for SQL database operations using SQLAlchemy.

    Supports SQLite and PostgreSQL backends through SQLAlchemy connection URLs.

    Parameters
    ----------
    database_url : str
        SQLAlchemy database URL (e.g., "sqlite:///path/to/db.db" or 
        "postgresql://user:pass@localhost/db").

    Attributes
    ----------
    database_url : str
        SQLAlchemy database URL.
    engine : Engine or None
        SQLAlchemy engine instance.
    SessionLocal : sessionmaker or None
        SQLAlchemy session factory.
    _session : Session or None
        Active database session if connected.

    Examples
    --------
    >>> # SQLite
    >>> from abstracts_explorer.config import get_config
    >>> config = get_config()
    >>> db = DatabaseManager(database_url=config.database_url)
    >>> db.connect()
    >>> db.create_tables()
    >>> db.close()

    >>> # PostgreSQL
    >>> db = DatabaseManager(database_url="postgresql://user:pass@localhost/abstracts")
    >>> db.connect()
    >>> db.create_tables()
    >>> db.close()
    """

    def __init__(self, database_url: str):
        """
        Initialize the DatabaseManager.

        Parameters
        ----------
        database_url : str
            SQLAlchemy database URL.
        """
        if not database_url:
            raise DatabaseError("database_url must be provided")
        
        self.database_url = database_url
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._session: Optional[Session] = None

    def connect(self) -> None:
        """
        Connect to the database.

        Creates the database file if it doesn't exist (SQLite only).

        Raises
        ------
        DatabaseError
            If connection fails.
        """
        try:
            # Create parent directories for SQLite
            if self.database_url.startswith("sqlite:///"):
                db_path_str = self.database_url.replace("sqlite:///", "")
                db_path = Path(db_path_str)
                db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create engine with appropriate settings
            connect_args = {}
            if self.database_url.startswith("sqlite"):
                # SQLite-specific settings
                connect_args = {"check_same_thread": False}

            self.engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                echo=False,  # Set to True for SQL debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

            # Create a session
            self._session = self.SessionLocal()
            
            # Set legacy connection attribute to provide raw database connection for backward compatibility
            # This allows tests to use .connection.cursor() 
            self._raw_connection = self.engine.raw_connection()
            self.connection = self._raw_connection.driver_connection

            logger.info(f"Connected to database: {self._mask_url(self.database_url)}")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {str(e)}") from e

    def _mask_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        if "@" in url and ":" in url:
            parts = url.split("@")
            if len(parts) == 2:
                before_at = parts[0]
                if "://" in before_at:
                    protocol_user = before_at.rsplit(":", 1)[0]
                    return f"{protocol_user}:***@{parts[1]}"
        return url

    def close(self) -> None:
        """
        Close the database connection.

        Does nothing if not connected.
        """
        if self._session:
            self._session.close()
            self._session = None
        if hasattr(self, '_raw_connection') and self._raw_connection:
            self._raw_connection.close()
            self._raw_connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.SessionLocal = None
        self.connection = None
        logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_tables(self) -> None:
        """
        Create database tables for papers and embeddings metadata.

        Creates the following tables:
        - papers: Main table for paper information with lightweight ML4PS schema
        - embeddings_metadata: Metadata about embeddings (model used, creation date)
        - clustering_cache: Cache for clustering results
        
        This method is idempotent - it can be called multiple times without error.
        Tables are only created if they don't already exist.

        Raises
        ------
        DatabaseError
            If table creation fails.
        """
        if not self.engine:
            raise DatabaseError("Not connected to database")

        try:
            # Create tables only if they don't exist (checkfirst=True is the default)
            # This makes the operation idempotent
            Base.metadata.create_all(bind=self.engine, checkfirst=True)
            logger.info("Database tables created successfully")
        except (OperationalError, ProgrammingError, IntegrityError) as e:
            # These exceptions can occur when tables already exist, especially with:
            # - Race conditions in concurrent environments
            # - PostgreSQL's pg_type_typname_nsp_index constraint
            # - SQLite "table already exists" errors
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["already exists", "duplicate", "pg_type_typname_nsp_index"]):
                # Tables already exist - this is fine, just log it
                logger.debug(f"Tables already exist (this is normal): {str(e)}")
                return
            # For other database errors, re-raise with context
            raise DatabaseError(f"Failed to create tables: {str(e)}") from e
        except Exception as e:
            # Catch any other unexpected errors
            raise DatabaseError(f"Failed to create tables: {str(e)}") from e

    def add_paper(self, paper: LightweightPaper) -> Optional[str]:
        """
        Add a single paper to the database.

        Parameters
        ----------
        paper : LightweightPaper
            Validated paper object to insert.

        Returns
        -------
        str or None
            The UID of the inserted paper, or None if paper was skipped (duplicate).

        Raises
        ------
        DatabaseError
            If insertion fails.

        Examples
        --------
        >>> from abstracts_explorer.plugin import LightweightPaper
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     db.create_tables()
        ...     paper = LightweightPaper(
        ...         title="Test Paper",
        ...         authors=["John Doe"],
        ...         abstract="Test abstract",
        ...         session="Session 1",
        ...         poster_position="P1",
        ...         year=2025,
        ...         conference="NeurIPS"
        ...     )
        ...     paper_uid = db.add_paper(paper)
        >>> print(f"Inserted paper with UID: {paper_uid}")
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Extract validated fields from LightweightPaper
            paper_id = paper.original_id if paper.original_id else None
            title = paper.title
            abstract = paper.abstract

            # Handle authors - store as semicolon-separated names
            authors_data = paper.authors
            if isinstance(authors_data, list):
                authors_str = "; ".join(str(author) for author in authors_data)
            else:
                authors_str = str(authors_data) if authors_data else ""

            # Generate UID as hash from title + conference + year
            uid_source = f"{title}:{paper_id}:{paper.conference}:{paper.year}"
            uid = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]

            # Check if paper already exists (by UID)
            existing = self._session.execute(
                select(Paper).where(Paper.uid == uid)
            ).scalar_one_or_none()

            if existing:
                logger.debug(f"Skipping duplicate paper: {title} (uid: {uid})")
                return None

            # Handle keywords (could be list or None)
            keywords_list = paper.keywords
            keywords_str: str
            if isinstance(keywords_list, list):
                keywords_str = ", ".join(str(k) for k in keywords_list)
            elif keywords_list is None:
                keywords_str = ""
            else:
                keywords_str = ""

            # Use paper's original_id if available
            original_id = str(paper.original_id) if paper.original_id else None

            # Create Paper ORM object
            new_paper = Paper(
                uid=uid,
                original_id=original_id,
                title=title,
                authors=authors_str,
                abstract=abstract,
                session=paper.session,
                poster_position=paper.poster_position,
                paper_pdf_url=paper.paper_pdf_url,
                poster_image_url=paper.poster_image_url,
                url=paper.url,
                room_name=paper.room_name,
                keywords=keywords_str,
                starttime=paper.starttime,
                endtime=paper.endtime,
                award=paper.award,
                year=paper.year,
                conference=paper.conference,
            )

            self._session.add(new_paper)
            self._session.commit()

            return uid

        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to add paper: {str(e)}") from e

    def add_papers(self, papers: List[LightweightPaper]) -> int:
        """
        Add multiple papers to the database in a batch.

        Parameters
        ----------
        papers : list of LightweightPaper
            List of validated paper objects to insert.

        Returns
        -------
        int
            Number of papers successfully inserted (excludes duplicates).

        Raises
        ------
        DatabaseError
            If batch insertion fails.

        Examples
        --------
        >>> from abstracts_explorer.plugin import LightweightPaper
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     db.create_tables()
        ...     papers = [
        ...         LightweightPaper(
        ...             title="Paper 1",
        ...             authors=["Author 1"],
        ...             abstract="Abstract 1",
        ...             session="Session 1",
        ...             poster_position="P1",
        ...             year=2025,
        ...             conference="NeurIPS"
        ...         ),
        ...         LightweightPaper(
        ...             title="Paper 2",
        ...             authors=["Author 2"],
        ...             abstract="Abstract 2",
        ...             session="Session 2",
        ...             poster_position="P2",
        ...             year=2025,
        ...             conference="NeurIPS"
        ...         )
        ...     ]
        ...     count = db.add_papers(papers)
        >>> print(f"Inserted {count} papers")
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        inserted_count = 0

        for paper in papers:
            result = self.add_paper(paper)
            if result is not None:
                inserted_count += 1

        logger.info(f"Successfully inserted {inserted_count} of {len(papers)} papers")
        return inserted_count

    def query(self, sql: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.

        Note: This method provides backward compatibility with raw SQL queries.
        For new code, prefer using SQLAlchemy ORM methods.

        Parameters
        ----------
        sql : str
            SQL query to execute (use named parameters like :param1, :param2).
        parameters : tuple, optional
            Query parameters for parameterized queries.

        Returns
        -------
        list of dict
            Query results as list of dictionaries.

        Raises
        ------
        DatabaseError
            If query execution fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     results = db.query("SELECT * FROM papers WHERE session = ?", ("Poster",))
        >>> for row in results:
        ...     print(row['title'])
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Convert ? placeholders to :0, :1, :2 for SQLAlchemy
            # Count the number of ? placeholders
            param_count = sql.count("?")
            
            # Replace ? with numbered parameters
            converted_sql = sql
            for i in range(param_count):
                converted_sql = converted_sql.replace("?", f":param{i}", 1)
            
            # Create parameter dict
            param_dict = {f"param{i}": parameters[i] for i in range(len(parameters))}
            
            # Execute raw SQL using text()
            result = self._session.execute(text(converted_sql), param_dict)
            
            # Convert result to list of dicts
            rows = []
            for row in result:
                # Convert row to dict
                row_dict = dict(row._mapping)
                rows.append(row_dict)
            
            return rows
        except Exception as e:
            raise DatabaseError(f"Query failed: {str(e)}") from e

    def get_paper_count(self) -> int:
        """
        Get the total number of papers in the database.

        Returns
        -------
        int
            Number of papers.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            count = self._session.execute(
                select(func.count()).select_from(Paper)
            ).scalar()
            return count or 0
        except Exception as e:
            raise DatabaseError(f"Failed to count papers: {str(e)}") from e

    def search_papers(
        self,
        keyword: Optional[str] = None,
        session: Optional[str] = None,
        sessions: Optional[List[str]] = None,
        year: Optional[int] = None,
        years: Optional[List[int]] = None,
        conference: Optional[str] = None,
        conferences: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers by various criteria (lightweight schema).

        Parameters
        ----------
        keyword : str, optional
            Keyword to search in title, abstract, or keywords fields.
        session : str, optional
            Single session to filter by (deprecated, use sessions instead).
        sessions : list[str], optional
            List of sessions to filter by (matches ANY).
        year : int, optional
            Single year to filter by (deprecated, use years instead).
        years : list[int], optional
            List of years to filter by (matches ANY).
        conference : str, optional
            Single conference to filter by (deprecated, use conferences instead).
        conferences : list[str], optional
            List of conferences to filter by (matches ANY).
        limit : int, default=100
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Matching papers as dictionaries.

        Raises
        ------
        DatabaseError
            If search fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     papers = db.search_papers(keyword="neural network", limit=10)
        >>> for paper in papers:
        ...     print(paper['title'])

        >>> # Search with multiple sessions
        >>> papers = db.search_papers(sessions=["Session 1", "Session 2"])

        >>> # Search with years
        >>> papers = db.search_papers(years=[2024, 2025])
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Build query conditions
            conditions = []

            if keyword:
                search_pattern = f"%{keyword}%"
                conditions.append(
                    or_(
                        Paper.title.ilike(search_pattern),
                        Paper.abstract.ilike(search_pattern),
                        Paper.keywords.ilike(search_pattern),
                    )
                )

            # Handle sessions (prefer list form, fall back to single)
            session_list = sessions if sessions else ([session] if session else [])
            if session_list:
                conditions.append(Paper.session.in_(session_list))

            # Handle years (prefer list form, fall back to single)
            year_list = years if years else ([year] if year else [])
            if year_list:
                conditions.append(Paper.year.in_(year_list))

            # Handle conferences (prefer list form, fall back to single)
            conference_list = conferences if conferences else ([conference] if conference else [])
            if conference_list:
                conditions.append(Paper.conference.in_(conference_list))

            # Build query
            stmt = select(Paper)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            if limit:
                stmt = stmt.limit(limit)

            # Execute query
            results = self._session.execute(stmt).scalars().all()

            # Convert ORM objects to dicts
            return [self._paper_to_dict(paper) for paper in results]

        except Exception as e:
            raise DatabaseError(f"Search failed: {str(e)}") from e

    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper ORM object to dictionary."""
        return {
            "uid": paper.uid,
            "original_id": paper.original_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "session": paper.session,
            "poster_position": paper.poster_position,
            "paper_pdf_url": paper.paper_pdf_url,
            "poster_image_url": paper.poster_image_url,
            "url": paper.url,
            "room_name": paper.room_name,
            "keywords": paper.keywords,
            "starttime": paper.starttime,
            "endtime": paper.endtime,
            "award": paper.award,
            "year": paper.year,
            "conference": paper.conference,
            "created_at": paper.created_at,
        }

    def search_authors_in_papers(
        self,
        name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for authors by name within the papers' authors field.

        Parameters
        ----------
        name : str, optional
            Name to search for (partial match).
        limit : int, default=100
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Unique authors found in papers with fields: name.

        Raises
        ------
        DatabaseError
            If search fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     authors = db.search_authors_in_papers(name="Huang")
        >>> for author in authors:
        ...     print(author['name'])
        """
        if not name or not self._session:
            return []

        try:
            # Search for authors in the semicolon-separated authors field
            search_pattern = f"%{name}%"
            stmt = (
                select(Paper.authors)
                .where(Paper.authors.ilike(search_pattern))
                .distinct()
                .limit(limit * 10)  # Get more papers to extract unique authors
            )

            results = self._session.execute(stmt).scalars().all()

            # Extract unique author names
            author_names = set()
            for authors_str in results:
                if authors_str:
                    # Split semicolon-separated authors
                    for author in authors_str.split(";"):
                        author = author.strip()
                        if name.lower() in author.lower():
                            author_names.add(author)
                            if len(author_names) >= limit:
                                break
                if len(author_names) >= limit:
                    break

            return [{"name": name} for name in sorted(author_names)[:limit]]
        except Exception as e:
            raise DatabaseError(f"Author search failed: {str(e)}") from e

    def get_author_count(self) -> int:
        """
        Get the approximate number of unique authors in the database.

        Note: This provides an estimate by counting unique author names
        across all papers. The actual count may vary.

        Returns
        -------
        int
            Approximate number of unique authors.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Get all author fields
            stmt = select(Paper.authors).where(
                and_(Paper.authors.isnot(None), Paper.authors != "")
            )
            results = self._session.execute(stmt).scalars().all()

            # Extract unique author names
            author_names = set()
            for authors_str in results:
                if authors_str:
                    for author in authors_str.split(";"):
                        author_names.add(author.strip())

            return len(author_names)
        except Exception as e:
            raise DatabaseError(f"Failed to count authors: {str(e)}") from e

    def get_filter_options(self, year: Optional[int] = None, conference: Optional[str] = None) -> dict:
        """
        Get distinct values for filterable fields (lightweight schema).

        Returns a dictionary with lists of distinct values for session, year,
        and conference fields that can be used to populate filter dropdowns.
        Optionally filters by year and/or conference.

        Parameters
        ----------
        year : int, optional
            Filter results to only show options for this year
        conference : str, optional
            Filter results to only show options for this conference

        Returns
        -------
        dict
            Dictionary with keys 'sessions', 'years', 'conferences' containing
            lists of distinct non-null values sorted alphabetically (or numerically for years).

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     filters = db.get_filter_options()
        >>> print(filters['sessions'])
        ['Session 1', 'Session 2', ...]
        >>> print(filters['years'])
        [2023, 2024, 2025]
        >>> # Get filters for specific year
        >>> filters = db.get_filter_options(year=2025)
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Build WHERE conditions
            conditions = []
            if year is not None:
                conditions.append(Paper.year == year)
            if conference is not None:
                conditions.append(Paper.conference == conference)

            # Get distinct sessions (with filters)
            stmt = select(Paper.session).distinct()
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.where(and_(Paper.session.isnot(None), Paper.session != "")).order_by(Paper.session)

            sessions_result = self._session.execute(stmt).scalars().all()
            sessions = list(sessions_result)

            # Get distinct years (not filtered)
            years_stmt = (
                select(Paper.year)
                .distinct()
                .where(Paper.year.isnot(None))
                .order_by(Paper.year.desc())
            )
            years_result = self._session.execute(years_stmt).scalars().all()
            years = list(years_result)

            # Get distinct conferences (not filtered)
            conferences_stmt = (
                select(Paper.conference)
                .distinct()
                .where(and_(Paper.conference.isnot(None), Paper.conference != ""))
                .order_by(Paper.conference)
            )
            conferences_result = self._session.execute(conferences_stmt).scalars().all()
            conferences = list(conferences_result)

            return {
                "sessions": sessions,
                "years": years,
                "conferences": conferences,
            }
        except Exception as e:
            raise DatabaseError(f"Failed to get filter options: {str(e)}") from e

    def get_embedding_model(self) -> Optional[str]:
        """
        Get the embedding model used for the current embeddings.

        Returns
        -------
        str or None
            Name of the embedding model, or None if not set.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     model = db.get_embedding_model()
        >>> print(model)
        'text-embedding-qwen3-embedding-4b'
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Get the most recent embedding model entry
            stmt = (
                select(EmbeddingsMetadata.embedding_model)
                .order_by(EmbeddingsMetadata.updated_at.desc())
                .limit(1)
            )
            result = self._session.execute(stmt).scalar_one_or_none()
            return result
        except Exception as e:
            raise DatabaseError(f"Failed to get embedding model: {str(e)}") from e

    def set_embedding_model(self, model_name: str) -> None:
        """
        Set the embedding model used for embeddings.

        This stores or updates the embedding model metadata. If a record exists,
        it updates the model and timestamp. Otherwise, it creates a new record.

        Parameters
        ----------
        model_name : str
            Name of the embedding model.

        Raises
        ------
        DatabaseError
            If update fails.

        Examples
        --------
        >>> from abstracts_explorer.config import get_config; config = get_config(); db = DatabaseManager(database_url=config.database_url)
        >>> with db:
        ...     db.set_embedding_model("text-embedding-qwen3-embedding-4b")
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Check if any record exists
            count_stmt = select(func.count()).select_from(EmbeddingsMetadata)
            count = self._session.execute(count_stmt).scalar()

            if count and count > 0:
                # Update the most recent record
                # Get the most recent entry
                latest_stmt = (
                    select(EmbeddingsMetadata)
                    .order_by(EmbeddingsMetadata.updated_at.desc())
                    .limit(1)
                )
                latest = self._session.execute(latest_stmt).scalar_one()
                latest.embedding_model = model_name
                latest.updated_at = datetime.now(timezone.utc)
            else:
                # Insert new record
                new_metadata = EmbeddingsMetadata(embedding_model=model_name)
                self._session.add(new_metadata)

            self._session.commit()
            logger.info(f"Set embedding model to: {model_name}")
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to set embedding model: {str(e)}") from e

    def get_clustering_cache(
        self,
        embedding_model: str,
        reduction_method: str,
        n_components: int,
        clustering_method: str,
        n_clusters: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached clustering results matching the parameters.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        reduction_method : str
            Dimensionality reduction method.
        n_components : int
            Number of components after reduction.
        clustering_method : str
            Clustering algorithm used.
        n_clusters : int, optional
            Number of clusters (for kmeans/agglomerative).

        Returns
        -------
        dict or None
            Cached clustering results as dictionary, or None if not found.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Build query conditions
            stmt = select(ClusteringCache).where(
                and_(
                    ClusteringCache.embedding_model == embedding_model,
                    ClusteringCache.reduction_method == reduction_method,
                    ClusteringCache.n_components == n_components,
                    ClusteringCache.clustering_method == clustering_method,
                )
            )

            # Add n_clusters condition if provided
            if n_clusters is not None:
                stmt = stmt.where(ClusteringCache.n_clusters == n_clusters)

            # Get most recent matching cache
            stmt = stmt.order_by(ClusteringCache.created_at.desc()).limit(1)

            result = self._session.execute(stmt).scalar_one_or_none()

            if result:
                import json
                return json.loads(result.results_json)

            return None

        except Exception as e:
            raise DatabaseError(f"Failed to get clustering cache: {str(e)}") from e

    def save_clustering_cache(
        self,
        embedding_model: str,
        reduction_method: str,
        n_components: int,
        clustering_method: str,
        results: Dict[str, Any],
        n_clusters: Optional[int] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save clustering results to cache.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        reduction_method : str
            Dimensionality reduction method.
        n_components : int
            Number of components after reduction.
        clustering_method : str
            Clustering algorithm used.
        results : dict
            Clustering results to cache.
        n_clusters : int, optional
            Number of clusters (for kmeans/agglomerative).
        clustering_params : dict, optional
            Additional clustering parameters.

        Raises
        ------
        DatabaseError
            If save fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            import json

            # Serialize results and params to JSON
            results_json = json.dumps(results)
            params_json = json.dumps(clustering_params) if clustering_params else None

            # Create new cache entry
            cache_entry = ClusteringCache(
                embedding_model=embedding_model,
                reduction_method=reduction_method,
                n_components=n_components,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                clustering_params=params_json,
                results_json=results_json,
            )

            self._session.add(cache_entry)
            self._session.commit()

            logger.info(
                f"Saved clustering cache: {clustering_method} with {n_clusters} clusters, "
                f"model={embedding_model}, reduction={reduction_method}"
            )

        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to save clustering cache: {str(e)}") from e

    def clear_clustering_cache(self, embedding_model: Optional[str] = None) -> int:
        """
        Clear clustering cache, optionally filtered by embedding model.

        This is useful when embeddings change or cache becomes stale.

        Parameters
        ----------
        embedding_model : str, optional
            If provided, only clear cache for this embedding model.
            If None, clear all cache entries.

        Returns
        -------
        int
            Number of cache entries deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            if embedding_model:
                # Delete only for specific model
                stmt = select(ClusteringCache).where(
                    ClusteringCache.embedding_model == embedding_model
                )
            else:
                # Delete all
                stmt = select(ClusteringCache)

            entries = self._session.execute(stmt).scalars().all()
            count = len(entries)

            for entry in entries:
                self._session.delete(entry)

            self._session.commit()

            if embedding_model:
                logger.info(f"Cleared {count} clustering cache entries for model: {embedding_model}")
            else:
                logger.info(f"Cleared all {count} clustering cache entries")

            return count

        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to clear clustering cache: {str(e)}") from e
