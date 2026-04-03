"""
Database Module
===============

This module provides functionality to load JSON data into a SQL database.
Supports both SQLite and PostgreSQL backends via SQLAlchemy.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, select, delete, func, or_, and_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError, IntegrityError

# Import Pydantic models from plugin framework
from abstracts_explorer.plugin import LightweightPaper, serialize_authors_to_string, serialize_keywords_to_string

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Base,
    Paper,
    EmbeddingsMetadata,
    ClusteringCache,
    HierarchicalLabelCache,
    ValidationData,
    EvalQAPair,
    EvalResult,
)

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class EmbeddingModelConflictError(DatabaseError):
    """
    Raised when the embedding model in imported data differs from the local database.

    Attributes
    ----------
    local_model : str
        Embedding model currently in the local database.
    remote_model : str
        Embedding model in the data being imported.
    """

    def __init__(self, local_model: str, remote_model: str) -> None:
        self.local_model = local_model
        self.remote_model = remote_model
        super().__init__(
            f"Embedding model mismatch: local database uses '{local_model}' "
            f"but imported data uses '{remote_model}'. Cannot import data "
            f"created with a different embedding model."
        )


def normalize_model_name(name: str) -> str:
    """
    Normalize an embedding model name for comparison.

    Strips a leading ``alias-`` prefix (case-insensitive) so that, e.g.,
    ``alias-qwen3-embeddings-8b`` is considered identical to
    ``qwen3-embeddings-8b``.

    Parameters
    ----------
    name : str
        Embedding model name.

    Returns
    -------
    str
        Normalized model name.
    """
    return re.sub(r"^alias-", "", name, flags=re.IGNORECASE)


class DatabaseManager:
    """
    Manager for SQL database operations using SQLAlchemy.

    Supports SQLite and PostgreSQL backends through SQLAlchemy connection URLs.
    Database configuration is read from the config file (PAPER_DB variable).

    Attributes
    ----------
    database_url : str
        SQLAlchemy database URL from configuration.
    engine : Engine or None
        SQLAlchemy engine instance.
    SessionLocal : sessionmaker or None
        SQLAlchemy session factory.
    _session : Session or None
        Active database session if connected.

    Examples
    --------
    >>> # Database configuration comes from config file
    >>> db = DatabaseManager()
    >>> db.connect()
    >>> db.create_tables()
    >>> db.close()
    """

    def __init__(self):
        """
        Initialize the DatabaseManager.

        Reads database configuration from the config file.
        """
        from abstracts_explorer.config import get_config

        config = get_config()
        self.database_url = config.database_url

        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._session: Optional[Session] = None
        self.connection = None  # Legacy attribute for backward compatibility (always None now)

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

            logger.debug(f"Connected to database: {self._mask_url(self.database_url)}")
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
        if hasattr(self, "_raw_connection") and self._raw_connection:
            self._raw_connection.close()
            self._raw_connection = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.SessionLocal = None
        self.connection = None
        logger.debug("Database connection closed")

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
            logger.debug("Database tables created successfully")
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
        >>> db = DatabaseManager()
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
            authors_str = serialize_authors_to_string(paper.authors)

            # Generate UID as hash from title + conference + year
            uid_source = f"{title}:{paper_id}:{paper.conference}:{paper.year}"
            uid = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]

            # Check if paper already exists (by UID)
            existing = self._session.execute(select(Paper).where(Paper.uid == uid)).scalar_one_or_none()

            if existing:
                logger.debug(f"Skipping duplicate paper: {title} (uid: {uid})")
                return None

            # Handle keywords (could be list or None)
            keywords_str = serialize_keywords_to_string(paper.keywords) if paper.keywords else ""

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
        >>> db = DatabaseManager()
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
        skipped_count = 0

        for paper in papers:
            try:
                result = self.add_paper(paper)
                if result is not None:
                    inserted_count += 1
            except DatabaseError as e:
                skipped_count += 1
                logger.warning(f"Skipping paper '{paper.title}': {e}")

        if skipped_count:
            logger.warning(f"Skipped {skipped_count} paper(s) due to import errors")
        logger.debug(f"Successfully inserted {inserted_count} of {len(papers)} papers")
        return inserted_count

    def donate_validation_data(self, paper_priorities: Dict[str, Dict[str, Any]]) -> int:
        """
        Store donated paper rating data for validation purposes.

        This method accepts anonymized paper ratings from users and stores them
        in the validation_data table for improving the service.

        Parameters
        ----------
        paper_priorities : Dict[str, Dict[str, Any]]
            Dictionary mapping paper UIDs to priority data.
            Each priority data dict must contain:
            - priority (int): Rating value
            - searchTerm (str, optional): Search term associated with the rating

        Returns
        -------
        int
            Number of papers successfully donated

        Raises
        ------
        ValueError
            If paper_priorities is empty or contains invalid data format
        DatabaseError
            If database operation fails

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     priorities = {
        ...         "abc123": {"priority": 5, "searchTerm": "machine learning"},
        ...         "def456": {"priority": 4, "searchTerm": "deep learning"}
        ...     }
        ...     count = db.donate_validation_data(priorities)
        ...     print(f"Donated {count} papers")
        """
        if not paper_priorities:
            raise ValueError("No data provided")

        if self._session is None:
            raise DatabaseError("Not connected to database")

        try:
            donated_count = 0
            for paper_uid, priority_data in paper_priorities.items():
                # Validate data format
                if not isinstance(priority_data, dict):
                    raise ValueError("Invalid data format. Expected dict with priority and searchTerm")

                priority = priority_data.get("priority", 0)
                search_term = priority_data.get("searchTerm", None)

                # Create validation data entry
                validation_entry = ValidationData(paper_uid=paper_uid, priority=priority, search_term=search_term)
                self._session.add(validation_entry)
                donated_count += 1

            # Commit all changes
            self._session.commit()

            logger.info(f"Successfully donated {donated_count} papers to validation data")
            return donated_count

        except ValueError:
            # Re-raise validation errors without rollback
            raise
        except Exception as e:
            self._session.rollback()
            logger.error(f"Error donating validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to donate validation data: {e}")

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
        >>> db = DatabaseManager()
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
            count = self._session.execute(select(func.count()).select_from(Paper)).scalar()
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
        >>> db = DatabaseManager("neurips.db")
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

    def search_papers_keyword(
        self,
        query: str,
        limit: int = 10,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search with filtering and author parsing.

        This is a convenience method that wraps search_papers and formats
        the results for web API consumption, including author parsing.

        Parameters
        ----------
        query : str
            Keyword to search in title, abstract, or keywords fields
        limit : int, optional
            Maximum number of results, by default 10
        sessions : list of str, optional
            Filter by paper sessions
        years : list of int, optional
            Filter by publication years
        conferences : list of str, optional
            Filter by conference names

        Returns
        -------
        list of dict
            List of paper dictionaries with parsed authors

        Examples
        --------
        >>> papers = db.search_papers_keyword(
        ...     "neural networks",
        ...     limit=5,
        ...     years=[2024, 2025]
        ... )
        """
        # Keyword search in database with multiple filter support
        papers = self.search_papers(
            keyword=query,
            sessions=sessions,
            years=years,
            conferences=conferences,
            limit=limit,
        )

        # Convert to list of dicts for JSON serialization
        papers = [dict(p) for p in papers]

        # Parse authors from comma-separated string for each paper
        for paper in papers:
            if "authors" in paper and paper["authors"]:
                paper["authors"] = [a.strip() for a in paper["authors"].split(";")]
            else:
                paper["authors"] = []

        return papers

    def get_stats(self, year: Optional[int] = None, conference: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database statistics, optionally filtered by year and conference.

        Parameters
        ----------
        year : int, optional
            Filter by specific year
        conference : str, optional
            Filter by specific conference

        Returns
        -------
        dict
            Statistics dictionary with:
            - total_papers: int - Number of papers matching filters
            - year: int or None - Filter year if provided
            - conference: str or None - Filter conference if provided

        Examples
        --------
        >>> stats = db.get_stats()
        >>> print(f"Total papers: {stats['total_papers']}")

        >>> stats_2024 = db.get_stats(year=2024)
        >>> print(f"Papers in 2024: {stats_2024['total_papers']}")
        """
        # Build WHERE clause for filtered count
        conditions: List[str] = []
        parameters: List[Any] = []

        if year is not None:
            conditions.append("year = ?")
            parameters.append(year)

        if conference is not None:
            conditions.append("conference = ?")
            parameters.append(conference)

        if conditions:
            where_clause = " AND ".join(conditions)
            result = self.query(f"SELECT COUNT(*) as count FROM papers WHERE {where_clause}", tuple(parameters))
            total_papers = result[0]["count"] if result else 0
        else:
            total_papers = self.get_paper_count()

        return {
            "total_papers": total_papers,
            "year": year,
            "conference": conference,
        }

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
        >>> db = DatabaseManager()
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
            stmt = select(Paper.authors).where(and_(Paper.authors.isnot(None), Paper.authors != ""))
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

    def get_years_for_conference(self, conference: str) -> List[int]:
        """
        Get distinct years available for a specific conference.

        Parameters
        ----------
        conference : str
            Conference name to query.

        Returns
        -------
        list of int
            Sorted list of distinct years for the conference.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = (
                select(Paper.year)
                .distinct()
                .where(and_(Paper.conference == conference, Paper.year.isnot(None)))
                .order_by(Paper.year)
            )
            return list(self._session.execute(stmt).scalars().all())
        except Exception as e:
            raise DatabaseError(f"Failed to get years for conference: {str(e)}") from e

    def resolve_conference_name(self, conference: str) -> str:
        """
        Resolve a conference name to the canonical form stored in the database.

        Performs a case-insensitive match against conference names already
        present in the database.  If no database match is found, falls back
        to a case-insensitive match against the ``conference_name`` attribute of
        every registered downloader plugin.  If neither lookup succeeds the
        original *conference* string is returned unchanged.

        This is the single authoritative place where conference-name
        normalization must happen.  CLI commands should call this method
        **once** at the entry point of each command and then work with the
        returned canonical name for all subsequent operations.

        Parameters
        ----------
        conference : str
            Conference name as supplied by the caller.  May differ in case or
            spelling from the form stored in the database (e.g. ``ml4ps@neurips``
            vs. ``ML4PS@Neurips``).

        Returns
        -------
        str
            The conference name exactly as it appears in the database (first
            match), or exactly as defined by the first matching plugin, or the
            input string if no match is found.

        Examples
        --------
        >>> with DatabaseManager() as db:
        ...     db.create_tables()
        ...     canonical = db.resolve_conference_name("ml4ps@neurips")
        ...     # Returns "ML4PS@Neurips" if that form is stored in the DB
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        # 1. Try case-insensitive match against conferences already in the DB
        try:
            filters = self.get_filter_options()
            for conf in filters.get("conferences", []):
                if conf.lower() == conference.lower():
                    return conf
        except Exception:
            pass

        # 2. Fall back to plugin conference names
        try:
            from abstracts_explorer.plugins import get_all_plugins

            for plugin in get_all_plugins():
                plugin_conf = getattr(plugin, "conference_name", None)
                if plugin_conf and plugin_conf.lower() == conference.lower():
                    return plugin_conf
        except Exception:
            pass

        raise DatabaseError(
            f"Failed to resolve conference name: {conference}.\n"
            f"No match found in database or plugins. Available conferences in the database: {filters.get('conferences', [])}"
        )

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
        >>> db = DatabaseManager()
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
            years_stmt = select(Paper.year).distinct().where(Paper.year.isnot(None)).order_by(Paper.year.desc())
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
        >>> db = DatabaseManager()
        >>> with db:
        ...     model = db.get_embedding_model()
        >>> print(model)
        'text-embedding-qwen3-embedding-4b'
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Get the most recent embedding model entry
            stmt = select(EmbeddingsMetadata.embedding_model).order_by(EmbeddingsMetadata.updated_at.desc()).limit(1)
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
        >>> db = DatabaseManager()
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
                latest_stmt = select(EmbeddingsMetadata).order_by(EmbeddingsMetadata.updated_at.desc()).limit(1)
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
        clustering_method: str,
        n_clusters: Optional[int] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
        reduction_method: Optional[str] = None,
        n_components: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached clustering results matching the parameters.

        When ``reduction_method`` and ``n_components`` are provided, only
        entries that match exactly (including the reduction method) are
        returned.  When they are omitted (``None``), the reduction method
        is ignored and the most recent entry matching the clustering
        parameters is returned.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        clustering_method : str
            Clustering algorithm used.
        n_clusters : int, optional
            Number of clusters (for kmeans/agglomerative).
        clustering_params : dict, optional
            Additional clustering parameters (e.g., distance_threshold, eps).
        reduction_method : str, optional
            Dimensionality reduction method.  When provided, the query
            requires an exact match on this column.
        n_components : int, optional
            Number of components after reduction.  When provided, the query
            requires an exact match on this column.

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
            import json

            # Build query conditions
            stmt = select(ClusteringCache).where(
                and_(
                    ClusteringCache.embedding_model == embedding_model,
                    ClusteringCache.clustering_method == clustering_method,
                )
            )

            # Optionally filter by reduction method / n_components
            if reduction_method is not None:
                stmt = stmt.where(ClusteringCache.reduction_method == reduction_method)
            if n_components is not None:
                stmt = stmt.where(ClusteringCache.n_components == n_components)

            # Add n_clusters condition if provided
            if n_clusters is not None:
                stmt = stmt.where(ClusteringCache.n_clusters == n_clusters)

            # Get all matching results (we'll filter by params in Python)
            stmt = stmt.order_by(ClusteringCache.created_at.desc())
            results = self._session.execute(stmt).scalars().all()

            if not results:
                return None

            # When no clustering_params are requested, find the first entry whose
            # stored clustering_params is also NULL.
            # Entries that have extra params stored (e.g. distance_threshold) are
            # skipped here because they represent different clustering runs.
            if clustering_params is None:
                for result in results:
                    if result.clustering_params is not None:
                        continue  # entry has extra params – not a match for a no-param query
                    return json.loads(result.results_json)
                return None

            # Filter by clustering_params
            params_json = json.dumps(clustering_params, sort_keys=True)
            for result in results:
                if result.clustering_params is None:
                    continue
                # Compare params (normalize by sorting keys)
                cached_params = json.loads(result.clustering_params)
                cached_params_json = json.dumps(cached_params, sort_keys=True)
                if cached_params_json == params_json:
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

        The full results including visualization coordinates are stored.
        The ``reduction_method`` and ``n_components`` are stored so that
        an exact-match lookup can return cached points directly.  When only
        the reduction method changes, the clustering results are reused and
        only the reduction is re-applied.

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
            Clustering results to cache (full results including points).
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
                stmt = select(ClusteringCache).where(ClusteringCache.embedding_model == embedding_model)
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

    # ------------------------------------------------------------------
    # Hierarchical label cache
    # ------------------------------------------------------------------

    def get_hierarchical_label_cache(
        self,
        embedding_model: str,
        linkage: str = "ward",
    ) -> Optional[Dict[int, str]]:
        """
        Get cached hierarchical labels for agglomerative clustering.

        Hierarchical labels are independent of the number of clusters and
        the distance threshold, so they are reused for all agglomerative
        clustering settings that share the same embedding model and linkage.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        linkage : str, optional
            Agglomerative linkage method (default: ``"ward"``).

        Returns
        -------
        dict or None
            Mapping of ``{node_id: label}`` (integer keys), or ``None`` if
            no entry is found.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            import json

            stmt = (
                select(HierarchicalLabelCache)
                .where(
                    and_(
                        HierarchicalLabelCache.embedding_model == embedding_model,
                        HierarchicalLabelCache.linkage == linkage,
                    )
                )
                .order_by(HierarchicalLabelCache.created_at.desc())
                .limit(1)
            )
            result = self._session.execute(stmt).scalars().first()
            if result is None:
                return None
            raw = json.loads(result.labels_json)
            # JSON keys are always strings – convert back to int
            return {int(k): v for k, v in raw.items()}

        except Exception as e:
            raise DatabaseError(f"Failed to get hierarchical label cache: {str(e)}") from e

    def save_hierarchical_label_cache(
        self,
        embedding_model: str,
        labels: Dict[int, str],
        linkage: str = "ward",
    ) -> None:
        """
        Save hierarchical cluster labels to cache.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        labels : dict
            Mapping of ``{node_id: label}`` to store.
        linkage : str, optional
            Agglomerative linkage method (default: ``"ward"``).

        Raises
        ------
        DatabaseError
            If save fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            import json

            labels_json = json.dumps({str(k): v for k, v in labels.items()})
            entry = HierarchicalLabelCache(
                embedding_model=embedding_model,
                linkage=linkage,
                labels_json=labels_json,
            )
            self._session.add(entry)
            self._session.commit()
            logger.info(
                f"Saved hierarchical label cache: {len(labels)} labels, "
                f"model={embedding_model}, linkage={linkage}"
            )

        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to save hierarchical label cache: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #  Evaluation Q/A pair and result methods                              #
    # ------------------------------------------------------------------ #

    def add_eval_qa_pair(
        self,
        conversation_id: str,
        turn_number: int,
        query: str,
        expected_answer: str,
        tool_name: Optional[str] = None,
        source_info: Optional[str] = None,
    ) -> int:
        """
        Insert a single evaluation Q/A pair.

        Parameters
        ----------
        conversation_id : str
            Identifier grouping turns in a conversation.
        turn_number : int
            Position within the conversation (0 = first).
        query : str
            The user query text.
        expected_answer : str
            The expected/reference answer.
        tool_name : str, optional
            MCP tool expected to be invoked.
        source_info : str, optional
            JSON metadata about how the pair was generated.

        Returns
        -------
        int
            Primary key of the inserted row.

        Raises
        ------
        DatabaseError
            If insertion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            pair = EvalQAPair(
                conversation_id=conversation_id,
                turn_number=turn_number,
                query=query,
                expected_answer=expected_answer,
                tool_name=tool_name,
                source_info=source_info,
            )
            self._session.add(pair)
            self._session.commit()
            return pair.id
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to add eval QA pair: {str(e)}") from e

    def get_eval_qa_pairs(
        self,
        verified_only: bool = False,
        tool_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation Q/A pairs with optional filters.

        Parameters
        ----------
        verified_only : bool
            If ``True``, return only pairs with ``verified == 1``.
        tool_name : str, optional
            Filter by expected MCP tool name.
        conversation_id : str, optional
            Filter by conversation.
        limit : int, optional
            Maximum number of pairs to return.
        offset : int
            Number of rows to skip (for pagination).

        Returns
        -------
        list of dict
            Matching Q/A pairs as dictionaries.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = select(EvalQAPair)
            if verified_only:
                stmt = stmt.where(EvalQAPair.verified == 1)
            if tool_name:
                stmt = stmt.where(EvalQAPair.tool_name == tool_name)
            if conversation_id:
                stmt = stmt.where(EvalQAPair.conversation_id == conversation_id)
            stmt = stmt.order_by(EvalQAPair.conversation_id, EvalQAPair.turn_number)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            rows = self._session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "turn_number": r.turn_number,
                    "query": r.query,
                    "expected_answer": r.expected_answer,
                    "tool_name": r.tool_name,
                    "verified": r.verified,
                    "source_info": r.source_info,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval QA pairs: {str(e)}") from e

    def get_eval_qa_pair_count(self, verified_only: bool = False) -> int:
        """
        Count evaluation Q/A pairs.

        Parameters
        ----------
        verified_only : bool
            If ``True``, count only verified pairs.

        Returns
        -------
        int
            Number of matching pairs.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = select(func.count()).select_from(EvalQAPair)
            if verified_only:
                stmt = stmt.where(EvalQAPair.verified == 1)
            return self._session.execute(stmt).scalar() or 0
        except Exception as e:
            raise DatabaseError(f"Failed to count eval QA pairs: {str(e)}") from e

    def update_eval_qa_pair(self, pair_id: int, **fields) -> bool:
        """
        Update fields on an existing Q/A pair.

        Parameters
        ----------
        pair_id : int
            Primary key of the pair to update.
        **fields
            Keyword arguments mapping column names to new values.
            Supported keys: ``query``, ``expected_answer``, ``tool_name``,
            ``verified``, ``source_info``.

        Returns
        -------
        bool
            ``True`` if a row was updated, ``False`` if the pair was not found.

        Raises
        ------
        DatabaseError
            If update fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        allowed = {"query", "expected_answer", "tool_name", "verified", "source_info"}
        to_set = {k: v for k, v in fields.items() if k in allowed}
        if not to_set:
            return False

        try:
            pair = self._session.get(EvalQAPair, pair_id)
            if pair is None:
                return False
            for k, v in to_set.items():
                setattr(pair, k, v)
            self._session.commit()
            return True
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to update eval QA pair: {str(e)}") from e

    def delete_eval_qa_pair(self, pair_id: int) -> bool:
        """
        Delete an evaluation Q/A pair by ID.

        Parameters
        ----------
        pair_id : int
            Primary key of the pair to delete.

        Returns
        -------
        bool
            ``True`` if a row was deleted, ``False`` if pair was not found.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            pair = self._session.get(EvalQAPair, pair_id)
            if pair is None:
                return False
            self._session.delete(pair)
            self._session.commit()
            return True
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to delete eval QA pair: {str(e)}") from e

    def delete_verified_eval_qa_pairs(self) -> int:
        """
        Delete all verified (accepted) evaluation Q/A pairs.

        Returns
        -------
        int
            Number of pairs deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            result = self._session.execute(delete(EvalQAPair).where(EvalQAPair.verified == 1))
            count = result.rowcount
            self._session.commit()
            return count
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to delete verified eval QA pairs: {str(e)}") from e

    def delete_eval_results(self, run_id: Optional[str] = None) -> int:
        """
        Delete stored evaluation results, optionally filtered to a single run.

        Parameters
        ----------
        run_id : str, optional
            If supplied, only results for this run are deleted.
            If ``None``, **all** stored results are deleted.

        Returns
        -------
        int
            Number of rows deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = delete(EvalResult)
            if run_id is not None:
                stmt = stmt.where(EvalResult.run_id == run_id)
            result = self._session.execute(stmt)
            count = result.rowcount
            self._session.commit()
            return count
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to delete eval results: {str(e)}") from e

    def add_eval_result(
        self,
        run_id: str,
        qa_pair_id: int,
        actual_answer: Optional[str] = None,
        actual_tool_name: Optional[str] = None,
        answer_score: Optional[float] = None,
        tool_correct: Optional[int] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        judge_reasoning: Optional[str] = None,
    ) -> int:
        """
        Insert a single evaluation result.

        Parameters
        ----------
        run_id : str
            Identifier for the evaluation run.
        qa_pair_id : int
            ID of the evaluated Q/A pair.
        actual_answer : str, optional
            Answer produced by the RAG system.
        actual_tool_name : str, optional
            MCP tool actually invoked.
        answer_score : float, optional
            LLM-judged quality score (1–5).
        tool_correct : int, optional
            1 if the correct tool was used, 0 otherwise.
        latency_ms : int, optional
            Query latency in milliseconds.
        error : str, optional
            Error message if the query failed.
        judge_reasoning : str, optional
            LLM judge's reasoning for the score.

        Returns
        -------
        int
            Primary key of the inserted row.

        Raises
        ------
        DatabaseError
            If insertion fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            result = EvalResult(
                run_id=run_id,
                qa_pair_id=qa_pair_id,
                actual_answer=actual_answer,
                actual_tool_name=actual_tool_name,
                answer_score=answer_score,
                tool_correct=tool_correct,
                latency_ms=latency_ms,
                error=error,
                judge_reasoning=judge_reasoning,
            )
            self._session.add(result)
            self._session.commit()
            return result.id
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to add eval result: {str(e)}") from e

    def get_eval_results(
        self,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evaluation results with optional run filter.

        Parameters
        ----------
        run_id : str, optional
            Filter by evaluation run. If ``None``, return results from all runs.
        limit : int, optional
            Maximum number of results to return.
        offset : int
            Number of rows to skip.

        Returns
        -------
        list of dict
            Evaluation results as dictionaries.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = select(EvalResult)
            if run_id:
                stmt = stmt.where(EvalResult.run_id == run_id)
            stmt = stmt.order_by(EvalResult.id)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            rows = self._session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "qa_pair_id": r.qa_pair_id,
                    "actual_answer": r.actual_answer,
                    "actual_tool_name": r.actual_tool_name,
                    "answer_score": r.answer_score,
                    "tool_correct": r.tool_correct,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "judge_reasoning": r.judge_reasoning,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval results: {str(e)}") from e

    def get_eval_run_ids(self) -> List[str]:
        """
        Return distinct evaluation run IDs ordered by run time, oldest first.

        The ordering is determined by the minimum ``created_at`` timestamp of
        all results in each run, so the most recent run appears last.

        Returns
        -------
        list of str
            Distinct run IDs ordered chronologically (oldest to newest).

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            stmt = (
                select(EvalResult.run_id).group_by(EvalResult.run_id).order_by(func.min(EvalResult.created_at).asc())
            )
            return [row.run_id for row in self._session.execute(stmt).all()]
        except Exception as e:
            raise DatabaseError(f"Failed to get eval run IDs: {str(e)}") from e

    def get_eval_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Compute summary statistics for an evaluation run.

        Parameters
        ----------
        run_id : str
            The evaluation run identifier.

        Returns
        -------
        dict
            Dictionary with keys:

            - total : int – number of evaluated pairs
            - avg_score : float or None – mean answer quality score
            - tool_accuracy : float or None – fraction of correct tool selections
            - avg_latency_ms : float or None – mean latency
            - error_count : int – number of queries that produced errors
            - run_date : datetime or None – timestamp of the first result in the run

        Raises
        ------
        DatabaseError
            If query fails.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            results = self.get_eval_results(run_id=run_id)
            if not results:
                return {
                    "total": 0,
                    "avg_score": None,
                    "tool_accuracy": None,
                    "avg_latency_ms": None,
                    "error_count": 0,
                    "run_date": None,
                }

            total = len(results)
            scores = [r["answer_score"] for r in results if r["answer_score"] is not None]
            tool_vals = [r["tool_correct"] for r in results if r["tool_correct"] is not None]
            latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
            errors = sum(1 for r in results if r["error"])

            # Determine the timestamp of the earliest result in this run
            stmt = select(func.min(EvalResult.created_at)).where(EvalResult.run_id == run_id)
            run_date = self._session.execute(stmt).scalar()

            return {
                "total": total,
                "avg_score": (sum(scores) / len(scores)) if scores else None,
                "tool_accuracy": (sum(tool_vals) / len(tool_vals)) if tool_vals else None,
                "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                "error_count": errors,
                "run_date": run_date,
            }
        except Exception as e:
            raise DatabaseError(f"Failed to compute eval run summary: {str(e)}") from e

    # ------------------------------------------------------------------
    # Registry export / import helpers
    # ------------------------------------------------------------------

    def export_papers_to_sqlite(
        self,
        output_path: "Path",
        conference: str,
        year: int,
    ) -> int:
        """
        Export papers for a given conference and year to a standalone SQLite file.

        The exported file includes hierarchical label cache and embeddings
        metadata rows.  Clustering cache is **not** included — it is exported
        separately via :meth:`export_clustering_cache_to_json`.

        Parameters
        ----------
        output_path : Path
            Destination path for the SQLite file.
        conference : str
            Conference name to export.
        year : int
            Year to export.

        Returns
        -------
        int
            Number of papers exported.

        Raises
        ------
        DatabaseError
            If the export fails or no papers are found.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_engine = create_engine(f"sqlite:///{output_path}")
            Base.metadata.create_all(export_engine)

            paper_count = 0
            with Session(export_engine) as export_session:
                # Export papers filtered by conference and year
                query = select(Paper).where(and_(Paper.conference == conference, Paper.year == year))
                for paper in self._session.execute(query).scalars():
                    paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
                    export_session.add(Paper(**paper_dict))
                    paper_count += 1

                # Export hierarchical label cache
                for entry in self._session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
                    export_session.add(HierarchicalLabelCache(**entry_dict))

                # Export embeddings metadata
                for entry in self._session.execute(select(EmbeddingsMetadata)).scalars():
                    entry_dict = {c.name: getattr(entry, c.name) for c in EmbeddingsMetadata.__table__.columns}
                    export_session.add(EmbeddingsMetadata(**entry_dict))

                export_session.commit()

            export_engine.dispose()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to export papers: {str(e)}") from e

    @staticmethod
    def _clustering_cache_matches_conference_year(entry: ClusteringCache, conference: str, year: int) -> bool:
        """
        Check whether a ClusteringCache entry is scoped to a specific conference and year.

        An entry matches if its ``clustering_params`` JSON contains a
        ``conferences`` list that includes *conference* **and** a ``years``
        list that includes *year*.

        Parameters
        ----------
        entry : ClusteringCache
            The cache entry to check.
        conference : str
            Conference name.
        year : int
            Conference year.

        Returns
        -------
        bool
            ``True`` if the entry matches the given conference and year.
        """
        import json as _json

        if not entry.clustering_params:
            return False
        try:
            params = _json.loads(entry.clustering_params)
        except (ValueError, TypeError):
            return False

        conferences = params.get("conferences", [])
        years = params.get("years", [])
        return conference in conferences and year in years

    def import_papers_from_sqlite(
        self,
        sqlite_path: "Path",
        conference: str,
        year: int,
    ) -> int:
        """
        Import papers for a given conference and year from a SQLite file.

        Existing papers for the given conference/year are **replaced** (not
        merged).  Clustering cache and hierarchical label cache entries that
        match the conference and year are replaced.  Embeddings metadata is
        validated for consistency (the embedding model must match).

        Parameters
        ----------
        sqlite_path : Path
            Path to the source SQLite file.
        conference : str
            Conference name being imported.
        year : int
            Year being imported.

        Returns
        -------
        int
            Number of papers imported.

        Raises
        ------
        DatabaseError
            If the import fails or the embedding model is inconsistent.
        """
        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            source_engine = create_engine(
                f"sqlite:///{sqlite_path}",
                connect_args={"check_same_thread": False},
            )

            paper_count = 0
            with Session(source_engine) as source_session:
                # --- Validate EmbeddingsMetadata consistency ---
                imported_meta = source_session.execute(select(EmbeddingsMetadata)).scalars().first()
                if imported_meta:
                    existing_meta = self._session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if existing_meta and normalize_model_name(existing_meta.embedding_model) != normalize_model_name(
                        imported_meta.embedding_model
                    ):
                        raise EmbeddingModelConflictError(
                            existing_meta.embedding_model, imported_meta.embedding_model
                        )

                # Delete existing papers for this conference+year
                self._session.execute(delete(Paper).where(and_(Paper.conference == conference, Paper.year == year)))

                # Delete only hierarchical label cache entries whose
                # embedding_model matches one from the imported data
                imported_models_result = (
                    source_session.execute(select(HierarchicalLabelCache.embedding_model).distinct()).scalars().all()
                )
                if imported_models_result:
                    self._session.execute(
                        delete(HierarchicalLabelCache).where(
                            HierarchicalLabelCache.embedding_model.in_(imported_models_result)
                        )
                    )

                self._session.commit()

                # Import papers — use merge() to handle any UID collisions
                # (e.g. the same paper existing under a different conference
                # casing that the DELETE above didn't catch).
                for paper in source_session.execute(select(Paper)).scalars():
                    paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
                    self._session.merge(Paper(**paper_dict))
                    paper_count += 1

                # Import hierarchical labels
                for entry in source_session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
                    self._session.add(HierarchicalLabelCache(**entry_dict))

                # Import embeddings metadata (only if not already present)
                if imported_meta:
                    existing_meta = self._session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if not existing_meta:
                        meta_dict = {
                            c.name: getattr(imported_meta, c.name) for c in EmbeddingsMetadata.__table__.columns
                        }
                        self._session.add(EmbeddingsMetadata(**meta_dict))

                self._session.commit()

            source_engine.dispose()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to import papers: {str(e)}") from e

    # ------------------------------------------------------------------
    # Clustering cache JSON export / import
    # ------------------------------------------------------------------

    def export_clustering_cache_to_json(
        self,
        conference: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Export clustering cache entries matching *conference* and *year* as JSON.

        Parameters
        ----------
        conference : str
            Conference name to match.
        year : int
            Year to match.

        Returns
        -------
        dict
            A JSON-serialisable dictionary with an ``entries`` list.
            Each entry contains all :class:`ClusteringCache` columns except
            ``id`` (auto-generated on import).

        Raises
        ------
        DatabaseError
            If the export fails.
        """
        import json as _json

        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            entries: List[Dict[str, Any]] = []
            for entry in self._session.execute(select(ClusteringCache)).scalars():
                if self._clustering_cache_matches_conference_year(entry, conference, year):
                    row: Dict[str, Any] = {}
                    for col in ClusteringCache.__table__.columns:
                        if col.name == "id":
                            continue  # skip PK; it will be auto-generated on import
                        val = getattr(entry, col.name)
                        if col.name in ("clustering_params", "results_json") and isinstance(val, str):
                            val = _json.loads(val)
                        elif col.name == "created_at" and val is not None:
                            val = val.isoformat()
                        row[col.name] = val
                    entries.append(row)

            return {"entries": entries}

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to export clustering cache: {str(e)}") from e

    def import_clustering_cache_from_json(
        self,
        data: Dict[str, Any],
        conference: str,
        year: int,
    ) -> int:
        """
        Import clustering cache entries from a JSON dictionary.

        Existing clustering cache entries matching *conference* and *year*
        are deleted before importing the new entries.

        Parameters
        ----------
        data : dict
            Dictionary previously returned by
            :meth:`export_clustering_cache_to_json`.
        conference : str
            Conference name for scoping the delete.
        year : int
            Year for scoping the delete.

        Returns
        -------
        int
            Number of cache entries imported.

        Raises
        ------
        DatabaseError
            If the import fails.
        """
        import json as _json

        if not self._session:
            raise DatabaseError("Not connected to database")

        try:
            # Delete existing matching entries
            for entry in self._session.execute(select(ClusteringCache)).scalars().all():
                if self._clustering_cache_matches_conference_year(entry, conference, year):
                    self._session.delete(entry)

            count = 0
            for item in data.get("entries", []):
                row = dict(item)  # shallow copy

                # Re-serialise parsed JSON fields back to strings for DB storage
                if "clustering_params" in row and row["clustering_params"] is not None:
                    if not isinstance(row["clustering_params"], str):
                        row["clustering_params"] = _json.dumps(row["clustering_params"])

                if "results_json" in row and row["results_json"] is not None:
                    if not isinstance(row["results_json"], str):
                        row["results_json"] = _json.dumps(row["results_json"])

                # Convert ISO-format created_at back to datetime
                if "created_at" in row and isinstance(row["created_at"], str):
                    row["created_at"] = datetime.fromisoformat(row["created_at"])

                # Drop 'id' if present — let the DB auto-generate it
                row.pop("id", None)

                self._session.add(ClusteringCache(**row))
                count += 1

            self._session.commit()
            return count

        except DatabaseError:
            raise
        except Exception as e:
            self._session.rollback()
            raise DatabaseError(f"Failed to import clustering cache: {str(e)}") from e
