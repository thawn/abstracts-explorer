"""
This module provides functionality to load JSON data into a SQL database.
Supports both SQLite and PostgreSQL backends via SQLAlchemy.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from sqlalchemy import create_engine, event, select, delete, func, or_, and_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError, IntegrityError

# Import Pydantic models from plugin framework
from abstracts_explorer.plugin import (
    LightweightPaper,
    serialize_authors_to_string,
    serialize_keywords_to_string,
    deserialize_authors_from_string,
    deserialize_keywords_from_string,
)

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Base,
    Paper,
    EmbeddingsMetadata,
    ClusteringCache,
    HierarchicalLabelCache,
    ValidationData,
    ChatDonation,
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
        Internal active database session.
    session : Session
        Lazy session property.  The first access connects to the database and
        creates any missing tables.

    Examples
    --------
    >>> # Database configuration comes from config file
    >>> db = DatabaseManager()
    >>> db.get_paper_count()  # connects and creates tables automatically
    0
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
        self._schema_ready = False
        self._raw_connection: Optional[Any] = None
        self.connection: Optional[Any] = None  # Legacy attribute for backward compatibility

    @property
    def session(self) -> Session:
        """
        Return the active database session, connecting lazily if needed.

        The first access calls :meth:`connect` and :meth:`create_tables` so
        that the database schema is available automatically.

        Returns
        -------
        Session
            Active SQLAlchemy session.

        Raises
        ------
        DatabaseError
            If connecting to the database or creating the schema fails.
        """
        if self._session is None:
            self.connect()
        if not self._schema_ready:
            self.create_tables()
        assert self._session is not None
        return self._session

    def connect(self) -> None:
        """
        Connect to the database.

        Does nothing if already connected. Creates the database file if it
        does not exist (SQLite only).

        Raises
        ------
        DatabaseError
            If connection fails.
        """
        if self._session is not None:
            return
        try:
            # Create parent directories for SQLite
            if self.database_url.startswith("sqlite:///"):
                db_path_str = self.database_url.replace("sqlite:///", "")
                db_path = Path(db_path_str)
                db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create engine with appropriate settings
            connect_args = {}
            if self.database_url.startswith("sqlite"):
                # SQLite-specific settings. ``timeout`` is the busy timeout used
                # while another connection holds a write lock.
                connect_args = {"check_same_thread": False, "timeout": 30}

            self.engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                echo=False,  # Set to True for SQL debugging
            )

            if self.database_url.startswith("sqlite"):

                @event.listens_for(self.engine, "connect")
                def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
                    cursor = dbapi_connection.cursor()
                    try:
                        # WAL allows reads while a write transaction is active;
                        # busy_timeout makes writers wait instead of failing
                        # immediately with "database is locked".
                        cursor.execute("PRAGMA journal_mode=WAL")
                        cursor.execute("PRAGMA busy_timeout=30000")
                    finally:
                        cursor.close()

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

            # Create a session
            self._session = self.SessionLocal()
            self._schema_ready = False

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

    def _rollback(self) -> None:
        """Roll back the active session if one exists."""
        if self._session is not None:
            self._session.rollback()

    def close(self) -> None:
        """
        Close the database connection.

        Does nothing if not connected.
        """
        if self._session:
            self._session.close()
            self._session = None
        self._schema_ready = False
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @staticmethod
    def compute_uid(title: str, original_id: Optional[Union[str, int]], conference: str, year: int) -> str:
        """
        Compute a deterministic paper UID from its identifying fields.

        The UID is a 16-character hex string derived from a SHA-256 hash
        of the concatenated title, original_id, conference and year.

        Parameters
        ----------
        title : str
            Paper title.
        original_id : str, int, or None
            Original paper ID from the source (e.g., OpenReview ID).
        conference : str
            Conference name (e.g., "NeurIPS").
        year : int
            Conference year.

        Returns
        -------
        str
            16-character hex UID.

        Examples
        --------
        >>> DatabaseManager.compute_uid("My Paper", "abc123", "NeurIPS", 2025)
        'a1b2c3d4e5f67890'
        """
        uid_source = f"{title}:{original_id}:{conference}:{year}"
        return hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]

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
        if self.engine is None:
            self.connect()
        assert self.engine is not None

        try:
            # Create tables only if they don't exist (checkfirst=True is the default)
            # This makes the operation idempotent
            Base.metadata.create_all(bind=self.engine, checkfirst=True)
            self._schema_ready = True
            logger.debug("Database tables created successfully")
        except (OperationalError, ProgrammingError, IntegrityError) as e:
            # These exceptions can occur when tables already exist, especially with:
            # - Race conditions in concurrent environments
            # - PostgreSQL's pg_type_typname_nsp_index constraint
            # - SQLite "table already exists" errors
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["already exists", "duplicate", "pg_type_typname_nsp_index"]):
                # Tables already exist - this is fine, just log it
                self._schema_ready = True
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
        try:
            # Extract validated fields from LightweightPaper
            paper_id = paper.original_id if paper.original_id else None
            title = paper.title
            abstract = paper.abstract

            # Handle authors - store as semicolon-separated names
            authors_str = serialize_authors_to_string(paper.authors)

            # Generate UID as hash from title + conference + year
            uid = self.compute_uid(title, paper_id, paper.conference, paper.year)

            # Check if paper already exists (by UID)
            existing = self.session.execute(select(Paper).where(Paper.uid == uid)).scalar_one_or_none()

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

            self.session.add(new_paper)
            self.session.commit()

            return uid

        except Exception as e:
            self._rollback()
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
                self.session.add(validation_entry)
                donated_count += 1

            # Commit all changes
            self.session.commit()

            logger.info(f"Successfully donated {donated_count} papers to validation data")
            return donated_count

        except ValueError:
            # Re-raise validation errors without rollback
            raise
        except Exception as e:
            self._rollback()
            logger.error(f"Error donating validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to donate validation data: {e}")

    def donate_chat_transcript(self, rating: str, transcript: List[Dict[str, str]]) -> int:
        """
        Store a donated chat transcript with thumbs up/down feedback.

        This method accepts an anonymized chat transcript and a rating from
        the user and stores it for improving the chat system.

        Parameters
        ----------
        rating : str
            User feedback rating, must be 'up' or 'down'.
        transcript : List[Dict[str, str]]
            List of message dicts, each with 'role' and 'text' keys.

        Returns
        -------
        int
            ID of the stored donation entry.

        Raises
        ------
        ValueError
            If rating is invalid or transcript is empty/malformed.
        DatabaseError
            If database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     transcript = [
        ...         {"role": "user", "text": "What papers discuss transformers?"},
        ...         {"role": "assistant", "text": "Here are some relevant papers..."}
        ...     ]
        ...     donation_id = db.donate_chat_transcript("up", transcript)
        """
        if rating not in ("up", "down"):
            raise ValueError(f"Invalid rating: {rating}. Must be 'up' or 'down'.")

        if not transcript or not isinstance(transcript, list):
            raise ValueError("Transcript must be a non-empty list of messages.")

        for msg in transcript:
            if not isinstance(msg, dict) or "role" not in msg or "text" not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'text' keys.")

        try:
            import json

            donation = ChatDonation(
                rating=rating,
                transcript=json.dumps(transcript),
            )
            self.session.add(donation)
            self.session.commit()

            logger.info(f"Successfully donated chat transcript (rating={rating})")
            return cast(int, donation.id)

        except ValueError:
            raise
        except Exception as e:
            self._rollback()
            logger.error(f"Error donating chat transcript: {e}", exc_info=True)
            raise DatabaseError(f"Failed to donate chat transcript: {e}")

    def get_chat_donations(
        self,
        limit: Optional[int] = None,
        rating: Optional[str] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve donated chat transcripts from the database.

        Parameters
        ----------
        limit : int, optional
            Maximum number of entries to return. Returns all if None.
        rating : str, optional
            Filter by rating ('up' or 'down'). Returns all ratings if None.
        offset : int, optional
            Number of entries to skip for pagination (default: 0).

        Returns
        -------
        list of dict
            List of donation dicts, each containing:
            - id (int): Entry ID.
            - rating (str): 'up' or 'down'.
            - transcript (list): Parsed list of message dicts.
            - donated_at (datetime): Donation timestamp.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     donations = db.get_chat_donations(limit=10, rating='up')
        """
        try:
            import json

            query = self.session.query(ChatDonation)
            if rating is not None:
                query = query.filter(ChatDonation.rating == rating)
            query = query.order_by(ChatDonation.donated_at.desc())
            if offset:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            results = []
            for entry in query.all():
                try:
                    transcript = json.loads(entry.transcript)
                except Exception:
                    transcript = []
                results.append(
                    {
                        "id": entry.id,
                        "rating": entry.rating,
                        "transcript": transcript,
                        "donated_at": entry.donated_at,
                    }
                )
            return results

        except Exception as e:
            logger.error(f"Error retrieving chat donations: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve chat donations: {e}")

    def get_chat_donation_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for donated chat transcripts.

        Returns
        -------
        dict
            Statistics dict containing:
            - total (int): Total number of donations.
            - up (int): Number of thumbs-up donations.
            - down (int): Number of thumbs-down donations.
            - avg_turns (float): Average number of turns per transcript.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     stats = db.get_chat_donation_stats()
        ...     print(f"Total donations: {stats['total']}")
        """
        try:
            import json

            all_entries = self.session.query(ChatDonation).all()
            total = len(all_entries)
            up = sum(1 for e in all_entries if e.rating == "up")
            down = sum(1 for e in all_entries if e.rating == "down")
            turn_counts = []
            for entry in all_entries:
                try:
                    turns = len(json.loads(str(entry.transcript)))
                except Exception:
                    turns = 0
                turn_counts.append(turns)
            avg_turns = sum(turn_counts) / total if total > 0 else 0.0
            return {"total": total, "up": up, "down": down, "avg_turns": avg_turns}

        except Exception as e:
            logger.error(f"Error retrieving chat donation stats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve chat donation stats: {e}")

    def get_validation_data(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve donated validation (interesting-paper) data from the database.

        Parameters
        ----------
        limit : int, optional
            Maximum number of entries to return. Returns all if None.
        offset : int, optional
            Number of entries to skip for pagination (default: 0).

        Returns
        -------
        list of dict
            List of validation data dicts, each containing:
            - id (int): Entry ID.
            - paper_uid (str): Paper UID.
            - priority (int): Priority rating.
            - search_term (str or None): Associated search term.
            - donated_at (datetime): Donation timestamp.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     data = db.get_validation_data(limit=20)
        """
        try:
            query = self.session.query(ValidationData).order_by(ValidationData.donated_at.desc())
            if offset:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            return [
                {
                    "id": entry.id,
                    "paper_uid": entry.paper_uid,
                    "priority": entry.priority,
                    "search_term": entry.search_term,
                    "donated_at": entry.donated_at,
                }
                for entry in query.all()
            ]

        except Exception as e:
            logger.error(f"Error retrieving validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve validation data: {e}")

    def get_validation_data_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for donated validation (interesting-paper) data.

        Returns
        -------
        dict
            Statistics dict containing:
            - total (int): Total number of donated paper ratings.
            - unique_papers (int): Number of distinct paper UIDs.
            - avg_priority (float): Average priority rating.
            - priority_distribution (dict): Count per priority value.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     stats = db.get_validation_data_stats()
        ...     print(f"Total data donations: {stats['total']}")
        """
        try:
            all_entries = self.session.query(ValidationData).all()
            total = len(all_entries)
            unique_papers = len({e.paper_uid for e in all_entries})
            priorities = [int(e.priority) for e in all_entries]
            avg_priority = sum(priorities) / total if total > 0 else 0.0
            distribution: Dict[int, int] = {}
            for p in priorities:
                distribution[p] = distribution.get(p, 0) + 1
            return {
                "total": total,
                "unique_papers": unique_papers,
                "avg_priority": avg_priority,
                "priority_distribution": distribution,
            }

        except Exception as e:
            logger.error(f"Error retrieving validation data stats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve validation data stats: {e}")

    def delete_chat_donations(self, ids: Optional[List[int]] = None) -> int:
        """
        Delete donated chat transcripts.

        Parameters
        ----------
        ids : list of int, optional
            List of donation IDs to delete. If None, deletes all donations.

        Returns
        -------
        int
            Number of donations deleted.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     deleted = db.delete_chat_donations()
        ...     print(f"Deleted {deleted} donations")
        """
        try:
            query = self.session.query(ChatDonation)
            if ids is not None:
                query = query.filter(ChatDonation.id.in_(ids))
            count = query.count()
            query.delete(synchronize_session=False)
            self.session.commit()
            logger.info(f"Deleted {count} chat donation(s)")
            return count

        except Exception as e:
            self._rollback()
            logger.error(f"Error deleting chat donations: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete chat donations: {e}")

    def delete_validation_data(self, ids: Optional[List[int]] = None) -> int:
        """
        Delete donated validation (interesting-paper) data.

        Parameters
        ----------
        ids : list of int, optional
            List of entry IDs to delete. If None, deletes all validation data.

        Returns
        -------
        int
            Number of entries deleted.

        Raises
        ------
        DatabaseError
            If the database operation fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     deleted = db.delete_validation_data()
        ...     print(f"Deleted {deleted} entries")
        """
        try:
            query = self.session.query(ValidationData)
            if ids is not None:
                query = query.filter(ValidationData.id.in_(ids))
            count = query.count()
            query.delete(synchronize_session=False)
            self.session.commit()
            logger.info(f"Deleted {count} validation data entry(ies)")
            return count

        except Exception as e:
            self._rollback()
            logger.error(f"Error deleting validation data: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete validation data: {e}")

    def _query(self, sql: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return results.

        This is a private helper.  Code outside this module must use the
        SQLAlchemy-based public methods instead.

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
        """
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
            result = self.session.execute(text(converted_sql), param_dict)

            # Convert result to list of dicts
            rows = []
            for row in result:
                # Convert row to dict
                row_dict = dict(row._mapping)
                rows.append(row_dict)

            return rows
        except Exception as e:
            raise DatabaseError(f"Query failed: {str(e)}") from e

    def _build_where_conditions(self, where_filters: Optional[Dict[str, Any]]) -> List[Any]:
        """
        Translate a where-filter dictionary into SQLAlchemy conditions.

        Filter keys are Paper column names (``author`` is accepted as an alias
        for ``authors``).  A scalar value adds an equality condition.  A dict
        value adds one or more operator conditions:

        - ``$eq``, ``$ne``
        - ``$in``, ``$nin``
        - ``$contains`` (case-insensitive substring)
        - ``$startswith``, ``$endswith``
        - ``$gt``, ``$gte``, ``$lt``, ``$lte``

        The logical operators ``$and``, ``$or`` and ``$not`` accept nested
        filter dictionaries (``$not`` accepts one dict; the others accept a
        list).
        """
        if not where_filters:
            return []

        columns = {c.name: c for c in Paper.__table__.columns if c.name != "created_at"}
        alias_lower = {k.lower(): v for k, v in self.FIELD_ALIASES.items()}
        conditions: List[Any] = []

        for raw_key, value in where_filters.items():
            key = raw_key.lower()

            if key in ("$and", "$or"):
                if not isinstance(value, list):
                    raise DatabaseError(f"'{raw_key}' expects a list of filter dictionaries")
                nested: List[List[Any]] = [self._build_where_conditions(item) for item in value]
                flattened: List[Any] = [cond for group in nested for cond in group]
                if flattened:
                    conditions.append(and_(*flattened) if key == "$and" else or_(*flattened))
                continue

            if key == "$not":
                if not isinstance(value, dict):
                    raise DatabaseError("'$not' expects a filter dictionary")
                not_conditions: List[Any] = self._build_where_conditions(value)
                if not_conditions:
                    conditions.append(~and_(*not_conditions))
                continue

            field_name = alias_lower.get(key, key)
            column = columns.get(field_name)
            if column is None:
                raise DatabaseError(f"Unknown paper field: {raw_key}")

            if isinstance(value, dict):
                for op, operand in value.items():
                    op_lower = op.lower()
                    if op_lower == "$eq":
                        conditions.append(column == operand)
                    elif op_lower == "$ne":
                        conditions.append(column != operand)
                    elif op_lower == "$in":
                        conditions.append(column.in_(operand))
                    elif op_lower == "$nin":
                        conditions.append(column.notin_(operand))
                    elif op_lower == "$contains":
                        conditions.append(column.ilike(f"%{operand}%"))
                    elif op_lower == "$startswith":
                        conditions.append(column.ilike(f"{operand}%"))
                    elif op_lower == "$endswith":
                        conditions.append(column.ilike(f"%{operand}"))
                    elif op_lower == "$gt":
                        conditions.append(column > operand)
                    elif op_lower == "$gte":
                        conditions.append(column >= operand)
                    elif op_lower == "$lt":
                        conditions.append(column < operand)
                    elif op_lower == "$lte":
                        conditions.append(column <= operand)
                    else:
                        raise DatabaseError(f"Unknown where-filter operator: {op}")
            else:
                conditions.append(column == value)

        return conditions

    def get_paper(
        self,
        where_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve papers matching ``where_filters``.

        This is the single authoritative paper-search method.  All other
        paper retrieval methods delegate to it.

        Parameters
        ----------
        where_filters : dict, optional
            Mapping of Paper column names to filter values.  See
            :meth:`_build_where_conditions` for the supported value forms.
        limit : int, default=100
            Maximum number of results.  ``0`` or ``None`` means no limit.

        Returns
        -------
        list of dict
            Matching papers as dictionaries.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        try:
            stmt = select(Paper)
            conditions = self._build_where_conditions(where_filters)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            if limit:
                stmt = stmt.limit(limit)
            papers = self.session.execute(stmt).scalars().all()
            return [self._paper_to_dict(paper) for paper in papers]
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve papers: {str(e)}") from e

    def get_paper_count(self, where_filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count papers in the database, optionally filtered.

        Parameters
        ----------
        where_filters : dict, optional
            Mapping of Paper column names to filter values.

        Returns
        -------
        int
            Number of matching papers.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            stmt = select(func.count()).select_from(Paper)
            conditions = self._build_where_conditions(where_filters)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            count = self.session.execute(stmt).scalar()
            return count or 0
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to count papers: {str(e)}") from e

    def get_paper_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by its UID.

        Parameters
        ----------
        uid : str
            UID of the paper to retrieve.

        Returns
        -------
        dict or None
            Paper data as a dictionary, or None if not found.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        papers = self.get_paper({"uid": uid}, limit=1)
        return papers[0] if papers else None

    def get_papers_by_uids(self, uids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple papers by their UIDs in a single query.

        Parameters
        ----------
        uids : list of str
            UIDs of the papers to retrieve.

        Returns
        -------
        list of dict
            Paper data dictionaries for all matching UIDs.  The order follows
            the input *uids* list, and unknown UIDs are omitted.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        if not uids:
            return []

        papers = self.get_paper({"uid": {"$in": uids}}, limit=0)
        by_uid = {paper["uid"]: paper for paper in papers}
        return [by_uid[uid] for uid in uids if uid in by_uid]

    def get_paper_by_original_id_or_uid(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by its UID or original_id (whichever matches first).

        Parameters
        ----------
        paper_id : str
            Value to match against the ``uid`` or ``original_id`` column.

        Returns
        -------
        dict or None
            Fully formatted paper data dictionary, or None if not found.

        Raises
        ------
        DatabaseError
            If the database query fails.
        """
        papers = self.get_paper(
            {"$or": [{"uid": paper_id}, {"original_id": paper_id}]},
            limit=1,
        )
        return papers[0] if papers else None

    #: Paper model column names that can be used as ``field:"value"`` filters
    #: in search queries.  Internal columns (``uid``, ``created_at``) are
    #: excluded because they are not meaningful search targets for users.
    SEARCHABLE_FIELDS: set = {c.name for c in Paper.__table__.columns if c.name not in ("uid", "created_at")}

    #: Aliases for field names in search queries.  An alias is transparently
    #: resolved to the canonical column name before applying the filter.
    FIELD_ALIASES: Dict[str, str] = {"author": "authors"}

    def search_papers(
        self,
        keyword: Optional[str] = None,
        field_filters: Optional[Dict[str, str]] = None,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search papers in the SQL database.

        This method handles database search requests with keyword,
        field-specific, session, year and conference filters.  It delegates
        the actual query to :meth:`get_paper`.

        Parameters
        ----------
        keyword : str, optional
            Keyword to search in title, abstract, or keywords fields.
        field_filters : dict of str to str, optional
            Mapping of Paper column names to search values.  Each entry adds
            a case-insensitive substring condition on the corresponding column
            (e.g. ``{"authors": "Smith", "award": "Best Paper"}``).
        sessions : list[str], optional
            List of sessions to filter by (matches ANY).
        years : list[int], optional
            List of years to filter by (matches ANY).
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
        """
        try:
            filter_terms: List[Dict[str, Any]] = []

            if keyword:
                filter_terms.append(
                    {
                        "$or": [
                            {"title": {"$contains": keyword}},
                            {"abstract": {"$contains": keyword}},
                            {"keywords": {"$contains": keyword}},
                        ]
                    }
                )

            if field_filters:
                for field_name, value in field_filters.items():
                    filter_terms.append({field_name: {"$contains": value}})

            if sessions:
                filter_terms.append({"session": {"$in": sessions}})
            if years:
                filter_terms.append({"year": {"$in": years}})
            if conferences:
                filter_terms.append({"conference": {"$in": conferences}})

            if not filter_terms:
                where_filters: Optional[Dict[str, Any]] = None
            elif len(filter_terms) == 1:
                where_filters = filter_terms[0]
            else:
                where_filters = {"$and": filter_terms}

            return self.get_paper(where_filters, limit=limit)
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Search failed: {str(e)}") from e

    @staticmethod
    def parse_field_filters(query: str) -> tuple:
        """
        Parse field-specific filters from a search query string.

        Extracts all ``field:"value"`` patterns from the query where *field*
        is a column name of the :class:`~abstracts_explorer.db_models.Paper`
        model (or a recognised alias; see :attr:`FIELD_ALIASES`).
        Unrecognised field names are left in the query text as-is.

        Parameters
        ----------
        query : str
            The raw search query, e.g.
            ``'authors:"John Smith" award:"Best Paper" transformers'``.

        Returns
        -------
        tuple of (dict, str)
            A tuple ``(field_filters, remaining_query)`` where
            *field_filters* maps canonical column names to their search values
            and *remaining_query* is the query with recognised filters removed.

        Examples
        --------
        >>> DatabaseManager.parse_field_filters('authors:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('author:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('Author:"John Smith" transformers')
        ({'authors': 'John Smith'}, 'transformers')
        >>> DatabaseManager.parse_field_filters('transformers')
        ({}, 'transformers')
        >>> DatabaseManager.parse_field_filters('award:"Best Paper" authors:"Doe"')
        ({'award': 'Best Paper', 'authors': 'Doe'}, '')
        """
        field_filters: Dict[str, str] = {}
        remaining = query

        # Build a lower-cased lookup for aliases and searchable fields so
        # that user input is matched case-insensitively (e.g. Author, AUTHOR).
        alias_lower = {k.lower(): v for k, v in DatabaseManager.FIELD_ALIASES.items()}
        fields_lower = {f.lower(): f for f in DatabaseManager.SEARCHABLE_FIELDS}

        # Iterate in reverse so that removing matched spans does not
        # invalidate the start/end offsets of earlier matches.
        for match in reversed(list(re.finditer(r'(\w+):"([^"]+)"', query))):
            raw_field = match.group(1).lower()
            # Resolve alias (e.g. "author" → "authors") if applicable
            field_name = alias_lower.get(raw_field, raw_field)
            value = match.group(2).strip()
            if field_name in fields_lower:
                field_filters[fields_lower[field_name]] = value
                remaining = remaining[: match.start()] + remaining[match.end() :]

        remaining = " ".join(remaining.split())  # normalise whitespace
        return field_filters, remaining

    def search_papers_keyword(
        self,
        query: str,
        limit: int = 10,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
        *,
        use_embeddings: bool = False,
        distance_threshold: float = 1.1,
        embeddings_manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a user-query search, optionally using semantic embeddings.

        This is the main search entry point for the web UI and CLI.  It parses
        ``field:"value"`` filters from *query*, converts the remaining filter
        arguments into structured where filters, and then delegates either to
        :meth:`search_papers` or :meth:`search_papers_semantic`.

        Parameters
        ----------
        query : str
            Search query text.  May include ``field:"value"`` filters.
        limit : int, optional
            Maximum number of results, by default 10.
        sessions : list of str, optional
            Filter by paper sessions.
        years : list of int, optional
            Filter by publication years.
        conferences : list of str, optional
            Filter by conference names.
        use_embeddings : bool, optional
            Use semantic search instead of SQL keyword search.
        distance_threshold : float, optional
            Maximum embedding distance for semantic results.

        Returns
        -------
        list of dict
            List of paper dictionaries.

        Raises
        ------
        DatabaseError
            If the search fails.
        """
        field_filters, remaining_query = self.parse_field_filters(query)

        if use_embeddings:
            return self.search_papers_semantic(
                remaining_query,
                limit=limit,
                sessions=sessions,
                years=years,
                conferences=conferences,
                distance_threshold=distance_threshold,
                field_filters=field_filters,
                embeddings_manager=embeddings_manager,
            )

        return self.search_papers(
            keyword=remaining_query if remaining_query else None,
            field_filters=field_filters if field_filters else None,
            sessions=sessions,
            years=years,
            conferences=conferences,
            limit=limit,
        )

    @staticmethod
    def _papers_from_chroma_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build paper dictionaries from raw ChromaDB results as a fallback."""
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        if not ids:
            return []

        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []

        papers: List[Dict[str, Any]] = []
        for idx, paper_id in enumerate(ids):
            metadata = dict(metadatas[idx]) if idx < len(metadatas) and metadatas[idx] else {}
            authors = metadata.get("authors", [])
            if isinstance(authors, str):
                authors = [name.strip() for name in authors.split(";") if name.strip()]

            year = metadata.get("year")
            try:
                year = int(year) if year is not None else None
            except (TypeError, ValueError):
                pass

            distance = distances[idx] if idx < len(distances) else None
            paper: Dict[str, Any] = {
                **metadata,
                "uid": paper_id,
                "title": metadata.get("title", ""),
                "authors": authors,
                "abstract": documents[idx] if idx < len(documents) else metadata.get("abstract", ""),
                "session": metadata.get("session", ""),
                "conference": metadata.get("conference", ""),
                "year": year,
                "poster_position": metadata.get("poster_position"),
                "url": metadata.get("url") or metadata.get("paper_url"),
            }
            if distance is not None:
                paper["distance"] = distance
                paper["similarity"] = max(0.0, 1.0 - float(distance))
            papers.append(paper)

        return papers

    def search_papers_semantic(
        self,
        query: str,
        limit: int = 10,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
        distance_threshold: float = 1.1,
        field_filters: Optional[Dict[str, str]] = None,
        where: Optional[Dict[str, Any]] = None,
        embeddings_manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic paper search using embeddings.

        Field filters parsed from the query are applied through the SQL
        database.  The remaining query text is used for the embedding
        similarity search.  Author-name matches are returned first when the
        query text matches an author.

        Parameters
        ----------
        query : str
            Semantic query text (field filters already removed when
            *field_filters* is supplied).
        limit : int, optional
            Maximum number of results, by default 10.
        sessions : list of str, optional
            Filter by paper sessions.
        years : list of int, optional
            Filter by publication years.
        conferences : list of str, optional
            Filter by conference names.
        distance_threshold : float, optional
            Maximum embedding distance for results, by default 1.1.
        field_filters : dict, optional
            Field filters to apply in SQL.  When omitted, filters are parsed
            from *query*.
        where : dict, optional
            Additional ChromaDB metadata filter.  Merged with the
            session/year/conference filters.

        Returns
        -------
        list of dict
            Paper dictionaries ordered by relevance.

        Raises
        ------
        DatabaseError
            If the search fails.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager
        from abstracts_explorer.paper_utils import PaperFormattingError, format_search_results

        if field_filters is None:
            field_filters, query = self.parse_field_filters(query)
        # Query consisting only of field filters: no meaningful embedding.
        if field_filters and not query:
            return self.search_papers(
                field_filters=field_filters,
                sessions=sessions,
                years=years,
                conferences=conferences,
                limit=limit,
            )

        semantic_query = query
        matching_uids: Optional[List[str]] = None

        if field_filters:
            matching_papers = self.search_papers(field_filters=field_filters, limit=0)
            if not matching_papers:
                return []
            matching_uids = [paper["uid"] for paper in matching_papers]
        else:
            author_matches = self.search_papers(
                field_filters={"authors": semantic_query},
                sessions=sessions,
                years=years,
                conferences=conferences,
                limit=limit,
            )
            if author_matches:
                return author_matches

        filter_conditions: List[Dict[str, Any]] = []
        if sessions:
            filter_conditions.append({"session": {"$in": sessions}})
        if years:
            filter_conditions.append({"year": {"$in": [str(y) for y in years]}})
        if conferences:
            filter_conditions.append({"conference": {"$in": conferences}})

        if len(filter_conditions) > 1:
            where_filter: Optional[Dict[str, Any]] = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        else:
            where_filter = None

        if where:
            if where_filter is None:
                where_filter = where
            else:
                where_filter = {"$and": [where, where_filter]}

        owns_manager = embeddings_manager is None
        em = embeddings_manager or EmbeddingsManager()
        try:
            results = em.search_similar(
                semantic_query,
                n_results=limit * 2,
                where=where_filter,
                ids=matching_uids,
            )
            try:
                papers = format_search_results(results, self, include_documents=False)
            except PaperFormattingError:
                papers = []
            if not papers:
                # Keep semantic search usable when embeddings exist but the
                # corresponding SQL rows are unavailable.
                papers = self._papers_from_chroma_results(results)
        finally:
            if owns_manager:
                em.close()

        papers = [paper for paper in papers if paper.get("distance", 0.0) <= distance_threshold]
        return papers[:limit]

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
        where_filters: Dict[str, Any] = {}
        if year is not None:
            where_filters["year"] = year
        if conference is not None:
            where_filters["conference"] = conference

        total_papers = self.get_paper_count(where_filters=where_filters or None)

        return {
            "total_papers": total_papers,
            "year": year,
            "conference": conference,
        }

    @staticmethod
    def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
        """Convert Paper ORM object to dictionary."""
        return {
            "uid": paper.uid,
            "original_id": paper.original_id,
            "title": paper.title,
            "authors": deserialize_authors_from_string(str(paper.authors or "")),
            "abstract": paper.abstract,
            "session": paper.session,
            "poster_position": paper.poster_position,
            "paper_pdf_url": paper.paper_pdf_url,
            "poster_image_url": paper.poster_image_url,
            "url": paper.url,
            "room_name": paper.room_name,
            "keywords": deserialize_keywords_from_string(str(paper.keywords or "")),
            "starttime": str(paper.starttime) if paper.starttime is not None else None,
            "endtime": str(paper.endtime) if paper.endtime is not None else None,
            "award": paper.award,
            "year": paper.year,
            "conference": paper.conference,
            "created_at": str(paper.created_at) if paper.created_at is not None else None,
        }

    def get_conference_years_from_db(self) -> Dict[str, List[int]]:
        """
        Return a mapping of each conference to the years that have papers in the database.

        Years are sorted in descending order (most recent first).

        Returns
        -------
        dict[str, list[int]]
            Mapping of conference name to list of years (descending) that have
            at least one paper in the database.  Returns an empty dict when the
            database is empty.

        Raises
        ------
        DatabaseError
            If the query fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> mapping = db.get_conference_years_from_db()
        >>> print(mapping)
        {'NeurIPS': [2025, 2024], 'ICLR': [2024]}
        """
        try:
            conferences_in_db = self.get_conferences()
            result: Dict[str, List[int]] = {}
            for conf in conferences_in_db:
                years = self.get_years(conference=conf)
                if years:
                    result[conf] = years  # already sorted descending
            return result
        except Exception as e:
            raise DatabaseError(f"Failed to get conference years from DB: {str(e)}") from e

    def confirm_default_conference_year_exists(
        self,
        configured_conference: str,
        configured_year: Optional[int],
    ) -> Dict[str, Any]:
        """
        Confirm that the configured default conference/year exists in the database.

        Unlike the previous fallback logic, this method never guesses an
        alternative.  If the configured default is missing, it returns a
        structured error describing available data and how to fix the problem.

        Parameters
        ----------
        configured_conference : str
            Conference name from the application configuration.
        configured_year : int or None
            Year from the application configuration.  ``0`` or ``None`` means
            that no year is configured; only the conference is validated.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``conference`` : str or None – canonical configured conference,
              or ``None`` when it is unavailable.
            - ``year`` : int or None – confirmed configured year, or ``None``.
            - ``error`` : dict or None – error details when the default is
              unavailable.  Contains ``message``, ``available_conferences``,
              ``available_years`` and ``guidance``.
        """
        db_conference_years = self.get_conference_years_from_db()
        available_conferences = sorted(db_conference_years.keys())

        if not configured_conference:
            return {"conference": None, "year": None, "error": None}

        matched_conf: Optional[str] = None
        for db_conf in available_conferences:
            if db_conf.lower() == configured_conference.lower():
                matched_conf = db_conf
                break

        if matched_conf is None:
            message = f"Configured default conference '{configured_conference}' has no data in the database."
            if not available_conferences:
                message = "The database is empty; the configured default conference cannot be used."
            return {
                "conference": None,
                "year": None,
                "error": {
                    "message": message,
                    "available_conferences": available_conferences,
                    "available_years": [],
                    "guidance": (
                        "Populate the database with 'abstracts-explorer registry download "
                        "--conference <name>' or run 'abstracts-explorer download --conference <name>' "
                        "followed by 'abstracts-explorer create-embeddings', or configure a different "
                        "default with DEFAULT_CONFERENCE and DEFAULT_YEAR."
                    ),
                },
            }

        available_years = db_conference_years.get(matched_conf, [])

        if configured_year:
            if configured_year not in available_years:
                return {
                    "conference": None,
                    "year": None,
                    "error": {
                        "message": (
                            f"Configured default year {configured_year} has no data for "
                            f"conference '{matched_conf}'."
                        ),
                        "available_conferences": available_conferences,
                        "available_years": available_years,
                        "guidance": (
                            "Choose an available year with DEFAULT_YEAR or populate this year with "
                            "'abstracts-explorer registry download --conference <name> --year <year>' "
                            "or 'abstracts-explorer download --conference <name> --year <year>' followed "
                            "by 'abstracts-explorer create-embeddings'."
                        ),
                    },
                }
            return {"conference": matched_conf, "year": configured_year, "error": None}

        return {"conference": matched_conf, "year": None, "error": None}

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
            spelling from the form stored in the database (e.g. ``ml4ps``
            vs. ``ML4PS``).

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
        ...     canonical = db.resolve_conference_name("ml4ps")
        ...     # Returns "ML4PS" if that form is stored in the DB
        """
        # 1. Try case-insensitive match against conferences already in the DB
        try:
            for conf in self.get_conferences():
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
            f"Failed to resolve conference name: {conference}.\n" f"No match found in database or plugins."
        )

    def resolve_conference_for_url(self, url_path: str) -> dict:
        """
        Resolve a URL path segment to a conference, checking data availability.

        Combines plugin-based name resolution with a database data check.
        Returns a result dict describing the outcome:

        - **found with data**: ``{"conference": "<name>", "error": None}``
        - **found without data**: ``{"conference": None, "error": {"message": "...", "available_conferences": [...]}}``
        - **not found**: ``{"conference": None, "error": {"message": "...", "available_conferences": [...]}}``

        Parameters
        ----------
        url_path : str
            URL path segment (e.g. ``"neurips"``, ``"ICLR"``).

        Returns
        -------
        dict
            Dictionary with keys ``"conference"`` (str or None) and
            ``"error"`` (dict or None).

        Examples
        --------
        >>> with DatabaseManager() as db:
        ...     result = db.resolve_conference_for_url("neurips")
        ...     if result["conference"]:
        ...         print(f"Found: {result['conference']}")
        """
        from abstracts_explorer.plugin import get_available_filters, resolve_conference_from_url

        db_conference_years: Dict[str, List[int]] = {}
        try:
            db_conference_years = self.get_conference_years_from_db()
        except Exception:
            pass

        # Also check DB conferences directly (in case they don't have a plugin)
        resolved = resolve_conference_from_url(url_path)
        if resolved is None:
            for conf in db_conference_years:
                if conf.lower() == url_path.lower():
                    resolved = conf
                    break

        if resolved:
            if resolved in db_conference_years:
                return {"conference": resolved, "error": None}
            else:
                available_conferences = sorted(db_conference_years.keys())
                return {
                    "conference": None,
                    "error": {
                        "message": f"No data available for conference '{resolved}'. Please download data first.",
                        "available_conferences": available_conferences,
                    },
                }
        else:
            plugin_conferences = get_available_filters().get("conferences", [])
            available_conferences = sorted(set(list(db_conference_years.keys()) + plugin_conferences))
            return {
                "conference": None,
                "error": {
                    "message": f"Conference '{url_path}' not found.",
                    "available_conferences": available_conferences,
                },
            }

    def get_sessions(self, conference: Optional[str] = None, year: Optional[int] = None) -> List[str]:
        """
        Get distinct session names from the database.

        Parameters
        ----------
        conference : str, optional
            Filter sessions to only those belonging to this conference.
        year : int, optional
            Filter sessions to only those belonging to this year.

        Returns
        -------
        list of str
            Sorted list of distinct non-empty session names.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     sessions = db.get_sessions()
        >>> print(sessions)
        ['Session 1', 'Session 2', ...]
        >>> sessions = db.get_sessions(conference="NeurIPS", year=2025)
        """
        try:
            conditions: list = []
            if conference is not None:
                conditions.append(Paper.conference == conference)
            if year is not None:
                conditions.append(Paper.year == year)

            stmt: Any = select(Paper.session).distinct()
            if conditions:
                stmt = stmt.where(and_(*conditions))
            stmt = stmt.where(and_(Paper.session.isnot(None), Paper.session != "")).order_by(Paper.session)

            return list(self.session.execute(stmt).scalars().all())
        except Exception as e:
            raise DatabaseError(f"Failed to get sessions: {str(e)}") from e

    def get_conferences(self, year: Optional[int] = None) -> List[str]:
        """
        Get distinct conference names from the database.

        Parameters
        ----------
        year : int, optional
            Filter conferences to only those that have papers for this year.

        Returns
        -------
        list of str
            Sorted list of distinct non-empty conference names.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     conferences = db.get_conferences()
        >>> print(conferences)
        ['ICLR', 'NeurIPS']
        >>> conferences = db.get_conferences(year=2025)
        """
        try:
            stmt: Any = (
                select(Paper.conference).distinct().where(and_(Paper.conference.isnot(None), Paper.conference != ""))
            )
            if year is not None:
                stmt = stmt.where(Paper.year == year)
            stmt = stmt.order_by(Paper.conference)

            return list(self.session.execute(stmt).scalars().all())
        except Exception as e:
            raise DatabaseError(f"Failed to get conferences: {str(e)}") from e

    def get_years(self, conference: Optional[str] = None) -> List[int]:
        """
        Get distinct years from the database, sorted descending (most recent first).

        Parameters
        ----------
        conference : str, optional
            Filter years to only those that have papers for this conference.

        Returns
        -------
        list of int
            Sorted list (descending) of distinct years.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     years = db.get_years()
        >>> print(years)
        [2025, 2024, 2023]
        >>> years = db.get_years(conference="NeurIPS")
        """
        try:
            stmt: Any = select(Paper.year).distinct().where(Paper.year.isnot(None))
            if conference is not None:
                stmt = stmt.where(Paper.conference == conference)
            stmt = stmt.order_by(Paper.year.desc())

            return list(self.session.execute(stmt).scalars().all())
        except Exception as e:
            raise DatabaseError(f"Failed to get years: {str(e)}") from e

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
        try:
            # Get the most recent embedding model entry
            stmt: Any = (
                select(EmbeddingsMetadata.embedding_model).order_by(EmbeddingsMetadata.updated_at.desc()).limit(1)
            )
            result = self.session.execute(stmt).scalar_one_or_none()
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
        try:
            # Check if any record exists
            count_stmt = select(func.count()).select_from(EmbeddingsMetadata)
            count = self.session.execute(count_stmt).scalar()

            if count and count > 0:
                # Update the most recent record
                # Get the most recent entry
                latest_stmt = select(EmbeddingsMetadata).order_by(EmbeddingsMetadata.updated_at.desc()).limit(1)
                latest = self.session.execute(latest_stmt).scalar_one()
                latest.embedding_model = model_name
                latest.updated_at = datetime.now(timezone.utc)
            else:
                # Insert new record
                new_metadata = EmbeddingsMetadata(embedding_model=model_name)
                self.session.add(new_metadata)

            self.session.commit()
            logger.info(f"Set embedding model to: {model_name}")
        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to set embedding model: {str(e)}") from e

    def get_clustering_cache(
        self,
        embedding_model: str,
        clustering_method: str,
        n_clusters: Optional[int] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
        reduction_method: Optional[str] = None,
        n_components: Optional[int] = None,
        conference: Optional[str] = None,
        year: Optional[int] = None,
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
            Number of clusters to match.  When ``None`` the query does
            **not** filter by this column.
        clustering_params : dict, optional
            Additional clustering parameters (e.g., distance_threshold, eps).
        reduction_method : str, optional
            Dimensionality reduction method.  When provided, the query
            requires an exact match on this column.
        n_components : int, optional
            Number of components after reduction.  When provided, the query
            requires an exact match on this column.
        conference : str, optional
            Conference name to filter by.  ``None`` matches entries that
            have no conference set.
        year : int, optional
            Conference year to filter by.  ``None`` matches entries that
            have no year set.

        Returns
        -------
        dict or None
            Cached clustering results as dictionary, or None if not found.

        Raises
        ------
        DatabaseError
            If query fails.
        """
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

            # Filter by conference/year columns
            if conference is not None:
                stmt = stmt.where(ClusteringCache.conference == conference)
            else:
                stmt = stmt.where(ClusteringCache.conference.is_(None))

            if year is not None:
                stmt = stmt.where(ClusteringCache.year == year)
            else:
                stmt = stmt.where(ClusteringCache.year.is_(None))

            # Get all matching results (we'll filter by params in Python)
            stmt = stmt.order_by(ClusteringCache.created_at.desc())
            results = self.session.execute(stmt).scalars().all()

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
        conference: Optional[str] = None,
        year: Optional[int] = None,
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
            Number of clusters.  When ``None``, the actual count is
            extracted from ``results["statistics"]["n_clusters"]``.
        clustering_params : dict, optional
            Additional clustering parameters.
        conference : str, optional
            Conference name this entry is scoped to.
        year : int, optional
            Conference year this entry is scoped to.

        Raises
        ------
        DatabaseError
            If save fails.
        """
        try:
            import json

            # Always try to fill n_clusters from the results statistics
            if n_clusters is None:
                stats = results.get("statistics", {})
                actual = stats.get("n_clusters")
                if actual is not None:
                    n_clusters = int(actual)

            # Serialize results and params to JSON
            results_json = json.dumps(results)
            params_json = json.dumps(clustering_params) if clustering_params else None

            # Create new cache entry
            cache_entry = ClusteringCache(
                embedding_model=embedding_model,
                conference=conference,
                year=year,
                reduction_method=reduction_method,
                n_components=n_components,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                clustering_params=params_json,
                results_json=results_json,
            )

            self.session.add(cache_entry)
            self.session.commit()

            logger.info(
                f"Saved clustering cache: {clustering_method} with {n_clusters} clusters, "
                f"model={embedding_model}, reduction={reduction_method}, "
                f"conference={conference}, year={year}"
            )

        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to save clustering cache: {str(e)}") from e

    def delete_papers_by_conference_year(self, conference: str, year: int) -> int:
        """
        Delete all papers for a specific conference and year combination.

        Parameters
        ----------
        conference : str
            Conference name (exact, as stored in the database).
        year : int
            Conference year.

        Returns
        -------
        int
            Number of papers deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            result = self.session.execute(
                delete(Paper).where(and_(Paper.conference == conference, Paper.year == year))
            )
            self.session.commit()
            count = getattr(result, "rowcount", 0) or 0
            logger.info(f"Deleted {count} papers for {conference}/{year}")
            return count

        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to delete papers for {conference}/{year}: {str(e)}") from e

    def delete_clustering_cache_by_conference_year(self, conference: str, year: int) -> int:
        """
        Delete all clustering cache entries for a specific conference and year.

        Parameters
        ----------
        conference : str
            Conference name (exact, as stored in the database).
        year : int
            Conference year.

        Returns
        -------
        int
            Number of cache entries deleted.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            stmt = select(ClusteringCache).where(
                and_(ClusteringCache.conference == conference, ClusteringCache.year == year)
            )
            entries = self.session.execute(stmt).scalars().all()
            count = len(entries)

            for entry in entries:
                self.session.delete(entry)

            self.session.commit()
            logger.info(f"Deleted {count} clustering cache entries for {conference}/{year}")
            return count

        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to delete clustering cache for {conference}/{year}: {str(e)}") from e

    def count_clustering_cache_by_conference_year(self, conference: str, year: int) -> int:
        """
        Count clustering cache entries for a specific conference and year.

        Parameters
        ----------
        conference : str
            Conference name (exact, as stored in the database).
        year : int
            Conference year.

        Returns
        -------
        int
            Number of cache entries found.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        try:
            result = self.session.execute(
                select(func.count())
                .select_from(ClusteringCache)
                .where(and_(ClusteringCache.conference == conference, ClusteringCache.year == year))
            )
            return result.scalar() or 0

        except Exception as e:
            raise DatabaseError(f"Failed to count clustering cache for {conference}/{year}: {str(e)}") from e

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
        try:
            if embedding_model:
                # Delete only for specific model
                stmt = select(ClusteringCache).where(ClusteringCache.embedding_model == embedding_model)
            else:
                # Delete all
                stmt = select(ClusteringCache)

            entries = self.session.execute(stmt).scalars().all()
            count = len(entries)

            for entry in entries:
                self.session.delete(entry)

            self.session.commit()

            if embedding_model:
                logger.info(f"Cleared {count} clustering cache entries for model: {embedding_model}")
            else:
                logger.info(f"Cleared all {count} clustering cache entries")

            return count

        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to clear clustering cache: {str(e)}") from e

    def update_clustering_cache_embedding_model(self, old_model: str, new_model: str) -> int:
        """
        Update the embedding model name in all clustering cache entries.

        Renames every ``ClusteringCache`` row whose ``embedding_model``
        matches *old_model* so that it refers to *new_model* instead.
        This keeps cached clustering results usable after the embedding
        model metadata is renamed (e.g. via
        ``update_embedding_model_metadata.py``).

        Parameters
        ----------
        old_model : str
            Current embedding model name stored in the cache.
        new_model : str
            New embedding model name to write.

        Returns
        -------
        int
            Number of cache entries updated.

        Raises
        ------
        DatabaseError
            If the update fails.

        Examples
        --------
        >>> db = DatabaseManager()
        >>> with db:
        ...     count = db.update_clustering_cache_embedding_model(
        ...         "old-model", "new-model"
        ...     )
        """
        try:
            stmt = select(ClusteringCache).where(ClusteringCache.embedding_model == old_model)
            entries = self.session.execute(stmt).scalars().all()
            count = len(entries)

            for entry in entries:
                setattr(entry, "embedding_model", new_model)

            self.session.commit()
            logger.info(f"Updated {count} clustering cache entries from model {old_model!r} to {new_model!r}")
            return count

        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to update clustering cache embedding model: {str(e)}") from e

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
            result = self.session.execute(stmt).scalars().first()
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
        try:
            import json

            labels_json = json.dumps({str(k): v for k, v in labels.items()})
            entry = HierarchicalLabelCache(
                embedding_model=embedding_model,
                linkage=linkage,
                labels_json=labels_json,
            )
            self.session.add(entry)
            self.session.commit()
            logger.info(
                f"Saved hierarchical label cache: {len(labels)} labels, "
                f"model={embedding_model}, linkage={linkage}"
            )

        except Exception as e:
            self._rollback()
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
        try:
            pair = EvalQAPair(
                conversation_id=conversation_id,
                turn_number=turn_number,
                query=query,
                expected_answer=expected_answer,
                tool_name=tool_name,
                source_info=source_info,
            )
            self.session.add(pair)
            self.session.commit()
            return cast(int, pair.id)
        except Exception as e:
            self._rollback()
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

            rows = self.session.execute(stmt).scalars().all()
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
        try:
            stmt = select(func.count()).select_from(EvalQAPair)
            if verified_only:
                stmt = stmt.where(EvalQAPair.verified == 1)
            return self.session.execute(stmt).scalar() or 0
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
        allowed = {"query", "expected_answer", "tool_name", "verified", "source_info"}
        to_set = {k: v for k, v in fields.items() if k in allowed}
        if not to_set:
            return False

        try:
            pair = self.session.get(EvalQAPair, pair_id)
            if pair is None:
                return False
            for k, v in to_set.items():
                setattr(pair, k, v)
            self.session.commit()
            return True
        except Exception as e:
            self._rollback()
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
        try:
            pair = self.session.get(EvalQAPair, pair_id)
            if pair is None:
                return False
            self.session.delete(pair)
            self.session.commit()
            return True
        except Exception as e:
            self._rollback()
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
        try:
            result = self.session.execute(delete(EvalQAPair).where(EvalQAPair.verified == 1))
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            return count
        except Exception as e:
            self._rollback()
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
        try:
            stmt = delete(EvalResult)
            if run_id is not None:
                stmt = stmt.where(EvalResult.run_id == run_id)
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            return count
        except Exception as e:
            self._rollback()
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
            self.session.add(result)
            self.session.commit()
            return cast(int, result.id)
        except Exception as e:
            self._rollback()
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
        try:
            stmt = select(EvalResult)
            if run_id:
                stmt = stmt.where(EvalResult.run_id == run_id)
            stmt = stmt.order_by(EvalResult.id)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            rows = self.session.execute(stmt).scalars().all()
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
        try:
            stmt: Any = (
                select(EvalResult.run_id).group_by(EvalResult.run_id).order_by(func.min(EvalResult.created_at).asc())
            )
            return [row.run_id for row in self.session.execute(stmt).all()]
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
            stmt: Any = select(func.min(EvalResult.created_at)).where(EvalResult.run_id == run_id)
            run_date = self.session.execute(stmt).scalar()

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
        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_engine = create_engine(f"sqlite:///{output_path}")
            Base.metadata.create_all(export_engine)

            paper_count = 0
            with Session(export_engine) as export_session:
                # Export papers filtered by conference and year
                query = select(Paper).where(and_(Paper.conference == conference, Paper.year == year))
                for paper in self.session.execute(query).scalars():
                    paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
                    export_session.add(Paper(**paper_dict))
                    paper_count += 1

                # Export hierarchical label cache
                for entry in self.session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
                    export_session.add(HierarchicalLabelCache(**entry_dict))

                # Export embeddings metadata
                for meta_entry in self.session.execute(select(EmbeddingsMetadata)).scalars():
                    entry_dict = {c.name: getattr(meta_entry, c.name) for c in EmbeddingsMetadata.__table__.columns}
                    export_session.add(EmbeddingsMetadata(**entry_dict))

                export_session.commit()

            export_engine.dispose()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to export papers: {str(e)}") from e

    def import_papers_from_sqlite(
        self,
        sqlite_path: "Path",
        conference: str,
        year: int,
    ) -> int:
        """
        Import papers for a given conference and year from a SQLite file.

        Existing papers for the given conference/year are **replaced** (not
        merged).  Hierarchical label cache entries that match the conference
        and year are replaced.  Embeddings metadata is validated for
        consistency (the embedding model must match).

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
                    existing_meta = self.session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if existing_meta and normalize_model_name(
                        str(existing_meta.embedding_model)
                    ) != normalize_model_name(str(imported_meta.embedding_model)):
                        raise EmbeddingModelConflictError(
                            str(existing_meta.embedding_model), str(imported_meta.embedding_model)
                        )

                # Delete existing papers for this conference+year
                self.session.execute(delete(Paper).where(and_(Paper.conference == conference, Paper.year == year)))

                # Delete only hierarchical label cache entries whose
                # embedding_model matches one from the imported data
                imported_models_result: List[str] = cast(
                    List[str],
                    source_session.execute(select(HierarchicalLabelCache.embedding_model).distinct()).scalars().all(),
                )
                if imported_models_result:
                    self.session.execute(
                        delete(HierarchicalLabelCache).where(
                            HierarchicalLabelCache.embedding_model.in_(imported_models_result)
                        )
                    )

                self.session.commit()

                # Import papers — use merge() to handle any UID collisions
                # (e.g. the same paper existing under a different conference
                # casing that the DELETE above didn't catch).
                for paper in source_session.execute(select(Paper)).scalars():
                    paper_dict = {c.name: getattr(paper, c.name) for c in Paper.__table__.columns}
                    self.session.merge(Paper(**paper_dict))
                    paper_count += 1

                # Import hierarchical labels
                for entry in source_session.execute(select(HierarchicalLabelCache)).scalars():
                    entry_dict = {c.name: getattr(entry, c.name) for c in HierarchicalLabelCache.__table__.columns}
                    self.session.add(HierarchicalLabelCache(**entry_dict))

                # Import embeddings metadata (only if not already present)
                if imported_meta:
                    existing_meta = self.session.execute(select(EmbeddingsMetadata)).scalars().first()
                    if not existing_meta:
                        meta_dict = {
                            c.name: getattr(imported_meta, c.name) for c in EmbeddingsMetadata.__table__.columns
                        }
                        self.session.add(EmbeddingsMetadata(**meta_dict))

                self.session.commit()

            source_engine.dispose()
            return paper_count

        except DatabaseError:
            raise
        except Exception as e:
            self._rollback()
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

        try:
            entries: List[Dict[str, Any]] = []
            stmt = select(ClusteringCache).where(
                and_(
                    ClusteringCache.conference == conference,
                    ClusteringCache.year == year,
                )
            )
            for entry in self.session.execute(stmt).scalars():
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
        overwrite_embedding_model: Optional[str] = None,
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
        overwrite_embedding_model : str, optional
            When provided, the ``embedding_model`` field of every imported
            entry is replaced with this value.  Use this when importing an
            artifact whose embedding model differs from the locally
            configured one (i.e. when ``--ignore-embedding-model-mismatch``
            was passed) so that :meth:`get_clustering_cache` can find the
            entries using the local model name.

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

        try:
            # Delete existing matching entries using column-based filtering.
            self.session.execute(
                delete(ClusteringCache).where(
                    and_(
                        ClusteringCache.conference == conference,
                        ClusteringCache.year == year,
                    )
                )
            )
            self.session.flush()

            # For PostgreSQL the auto-increment sequence can fall out of sync when rows
            # were previously inserted with explicit primary-key values (e.g. by an older
            # version of import_papers_from_sqlite that included the clustering cache).
            # If other rows for a different conference/year remain in the table and the
            # sequence tries to reuse one of their IDs, the next INSERT will raise
            # UniqueViolation.  Reset the sequence to max(existing id)+1 so that every
            # new row gets a safe ID regardless of sequence history.
            db_url = self.database_url.lower()
            if "postgresql" in db_url or "postgres" in db_url:
                self.session.execute(
                    text(
                        "SELECT setval("
                        "  pg_get_serial_sequence('clustering_cache', 'id'),"
                        "  COALESCE((SELECT MAX(id) FROM clustering_cache), 0) + 1,"
                        "  false"
                        ")"
                    )
                )
                self.session.flush()

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

                # Replace embedding_model when instructed (e.g. mismatch ignored)
                if overwrite_embedding_model is not None:
                    row["embedding_model"] = overwrite_embedding_model

                # Ensure conference/year columns are set from the import scope
                row["conference"] = conference
                row["year"] = year

                # Fill n_clusters from results statistics if not set
                if row.get("n_clusters") is None:
                    results = row.get("results_json")
                    if isinstance(results, str):
                        try:
                            results = _json.loads(results)
                        except (ValueError, TypeError):
                            results = {}
                    if isinstance(results, dict):
                        actual = results.get("statistics", {}).get("n_clusters")
                        if actual is not None:
                            row["n_clusters"] = int(actual)

                self.session.add(ClusteringCache(**row))
                count += 1

            self.session.commit()
            return count

        except DatabaseError:
            raise
        except Exception as e:
            self._rollback()
            raise DatabaseError(f"Failed to import clustering cache: {str(e)}") from e
