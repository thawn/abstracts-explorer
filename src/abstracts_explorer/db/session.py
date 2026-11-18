"""
Session and schema lifecycle for DatabaseManager.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError, ProgrammingError, IntegrityError

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Base,
)
from abstracts_explorer.metadata_cache import MetadataCache


from abstracts_explorer.db.errors import DatabaseError

logger = logging.getLogger(__name__)

_SCHEMA_READY_URLS: set[str] = set()
_SCHEMA_LOCK = threading.Lock()
_METADATA_CACHE = MetadataCache()


def mask_database_url(url: str) -> str:
    """Return *url* with any password replaced by ``***``."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return url


def reset_schema_cache() -> None:
    """
    Clear the process-level schema-ready cache.

    This is primarily intended for tests and for explicit schema migrations.
    """
    with _SCHEMA_LOCK:
        _SCHEMA_READY_URLS.clear()


def reset_metadata_cache() -> None:
    """Clear the process-level metadata cache."""
    _METADATA_CACHE.clear()


class SessionMixin:
    """Connection, session, schema, and metadata-cache management."""

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
        self._connect_lock = threading.RLock()

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
        with self._connect_lock:
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
        with self._connect_lock:
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

                logger.debug(f"Connected to database: {self._mask_url(self.database_url)}")
            except Exception as e:
                raise DatabaseError(f"Failed to connect to database: {str(e)}") from e

    def _mask_url(self, url: str) -> str:
        """Mask password in URL for logging."""
        return mask_database_url(url)

    def _rollback(self) -> None:
        """Roll back the active session if one exists."""
        if self._session is not None:
            self._session.rollback()

    def _db_error(self, label: str, e: Exception) -> DatabaseError:
        """
        Roll back the active session and build a DatabaseError for a failed
        write operation.

        Parameters
        ----------
        label : str
            Human-readable description of the failed operation, used as the
            error message prefix.
        e : Exception
            The original exception.

        Returns
        -------
        DatabaseError
            Exception chaining *e* with the message ``"{label}: {e}"``.
        """
        self._rollback()
        error = DatabaseError(f"{label}: {str(e)}")
        error.__cause__ = e
        return error

    def close(self) -> None:
        """
        Close the database connection.

        Does nothing if not connected.
        """
        if self._session:
            self._session.close()
            self._session = None
        self._schema_ready = False
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.SessionLocal = None
        logger.debug("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_tables(self, force: bool = False) -> None:
        """
        Create database tables for papers and embeddings metadata.

        Creates the following tables:
        - papers: Main table for paper information with lightweight ML4PS schema
        - embeddings_metadata: Metadata about embeddings (model used, creation date)
        - clustering_cache: Cache for clustering results

        This method is idempotent - it can be called multiple times without error.
        Tables are only created if they don't already exist.

        For PostgreSQL, successful schema verification is cached for the lifetime of
        the process and keyed by database URL.  Subsequent calls from new
        ``DatabaseManager`` instances skip the ``create_all`` reflection queries.

        Parameters
        ----------
        force : bool, optional
            Run the schema check even when the database URL is already marked as
            ready in the process-level cache.  Defaults to ``False``.

        Raises
        ------
        DatabaseError
            If table creation fails.
        """
        if self.engine is None:
            self.connect()
        assert self.engine is not None

        if not force and self._schema_cache_applies():
            with _SCHEMA_LOCK:
                if self.database_url in _SCHEMA_READY_URLS:
                    self._schema_ready = True
                    return

        try:
            # Create tables only if they don't exist (checkfirst=True is the default)
            # This makes the operation idempotent
            Base.metadata.create_all(bind=self.engine, checkfirst=True)
            self._mark_schema_ready()
            logger.debug("Database tables created successfully")
        except (OperationalError, ProgrammingError, IntegrityError) as e:
            # These exceptions can occur when tables already exist, especially with:
            # - Race conditions in concurrent environments
            # - PostgreSQL's pg_type_typname_nsp_index constraint
            # - SQLite "table already exists" errors
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["already exists", "duplicate", "pg_type_typname_nsp_index"]):
                # Tables already exist - this is fine, just log it
                self._mark_schema_ready()
                logger.debug(f"Tables already exist (this is normal): {str(e)}")
                return
            # For other database errors, re-raise with context
            raise DatabaseError(f"Failed to create tables: {str(e)}") from e
        except Exception as e:
            # Catch any other unexpected errors
            raise DatabaseError(f"Failed to create tables: {str(e)}") from e

    def _schema_cache_applies(self) -> bool:
        """
        Return whether process-level schema caching applies to this backend.

        Returns
        -------
        bool
            ``True`` for PostgreSQL, ``False`` otherwise.
        """
        return self.engine is not None and self.engine.dialect.name == "postgresql"

    def _mark_schema_ready(self) -> None:
        """Mark the current schema as ready and cache the URL when applicable."""
        self._schema_ready = True
        if self._schema_cache_applies():
            with _SCHEMA_LOCK:
                _SCHEMA_READY_URLS.add(self.database_url)

    def _metadata_cache_enabled(self) -> bool:
        """Return whether the metadata cache applies to this backend."""
        return self.database_url.startswith("postgresql")

    def _metadata_cache_ttl(self) -> int:
        """Return the configured metadata cache TTL in seconds."""
        from abstracts_explorer.config import get_config

        return max(0, int(get_config().metadata_cache_ttl))

    def _metadata_cache_get(self, kind: str, *parts: Any) -> Optional[Any]:
        """Return a cached metadata value, or ``None`` when unavailable."""
        if not self._metadata_cache_enabled():
            return None
        return _METADATA_CACHE.get((self.database_url, kind, *parts), self._metadata_cache_ttl())

    def _metadata_cache_set(self, value: Any, kind: str, *parts: Any) -> None:
        """Cache a metadata value when caching applies."""
        if not self._metadata_cache_enabled():
            return
        _METADATA_CACHE.set((self.database_url, kind, *parts), value, self._metadata_cache_ttl())

    def _invalidate_metadata_cache(self) -> None:
        """Invalidate all metadata entries for the current database."""
        _METADATA_CACHE.invalidate_database(self.database_url)
