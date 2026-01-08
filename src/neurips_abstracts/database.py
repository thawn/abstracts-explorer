"""
Database Module
===============

This module provides functionality to load JSON data into a SQLite database.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Import Pydantic models from plugin framework
from neurips_abstracts.plugin import LightweightPaper

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class DatabaseManager:
    """
    Manager for SQLite database operations.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Attributes
    ----------
    db_path : Path
        Path to the SQLite database file.
    connection : sqlite3.Connection or None
        Active database connection if connected.

    Examples
    --------
    >>> db = DatabaseManager("neurips.db")
    >>> db.connect()
    >>> db.create_tables()
    >>> db.close()
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the DatabaseManager.

        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """
        Connect to the SQLite database.

        Creates the database file if it doesn't exist.

        Raises
        ------
        DatabaseError
            If connection fails.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database: {str(e)}") from e

    def close(self) -> None:
        """
        Close the database connection.

        Does nothing if not connected.
        """
        if self.connection:
            self.connection.close()
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
        Create database tables for NeurIPS data.

        Creates the following tables:
        - papers: Main table for paper information with lightweight ML4PS schema
        - embeddings_metadata: Metadata about embeddings (model used, creation date)

        Raises
        ------
        DatabaseError
            If table creation fails.
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.connection.cursor()

            # Create papers table with lightweight ML4PS schema
            # Based on LightweightPaper model fields
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    uid TEXT PRIMARY KEY,
                    original_id TEXT,
                    title TEXT NOT NULL,
                    authors TEXT,
                    abstract TEXT,
                    session TEXT,
                    poster_position TEXT,
                    paper_pdf_url TEXT,
                    poster_image_url TEXT,
                    url TEXT,
                    room_name TEXT,
                    keywords TEXT,
                    starttime TEXT,
                    endtime TEXT,
                    award TEXT,
                    year INTEGER,
                    conference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create embeddings metadata table to track which model was used
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index on original_id
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_original_id 
                ON papers(original_id)
            """
            )

            # Create index on title for searching
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_title 
                ON papers(title)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session 
                ON papers(session)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_year 
                ON papers(year)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conference 
                ON papers(conference)
            """
            )

            self.connection.commit()
            logger.info("Database tables created successfully")

        except sqlite3.Error as e:
            self.connection.rollback()
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
        >>> from neurips_abstracts.plugin import LightweightPaper
        >>> db = DatabaseManager("neurips.db")
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
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            import hashlib

            cursor = self.connection.cursor()

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
            existing = cursor.execute("SELECT uid FROM papers WHERE uid = ?", (uid,)).fetchone()
            if existing:
                logger.debug(f"Skipping duplicate paper: {title} (uid: {uid})")
                return None

            # Extract lightweight schema fields
            session = paper.session
            poster_position = paper.poster_position
            paper_pdf_url = paper.paper_pdf_url
            poster_image_url = paper.poster_image_url
            url = paper.url
            room_name = paper.room_name
            starttime = paper.starttime
            endtime = paper.endtime
            award = paper.award
            year = paper.year
            conference = paper.conference

            # Handle keywords (could be list or None)
            keywords_list = paper.keywords
            keywords_str: str
            if isinstance(keywords_list, list):
                keywords_str = ", ".join(str(k) for k in keywords_list)
            elif keywords_list is None:
                keywords_str = ""
            else:
                keywords_str = ""

            # Use paper's original_id if available, otherwise use uid
            original_id = str(paper.original_id) if paper.original_id else None

            # Insert paper with lightweight schema
            cursor.execute(
                """
                INSERT INTO papers 
                (uid, original_id, title, authors, abstract, session, poster_position,
                 paper_pdf_url, poster_image_url, url, room_name, keywords, starttime, endtime,
                 award, year, conference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    uid,
                    original_id,
                    title,
                    authors_str,
                    abstract,
                    session,
                    poster_position,
                    paper_pdf_url,
                    poster_image_url,
                    url,
                    room_name,
                    keywords_str,
                    starttime,
                    endtime,
                    award,
                    year,
                    conference,
                ),
            )

            self.connection.commit()
            return uid

        except sqlite3.Error as e:
            self.connection.rollback()
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
        >>> from neurips_abstracts.plugin import LightweightPaper
        >>> db = DatabaseManager("neurips.db")
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
        if not self.connection:
            raise DatabaseError("Not connected to database")

        inserted_count = 0

        for paper in papers:
            result = self.add_paper(paper)
            if result is not None:
                inserted_count += 1

        logger.info(f"Successfully inserted {inserted_count} of {len(papers)} papers")
        return inserted_count

    def query(self, sql: str, parameters: tuple = ()) -> List[sqlite3.Row]:
        """
        Execute a SQL query and return results.

        Parameters
        ----------
        sql : str
            SQL query to execute.
        parameters : tuple, optional
            Query parameters for parameterized queries.

        Returns
        -------
        list of sqlite3.Row
            Query results.

        Raises
        ------
        DatabaseError
            If query execution fails.

        Examples
        --------
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     results = db.query("SELECT * FROM papers WHERE eventtype = ?", ("Poster",))
        >>> for row in results:
        ...     print(row['title'])
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, parameters)
            return cursor.fetchall()
        except sqlite3.Error as e:
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
        result = self.query("SELECT COUNT(*) as count FROM papers")
        return result[0]["count"] if result else 0

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
    ) -> List[sqlite3.Row]:
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
        list of sqlite3.Row
            Matching papers.

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
        conditions = []
        parameters: List[Any] = []

        if keyword:
            conditions.append("(title LIKE ? OR abstract LIKE ? OR keywords LIKE ?)")
            search_term = f"%{keyword}%"
            parameters.extend([search_term, search_term, search_term])

        # Handle sessions (prefer list form, fall back to single)
        session_list = sessions if sessions else ([session] if session else [])
        if session_list:
            placeholders = ",".join("?" * len(session_list))
            conditions.append(f"session IN ({placeholders})")
            parameters.extend(session_list)

        # Handle years (prefer list form, fall back to single)
        year_list = years if years else ([year] if year else [])
        if year_list:
            placeholders = ",".join("?" * len(year_list))
            conditions.append(f"year IN ({placeholders})")
            parameters.extend(year_list)

        # Handle conferences (prefer list form, fall back to single)
        conference_list = conferences if conferences else ([conference] if conference else [])
        if conference_list:
            placeholders = ",".join("?" * len(conference_list))
            conditions.append(f"conference IN ({placeholders})")
            parameters.extend(conference_list)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM papers WHERE {where_clause}"
        if limit:
            sql += " LIMIT ?"
            parameters.append(limit)

        return self.query(sql, tuple(parameters))

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
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     authors = db.search_authors_in_papers(name="Huang")
        >>> for author in authors:
        ...     print(author['name'])
        """
        if not name:
            return []

        try:
            # Search for authors in the semicolon-separated authors field
            sql = "SELECT DISTINCT authors FROM papers WHERE authors LIKE ? LIMIT ?"
            parameters = [f"%{name}%", limit * 10]  # Get more papers to extract unique authors

            rows = self.query(sql, tuple(parameters))

            # Extract unique author names
            author_names = set()
            for row in rows:
                if row["authors"]:
                    # Split semicolon-separated authors
                    for author in row["authors"].split(";"):
                        author = author.strip()
                        if name.lower() in author.lower():
                            author_names.add(author)
                            if len(author_names) >= limit:
                                break
                if len(author_names) >= limit:
                    break

            return [{"name": name} for name in sorted(author_names)[:limit]]
        except sqlite3.Error as e:
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
        try:
            # Get all author fields
            rows = self.query("SELECT authors FROM papers WHERE authors IS NOT NULL AND authors != ''")

            # Extract unique author names
            author_names = set()
            for row in rows:
                if row["authors"]:
                    for author in row["authors"].split(";"):
                        author_names.add(author.strip())

            return len(author_names)
        except sqlite3.Error as e:
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
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     filters = db.get_filter_options()
        >>> print(filters['sessions'])
        ['Session 1', 'Session 2', ...]
        >>> print(filters['years'])
        [2023, 2024, 2025]
        >>> # Get filters for specific year
        >>> filters = db.get_filter_options(year=2025)
        """
        try:
            # Build WHERE clause based on filters
            conditions = []
            parameters: List[Any] = []

            if year is not None:
                conditions.append("year = ?")
                parameters.append(year)

            if conference is not None:
                conditions.append("conference = ?")
                parameters.append(conference)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get distinct sessions
            sessions_result = self.query(
                f"SELECT DISTINCT session FROM papers WHERE {where_clause} AND session IS NOT NULL AND session != '' ORDER BY session",
                tuple(parameters) if parameters else (),
            )
            sessions = [row["session"] for row in sessions_result]

            # Get distinct years (not filtered)
            years_result = self.query("SELECT DISTINCT year FROM papers WHERE year IS NOT NULL ORDER BY year DESC")
            years = [row["year"] for row in years_result]

            # Get distinct conferences (not filtered)
            conferences_result = self.query(
                "SELECT DISTINCT conference FROM papers WHERE conference IS NOT NULL AND conference != '' ORDER BY conference"
            )
            conferences = [row["conference"] for row in conferences_result]

            return {
                "sessions": sessions,
                "years": years,
                "conferences": conferences,
            }
        except sqlite3.Error as e:
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
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     model = db.get_embedding_model()
        >>> print(model)
        'text-embedding-qwen3-embedding-4b'
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.connection.cursor()
            # Get the most recent embedding model entry
            cursor.execute(
                """
                SELECT embedding_model FROM embeddings_metadata 
                ORDER BY updated_at DESC 
                LIMIT 1
            """
            )
            result = cursor.fetchone()
            return result["embedding_model"] if result else None
        except sqlite3.Error as e:
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
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     db.set_embedding_model("text-embedding-qwen3-embedding-4b")
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.connection.cursor()
            
            # Check if any record exists
            cursor.execute("SELECT COUNT(*) as count FROM embeddings_metadata")
            result = cursor.fetchone()
            count = result["count"] if result else 0

            if count > 0:
                # Update the most recent record
                cursor.execute(
                    """
                    UPDATE embeddings_metadata 
                    SET embedding_model = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = (SELECT id FROM embeddings_metadata ORDER BY updated_at DESC LIMIT 1)
                """,
                    (model_name,),
                )
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO embeddings_metadata (embedding_model)
                    VALUES (?)
                """,
                    (model_name,),
                )

            self.connection.commit()
            logger.info(f"Set embedding model to: {model_name}")
        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"Failed to set embedding model: {str(e)}") from e
