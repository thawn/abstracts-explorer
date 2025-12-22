"""
Database Module
===============

This module provides functionality to load JSON data into a SQLite database.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

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
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT UNIQUE,
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

            # Create index on uid
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_uid 
                ON papers(uid)
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

    def load_json_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
        """
        Load JSON data into the database.

        Parameters
        ----------
        data : dict or list of dict
            JSON data to load. Can be a single dictionary or a list of dictionaries.

        Returns
        -------
        int
            Number of records inserted.

        Raises
        ------
        DatabaseError
            If loading data fails.

        Examples
        --------
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     db.create_tables()
        ...     count = db.load_json_data(json_data)
        >>> print(f"Inserted {count} records")
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        # Handle different data structures
        if isinstance(data, dict):
            # Check if data contains a list of papers
            if "results" in data:
                # NeurIPS 2025 format with paginated results
                records = data["results"]
            elif "papers" in data:
                records = data["papers"]
            elif "data" in data:
                records = data["data"]
            else:
                # Treat the dict as a single record
                records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")

        try:
            cursor = self.connection.cursor()
            inserted_count = 0
            validation_errors = []

            for idx, record in enumerate(records):
                try:
                    # Validate record using Pydantic model (lightweight schema)
                    paper = LightweightPaper(**record)
                except ValidationError as e:
                    error_msg = f"Validation error for record {idx}: {str(e)}"
                    logger.warning(error_msg)
                    validation_errors.append(error_msg)
                    continue

                # Extract validated fields (lightweight schema only)
                paper_id = paper.id if paper.id else None
                original_id = record.get("uid", "")  # Get original UID from record, not from paper object
                title = paper.title
                abstract = paper.abstract

                # Handle authors - store as semicolon-separated names
                authors_data = paper.authors
                authors_str = ""

                if isinstance(authors_data, list):
                    author_names = []
                    for author_item in authors_data:
                        if isinstance(author_item, dict):
                            # Extract fullname from dict
                            author_names.append(author_item.get("fullname", author_item.get("name", "")))
                        else:
                            # Simple string (author name)
                            author_names.append(str(author_item))
                    authors_str = "; ".join(author_names)
                else:
                    authors_str = str(authors_data) if authors_data else ""

                # Generate UID as hash from title + original_id
                import hashlib

                uid_source = f"{title}:{original_id if original_id else ''}"
                uid = hashlib.sha256(uid_source.encode("utf-8")).hexdigest()[:16]

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

                # Handle keywords (could be list or string)
                keywords = paper.keywords
                if isinstance(keywords, list):
                    keywords = ", ".join(str(k) for k in keywords)
                elif keywords is None:
                    keywords = ""

                try:
                    # Insert paper with lightweight schema
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO papers 
                        (id, uid, original_id, title, authors, abstract, session, poster_position,
                         paper_pdf_url, poster_image_url, url, room_name, keywords, starttime, endtime,
                         award, year, conference)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            paper_id,
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
                            keywords,
                            starttime,
                            endtime,
                            award,
                            year,
                            conference,
                        ),
                    )

                    inserted_count += 1
                except sqlite3.IntegrityError:
                    logger.warning(f"Skipping duplicate record: {paper_id}")
                    continue

            self.connection.commit()

            # Log summary
            if validation_errors:
                logger.warning(f"Encountered {len(validation_errors)} validation errors")
                logger.debug(f"Validation errors: {validation_errors}")

            logger.info(f"Successfully inserted {inserted_count} records")
            return inserted_count

        except sqlite3.Error as e:
            self.connection.rollback()
            raise DatabaseError(f"Failed to load data: {str(e)}") from e

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
        ...     print(row['name'])
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
        sql = f"SELECT * FROM papers WHERE {where_clause} LIMIT ?"
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
            # Search for authors in the comma-separated authors field
            sql = "SELECT DISTINCT authors FROM papers WHERE authors LIKE ? LIMIT ?"
            parameters = [f"%{name}%", limit * 10]  # Get more papers to extract unique authors

            rows = self.query(sql, tuple(parameters))

            # Extract unique author names
            author_names = set()
            for row in rows:
                if row["authors"]:
                    # Split comma-separated authors
                    for author in row["authors"].split(","):
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
                    for author in row["authors"].split(","):
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
