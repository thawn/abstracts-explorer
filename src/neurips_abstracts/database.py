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
from neurips_abstracts.plugin import (
    EventMediaModel,
    AuthorModel,
    PaperModel,
)

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
        - papers: Main table for paper information with full NeurIPS schema
        - authors: Authors table with their information
        - paper_authors: Junction table linking papers to authors

        Raises
        ------
        DatabaseError
            If table creation fails.
        """
        if not self.connection:
            raise DatabaseError("Not connected to database")

        try:
            cursor = self.connection.cursor()

            # Create authors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS authors (
                    id INTEGER PRIMARY KEY,
                    fullname TEXT NOT NULL,
                    url TEXT,
                    institution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index on author fullname for searching
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_author_fullname 
                ON authors(fullname)
            """
            )

            # Create index on institution
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_author_institution 
                ON authors(institution)
            """
            )

            # Create papers table with full NeurIPS schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY,
                    uid TEXT,
                    name TEXT NOT NULL,
                    authors TEXT,
                    abstract TEXT,
                    topic TEXT,
                    keywords TEXT,
                    decision TEXT,
                    session TEXT,
                    eventtype TEXT,
                    event_type TEXT,
                    room_name TEXT,
                    virtualsite_url TEXT,
                    url TEXT,
                    sourceid INTEGER,
                    sourceurl TEXT,
                    starttime TEXT,
                    endtime TEXT,
                    starttime2 TEXT,
                    endtime2 TEXT,
                    diversity_event TEXT,
                    paper_url TEXT,
                    paper_pdf_url TEXT,
                    children_url TEXT,
                    children TEXT,
                    children_ids TEXT,
                    parent1 TEXT,
                    parent2 TEXT,
                    parent2_id TEXT,
                    eventmedia TEXT,
                    show_in_schedule_overview INTEGER,
                    visible INTEGER,
                    poster_position TEXT,
                    schedule_html TEXT,
                    latitude REAL,
                    longitude REAL,
                    related_events TEXT,
                    related_events_ids TEXT,
                    raw_data TEXT,
                    year INTEGER,
                    conference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_uid 
                ON papers(uid)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_id 
                ON papers(id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_decision 
                ON papers(decision)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic 
                ON papers(topic)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eventtype 
                ON papers(eventtype)
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

            # Create paper_authors junction table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_authors (
                    paper_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    author_order INTEGER NOT NULL,
                    PRIMARY KEY (paper_id, author_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                    FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
                )
            """
            )

            # Create indices for junction table
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_authors_paper 
                ON paper_authors(paper_id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_authors_author 
                ON paper_authors(author_id)
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
                    # Validate record using Pydantic model
                    paper = PaperModel(**record)
                except ValidationError as e:
                    error_msg = f"Validation error for record {idx}: {str(e)}"
                    logger.warning(error_msg)
                    validation_errors.append(error_msg)
                    continue

                # Extract validated fields
                paper_id = paper.id
                uid = paper.uid
                name = paper.name
                abstract = paper.abstract

                # Handle authors (could be list of dicts or string)
                authors_data = paper.authors
                authors_str = ""  # Store comma-separated author IDs
                author_objects = []  # Store author objects for separate table

                if isinstance(authors_data, list):
                    # If list of dicts with author details, process them
                    if authors_data and isinstance(authors_data[0], dict):
                        author_ids = []
                        for author_dict in authors_data:
                            try:
                                # Validate author using Pydantic model
                                author = AuthorModel(**author_dict)
                                author_ids.append(str(author.id))
                                author_objects.append(
                                    {
                                        "id": author.id,
                                        "fullname": author.fullname,
                                        "url": author.url,
                                        "institution": author.institution,
                                    }
                                )
                            except ValidationError as e:
                                logger.warning(f"Author validation error in paper {paper_id}: {str(e)}")
                                # Skip invalid author but continue with paper
                                continue
                        # Store comma-separated author IDs
                        authors_str = ", ".join(author_ids)
                    else:
                        # Simple string list (keep as is for backward compatibility)
                        authors_str = ", ".join(str(a) for a in authors_data)
                else:
                    authors_str = str(authors_data) if authors_data else ""

                topic = paper.topic

                # Handle keywords (could be list or string)
                keywords = paper.keywords
                if isinstance(keywords, list):
                    keywords = ", ".join(str(k) for k in keywords)

                decision = paper.decision
                session = paper.session
                eventtype = paper.eventtype
                event_type = paper.event_type
                room_name = paper.room_name
                virtualsite_url = paper.virtualsite_url
                url = paper.url
                sourceid = paper.sourceid
                sourceurl = paper.sourceurl
                starttime = paper.starttime
                endtime = paper.endtime
                starttime2 = paper.starttime2
                endtime2 = paper.endtime2
                # Convert boolean diversity_event to string for database storage
                diversity_event = (
                    str(paper.diversity_event) if isinstance(paper.diversity_event, bool) else paper.diversity_event
                )
                paper_url = paper.paper_url
                paper_pdf_url = paper.paper_pdf_url
                children_url = paper.children_url

                # Handle array fields
                children = paper.children
                if isinstance(children, list):
                    children = json.dumps(children)

                children_ids = paper.children_ids
                if isinstance(children_ids, list):
                    children_ids = json.dumps(children_ids)

                parent1 = paper.parent1
                parent2 = paper.parent2
                parent2_id = paper.parent2_id

                # Validate and process eventmedia
                eventmedia_data = paper.eventmedia
                if isinstance(eventmedia_data, list):
                    validated_eventmedia = []
                    for media_item in eventmedia_data:
                        if isinstance(media_item, dict):
                            try:
                                # Validate each media item
                                validated_media = EventMediaModel(**media_item)
                                validated_eventmedia.append(validated_media.model_dump(exclude_none=True))
                            except ValidationError as e:
                                logger.warning(f"EventMedia validation error in paper {paper_id}: {str(e)}")
                                # Skip invalid media item but continue
                                continue
                        else:
                            # Keep non-dict items as-is for backward compatibility
                            validated_eventmedia.append(media_item)
                    eventmedia = json.dumps(validated_eventmedia)
                elif eventmedia_data is None:
                    eventmedia = None
                else:
                    # Handle string or other types
                    eventmedia = json.dumps(eventmedia_data) if eventmedia_data else None

                show_in_schedule_overview = 1 if paper.show_in_schedule_overview else 0
                visible = 1 if paper.visible else 0
                poster_position = paper.poster_position
                schedule_html = paper.schedule_html
                latitude = paper.latitude
                longitude = paper.longitude

                related_events = paper.related_events
                if isinstance(related_events, list):
                    related_events = json.dumps(related_events)

                related_events_ids = paper.related_events_ids
                if isinstance(related_events_ids, list):
                    related_events_ids = json.dumps(related_events_ids)

                # Extract year and conference fields
                year = paper.year
                conference = paper.conference

                # Store raw JSON for full data preservation
                raw_data = json.dumps(record, ensure_ascii=False)

                try:
                    # Insert paper
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO papers 
                        (id, uid, name, authors, abstract, topic, keywords, decision, session, 
                         eventtype, event_type, room_name, virtualsite_url, url, sourceid, sourceurl,
                         starttime, endtime, starttime2, endtime2, diversity_event, paper_url, 
                         paper_pdf_url, children_url, children, children_ids, parent1, parent2, 
                         parent2_id, eventmedia, show_in_schedule_overview, visible, poster_position, 
                         schedule_html, latitude, longitude, related_events, related_events_ids, 
                         year, conference, raw_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            paper_id,
                            uid,
                            name,
                            authors_str,
                            abstract,
                            topic,
                            keywords,
                            decision,
                            session,
                            eventtype,
                            event_type,
                            room_name,
                            virtualsite_url,
                            url,
                            sourceid,
                            sourceurl,
                            starttime,
                            endtime,
                            starttime2,
                            endtime2,
                            diversity_event,
                            paper_url,
                            paper_pdf_url,
                            children_url,
                            children,
                            children_ids,
                            parent1,
                            parent2,
                            parent2_id,
                            eventmedia,
                            show_in_schedule_overview,
                            visible,
                            poster_position,
                            schedule_html,
                            latitude,
                            longitude,
                            related_events,
                            related_events_ids,
                            year,
                            conference,
                            raw_data,
                        ),
                    )

                    # Insert authors into authors table (ignore duplicates)
                    for author_data in author_objects:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO authors (id, fullname, url, institution)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                author_data["id"],
                                author_data["fullname"],
                                author_data["url"],
                                author_data["institution"],
                            ),
                        )

                    # Insert paper-author relationships
                    for idx, author_data in enumerate(author_objects, start=1):
                        cursor.execute(
                            """
                            INSERT INTO paper_authors (paper_id, author_id, author_order)
                            VALUES (?, ?, ?)
                            """,
                            (paper_id, author_data["id"], idx),
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
        topic: Optional[str] = None,
        topics: Optional[List[str]] = None,
        decision: Optional[str] = None,
        eventtype: Optional[str] = None,
        eventtypes: Optional[List[str]] = None,
        session: Optional[str] = None,
        sessions: Optional[List[str]] = None,
        year: Optional[int] = None,
        years: Optional[List[int]] = None,
        conference: Optional[str] = None,
        conferences: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        """
        Search for papers by various criteria.

        Parameters
        ----------
        keyword : str, optional
            Keyword to search in name, abstract, topic, or keywords fields.
        topic : str, optional
            Single topic to filter by (deprecated, use topics instead).
        topics : list[str], optional
            List of topics to filter by (matches ANY).
        decision : str, optional
            Decision type to filter by (e.g., "Accept (poster)", "Accept (oral)").
        eventtype : str, optional
            Single event type to filter by (deprecated, use eventtypes instead).
        eventtypes : list[str], optional
            List of event types to filter by (matches ANY).
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
        ...     print(paper['name'])

        >>> # Search with multiple sessions
        >>> papers = db.search_papers(sessions=["Session 1", "Session 2"])

        >>> # Search with multiple topics and eventtypes
        >>> papers = db.search_papers(
        ...     topics=["Machine Learning", "Computer Vision"],
        ...     eventtypes=["Poster", "Oral"]
        ... )

        >>> # Search with years
        >>> papers = db.search_papers(years=[2024, 2025])
        """
        conditions = []
        parameters: List[Any] = []

        if keyword:
            conditions.append("(name LIKE ? OR abstract LIKE ? OR topic LIKE ? OR keywords LIKE ?)")
            search_term = f"%{keyword}%"
            parameters.extend([search_term, search_term, search_term, search_term])

        # Handle topics (prefer list form, fall back to single)
        topic_list = topics if topics else ([topic] if topic else [])
        if topic_list:
            placeholders = ",".join("?" * len(topic_list))
            conditions.append(f"topic IN ({placeholders})")
            parameters.extend(topic_list)

        if decision:
            conditions.append("decision = ?")
            parameters.append(decision)

        # Handle eventtypes (prefer list form, fall back to single)
        eventtype_list = eventtypes if eventtypes else ([eventtype] if eventtype else [])
        if eventtype_list:
            placeholders = ",".join("?" * len(eventtype_list))
            conditions.append(f"eventtype IN ({placeholders})")
            parameters.extend(eventtype_list)

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

    def get_paper_authors(self, paper_id: int) -> List[sqlite3.Row]:
        """
        Get all authors for a specific paper, ordered by author_order.

        Parameters
        ----------
        paper_id : int
            The paper ID to get authors for.

        Returns
        -------
        list of sqlite3.Row
            List of authors with fields: id, fullname, url, institution, author_order.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     authors = db.get_paper_authors(123456)
        >>> for author in authors:
        ...     print(f"{author['fullname']} - {author['institution']}")
        """
        sql = """
            SELECT a.id, a.fullname, a.url, a.institution, pa.author_order
            FROM authors a
            JOIN paper_authors pa ON a.id = pa.author_id
            WHERE pa.paper_id = ?
            ORDER BY pa.author_order
        """
        return self.query(sql, (paper_id,))

    def get_author_papers(self, author_id: int) -> List[sqlite3.Row]:
        """
        Get all papers by a specific author.

        Parameters
        ----------
        author_id : int
            The author ID to get papers for.

        Returns
        -------
        list of sqlite3.Row
            List of papers by this author.

        Raises
        ------
        DatabaseError
            If query fails.

        Examples
        --------
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     papers = db.get_author_papers(457880)
        >>> for paper in papers:
        ...     print(paper['name'])
        """
        sql = """
            SELECT p.*
            FROM papers p
            JOIN paper_authors pa ON p.id = pa.paper_id
            WHERE pa.author_id = ?
            ORDER BY p.name
        """
        return self.query(sql, (author_id,))

    def search_authors(
        self,
        name: Optional[str] = None,
        institution: Optional[str] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        """
        Search for authors by name or institution.

        Parameters
        ----------
        name : str, optional
            Name to search for (partial match).
        institution : str, optional
            Institution to search for (partial match).
        limit : int, default=100
            Maximum number of results to return.

        Returns
        -------
        list of sqlite3.Row
            Matching authors with fields: id, fullname, url, institution.

        Raises
        ------
        DatabaseError
            If search fails.

        Examples
        --------
        >>> db = DatabaseManager("neurips.db")
        >>> with db:
        ...     authors = db.search_authors(name="Huang")
        >>> for author in authors:
        ...     print(f"{author['fullname']} - {author['institution']}")

        >>> # Search by institution
        >>> authors = db.search_authors(institution="Stanford")
        """
        conditions = []
        parameters: List[Any] = []

        if name:
            conditions.append("fullname LIKE ?")
            parameters.append(f"%{name}%")

        if institution:
            conditions.append("institution LIKE ?")
            parameters.append(f"%{institution}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM authors WHERE {where_clause} LIMIT ?"
        parameters.append(limit)

        return self.query(sql, tuple(parameters))

    def get_author_count(self) -> int:
        """
        Get the total number of unique authors in the database.

        Returns
        -------
        int
            Number of unique authors.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        result = self.query("SELECT COUNT(*) as count FROM authors")
        return result[0]["count"] if result else 0

    def get_filter_options(self, year: Optional[int] = None, conference: Optional[str] = None) -> dict:
        """
        Get distinct values for filterable fields.

        Returns a dictionary with lists of distinct values for session, topic,
        eventtype, year, and conference fields that can be used to populate filter dropdowns.
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
            Dictionary with keys 'sessions', 'topics', 'eventtypes', 'years', 'conferences' containing
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

            # Get distinct topics
            topics_result = self.query(
                f"SELECT DISTINCT topic FROM papers WHERE {where_clause} AND topic IS NOT NULL AND topic != '' ORDER BY topic",
                tuple(parameters) if parameters else (),
            )
            topics = [row["topic"] for row in topics_result]

            # Get distinct eventtypes
            eventtypes_result = self.query(
                f"SELECT DISTINCT eventtype FROM papers WHERE {where_clause} AND eventtype IS NOT NULL AND eventtype != '' ORDER BY eventtype",
                tuple(parameters) if parameters else (),
            )
            eventtypes = [row["eventtype"] for row in eventtypes_result]

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
                "topics": topics,
                "eventtypes": eventtypes,
                "years": years,
                "conferences": conferences,
            }
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get filter options: {str(e)}") from e
