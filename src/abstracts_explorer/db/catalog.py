"""
Conference catalog metadata and embedding-model tracking.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func, and_

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    Paper,
    EmbeddingsMetadata,
)

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.papers import PapersMixin
from abstracts_explorer.utils.time_utils import utcnow

logger = logging.getLogger(__name__)


class CatalogMixin(PapersMixin):
    """Conference catalog metadata and embedding-model tracking."""

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
        cached = self._metadata_cache_get("conference_years")
        if cached is not None:
            return cached

        try:
            stmt: Any = (
                select(Paper.conference, Paper.year)
                .where(
                    and_(
                        Paper.conference.isnot(None),
                        Paper.conference != "",
                        Paper.year.isnot(None),
                    )
                )
                .distinct()
                .order_by(Paper.conference.asc(), Paper.year.desc())
            )
            rows: Any = self.session.execute(stmt).all()

            result: Dict[str, List[int]] = {}
            for row in rows:
                result.setdefault(row[0], []).append(row[1])
            self._metadata_cache_set(result, "conference_years")
            return result
        except Exception as e:
            raise self._db_error("Failed to get conference years from DB", e)

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
            if resolved not in db_conference_years:
                for stored in db_conference_years:
                    if stored.lower() == resolved.lower():
                        resolved = stored
                        break
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
        cached = self._metadata_cache_get("sessions", conference, year)
        if cached is not None:
            return cached

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

            result = list(self.session.execute(stmt).scalars().all())
            self._metadata_cache_set(result, "sessions", conference, year)
            return result
        except Exception as e:
            raise self._db_error("Failed to get sessions", e)

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
        cached = self._metadata_cache_get("conferences", year)
        if cached is not None:
            return cached

        try:
            stmt: Any = (
                select(Paper.conference).distinct().where(and_(Paper.conference.isnot(None), Paper.conference != ""))
            )
            if year is not None:
                stmt = stmt.where(Paper.year == year)
            stmt = stmt.order_by(Paper.conference)

            result = list(self.session.execute(stmt).scalars().all())
            self._metadata_cache_set(result, "conferences", year)
            return result
        except Exception as e:
            raise self._db_error("Failed to get conferences", e)

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
        cached = self._metadata_cache_get("years", conference)
        if cached is not None:
            return cached

        try:
            stmt: Any = select(Paper.year).distinct().where(Paper.year.isnot(None))
            if conference is not None:
                stmt = stmt.where(Paper.conference == conference)
            stmt = stmt.order_by(Paper.year.desc())

            result = list(self.session.execute(stmt).scalars().all())
            self._metadata_cache_set(result, "years", conference)
            return result
        except Exception as e:
            raise self._db_error("Failed to get years", e)

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
            raise self._db_error("Failed to get embedding model", e)

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
                latest.updated_at = utcnow()
            else:
                # Insert new record
                new_metadata = EmbeddingsMetadata(embedding_model=model_name)
                self.session.add(new_metadata)

            self.session.commit()
            logger.info(f"Set embedding model to: {model_name}")
        except Exception as e:
            raise self._db_error("Failed to set embedding model", e)
