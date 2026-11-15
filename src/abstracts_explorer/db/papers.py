"""
Paper CRUD and query methods.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select, delete, func, and_

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
    Paper,
)

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.filters import WhereFilterMixin, _filter_has_key

logger = logging.getLogger(__name__)


class PapersMixin(WhereFilterMixin):
    """Paper CRUD and query methods."""

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
            self._invalidate_metadata_cache()

            return uid

        except Exception as e:
            raise self._db_error("Failed to add paper", e)

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

    def resolve_paper_uids(
        self,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        field_filters: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Resolve structured filters to paper UIDs.

        Parameters
        ----------
        conferences : list of str, optional
            Conference names to match.
        years : list of int, optional
            Publication years to match.
        sessions : list of str, optional
            Session names to match.
        where_filters : dict, optional
            SQL-style filter dictionary.  See :meth:`_build_where_conditions`.
        field_filters : dict, optional
            Mapping of paper field to a case-insensitive substring.

        Returns
        -------
        list of str
            Matching paper UIDs.
        """
        terms: List[Dict[str, Any]] = []
        normalized_where = self._normalize_where_filters(where_filters)
        if normalized_where:
            terms.append(normalized_where)
        if conferences and not _filter_has_key(normalized_where, "conference"):
            terms.append({"conference": {"$in": list(conferences)}})
        if years and not _filter_has_key(normalized_where, "year"):
            terms.append({"year": {"$in": [int(year) for year in years]}})
        if sessions and not _filter_has_key(normalized_where, "session"):
            terms.append({"session": {"$in": list(sessions)}})
        if field_filters:
            for field_name, value in field_filters.items():
                terms.append({field_name: {"$contains": value}})

        if not terms:
            return self.get_paper_uids()
        where_clause = terms[0] if len(terms) == 1 else {"$and": terms}
        return self.get_paper_uids(where_clause)

    def get_paper_uids(
        self,
        where_filters: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> List[str]:
        """
        Return paper UIDs matching ``where_filters``.

        Parameters
        ----------
        where_filters : dict, optional
            Mapping of Paper column names to filter values.
        limit : int, default=0
            Maximum number of results.  ``0`` means no limit.

        Returns
        -------
        list of str
            Matching paper UIDs.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        try:
            stmt: Any = select(Paper.uid)
            conditions = self._build_where_conditions(where_filters)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            if limit:
                stmt = stmt.limit(limit)
            return list(self.session.execute(stmt).scalars().all())
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve paper UIDs: {str(e)}") from e

    def get_paper_summaries_by_uids(self, uids: List[str]) -> List[Dict[str, Any]]:
        """
        Return lightweight paper records for the given UIDs.

        Parameters
        ----------
        uids : list of str
            Paper UIDs to retrieve.  Unknown UIDs are omitted.

        Returns
        -------
        list of dict
            Dictionaries with ``uid``, ``title``, ``abstract``, ``year``,
            ``conference``, and ``session``.  Results follow input order.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        if not uids:
            return []

        try:
            summaries: Dict[str, Dict[str, Any]] = {}
            for start in range(0, len(uids), 500):
                batch = uids[start : start + 500]
                stmt: Any = select(
                    Paper.uid,
                    Paper.title,
                    Paper.abstract,
                    Paper.year,
                    Paper.conference,
                    Paper.session,
                ).where(Paper.uid.in_(batch))
                for row in self.session.execute(stmt):
                    summaries[str(row.uid)] = {
                        "uid": str(row.uid),
                        "title": row.title or "",
                        "abstract": row.abstract or "",
                        "year": row.year,
                        "conference": row.conference or "",
                        "session": row.session or "",
                    }
            return [summaries[uid] for uid in uids if uid in summaries]
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve paper summaries: {str(e)}") from e

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

        # Query in fixed-size batches to stay well below DB parameter limits
        papers: List[Dict[str, Any]] = []
        for start in range(0, len(uids), 500):
            batch = uids[start : start + 500]
            papers.extend(self.get_paper({"uid": {"$in": batch}}, limit=0))
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
            self._invalidate_metadata_cache()
            count = getattr(result, "rowcount", 0) or 0
            logger.info(f"Deleted {count} papers for {conference}/{year}")
            return count

        except Exception as e:
            raise self._db_error(f"Failed to delete papers for {conference}/{year}", e)
