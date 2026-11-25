"""
Keyword and semantic search methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.filters import _filter_has_key
from abstracts_explorer.db.papers import PapersMixin

if TYPE_CHECKING:
    from abstracts_explorer.embeddings.query import QueryEmbedding

PaperMatch = tuple[str, float]


class _SearchFilterKwargs(TypedDict, total=False):
    """Shared keyword arguments for structured search filters."""

    conferences: Optional[List[str]]
    years: Optional[List[int]]
    sessions: Optional[List[str]]
    locations: Optional[List[str]]
    where_filters: Optional[Dict[str, Any]]
    field_filters: Optional[Dict[str, str]]


logger = logging.getLogger(__name__)


class SearchMixin(PapersMixin):
    """Keyword and semantic search methods."""

    def _resolve_filter_ids(
        self,
        *,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        field_filters: Optional[Dict[str, str]] = None,
    ) -> Optional[List[str]]:
        """Return candidate UIDs, or None when no filters are present."""
        if not (conferences or years or sessions or locations or where_filters or field_filters):
            return None
        return self.resolve_paper_uids(
            conferences=conferences,
            years=years,
            sessions=sessions,
            locations=locations,
            where_filters=where_filters,
            field_filters=field_filters,
        )

    def keyword_search(
        self,
        keyword: Optional[str] = None,
        limit: int = 100,
        *,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        field_filters: Optional[Dict[str, str]] = None,
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
        limit : int, default=100
            Maximum number of results to return.
        conferences : list[str], optional
            List of conferences to filter by (matches ANY).
        years : list[int], optional
            List of years to filter by (matches ANY).
        sessions : list[str], optional
            List of sessions to filter by (matches ANY).
        locations : list[str], optional
            List of presentation locations to filter by (matches ANY).
        where_filters : dict, optional
            Additional SQL-style filters.  Structured conference/year/session
            filters already present here are not duplicated.
        field_filters : dict of str to str, optional
            Mapping of Paper column names to search values.  Each entry adds
            a case-insensitive substring condition on the corresponding column
            (e.g. ``{"authors": "Smith", "award": "Best Paper"}``).

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
            normalized_where = self._normalize_where_filters(where_filters)
            filter_terms: List[Dict[str, Any]] = []

            if normalized_where:
                filter_terms.append(normalized_where)

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

            if conferences and not _filter_has_key(normalized_where, "conference"):
                filter_terms.append({"conference": {"$in": conferences}})
            if years and not _filter_has_key(normalized_where, "year"):
                filter_terms.append({"year": {"$in": years}})
            if sessions and not _filter_has_key(normalized_where, "session"):
                filter_terms.append({"session": {"$in": sessions}})
            if locations and not _filter_has_key(normalized_where, "location"):
                filter_terms.append({"location": {"$in": locations}})
            if field_filters:
                for field_name, value in field_filters.items():
                    filter_terms.append({field_name: {"$contains": value}})

            if not filter_terms:
                combined_filters: Optional[Dict[str, Any]] = None
            elif len(filter_terms) == 1:
                combined_filters = filter_terms[0]
            else:
                combined_filters = {"$and": filter_terms}

            return self.get_paper(combined_filters, limit=limit)
        except DatabaseError:
            raise
        except Exception as e:
            raise self._db_error("Search failed", e)

    def find_paper_ids_within_distance(
        self,
        query: QueryEmbedding,
        distance_threshold: float = 1.1,
        *,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        field_filters: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[PaperMatch]:
        """
        Return ordered paper matches within a distance threshold.

        This method resolves structured filters to candidate paper UIDs and
        runs one Chroma distance query.  It returns ordered ``(paper_uid, distance)``
        pairs for all matches within the threshold, no full paper details.
        use :meth:`get_result_details` for that.

        Parameters
        ----------
        query : QueryEmbedding
            Query embedding object. Its vector is generated lazily once.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        conferences : list of str, optional
            Restrict results to these conferences.
        years : list of int, optional
            Restrict results to these years.
        sessions : list of str, optional
            Restrict results to these sessions.
        locations : list of str, optional
            Restrict results to these presentation locations.
        where_filters : dict, optional
            Additional SQL-style filters.
        field_filters : dict of str to str, optional
            Case-insensitive SQL substring filters applied per field.
        limit : int, optional
            Maximum number of matches to return.  When omitted, all matches
            within the threshold are returned.

        Returns
        -------
        list of tuple
            Ordered ``(paper_uid, distance)`` pairs within the threshold.
        """
        ids = self._resolve_filter_ids(
            conferences=conferences,
            years=years,
            sessions=sessions,
            locations=locations,
            where_filters=where_filters,
            field_filters=field_filters,
        )
        if ids is not None and not ids:
            return []

        return query.embeddings_manager.query_ids_within_distance(
            query=query,
            distance_threshold=distance_threshold,
            ids=ids,
            limit=limit,
        )

    def get_result_details(
        self,
        matches: Sequence[PaperMatch],
        result_slice: Optional[slice] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hydrate a slice of ordered semantic matches.

        The slice is applied before querying the SQL database, so only the
        selected paper UIDs are fetched. Distances and match order are
        preserved for the hydrated papers.

        Parameters
        ----------
        matches : sequence of tuple[str, float]
            Ordered ``(paper_uid, distance)`` pairs returned by
            :meth:`find_paper_ids_within_distance`.
        result_slice : slice, optional
            Slice of *matches* to hydrate. Use ``slice(0, limit)`` for the
            first page, ``slice(offset, offset + limit)`` for pagination, or
            ``None`` for all matches. Standard Python slice semantics apply,
            including negative indices and steps.

        Returns
        -------
        list of dict
            Hydrated paper dictionaries with ``distance`` and ``similarity``
            fields, preserving the selected match order. Returns an empty list
            when the slice selects no matches.

        Examples
        --------
        >>> matches = db.find_paper_ids_within_distance(query, threshold)
        >>> total_matches = len(matches)
        >>> first_page = db.get_result_details(matches, slice(0, 20))
        >>> second_page = db.get_result_details(matches, slice(20, 40))
        >>> all_results = db.get_result_details(matches)
        """
        selected = matches if result_slice is None else matches[result_slice]
        if not selected:
            return []

        ids = [uid for uid, _ in selected]
        distances = [distance for _, distance in selected]
        papers_by_uid = {paper["uid"]: paper for paper in self.get_papers_by_uids(ids)}
        papers: List[Dict[str, Any]] = []
        for uid, distance in zip(ids, distances):
            paper = papers_by_uid.get(uid)
            if paper is None:
                continue
            paper["distance"] = distance
            paper["similarity"] = max(0.0, 1.0 - float(distance))
            papers.append(paper)
        return papers

    @staticmethod
    def group_papers_by_content(papers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collapse duplicate presentations of the same abstract.

        Rows that share a ``content_hash`` (for example streamed orals or an
        oral and poster for the same paper) are merged into one result.  The
        merged result keeps the first row's fields and adds a ``locations``
        list plus a ``presentations`` list describing each presentation.

        Parameters
        ----------
        papers : sequence of dict
            Paper dictionaries, typically from a search result.

        Returns
        -------
        list of dict
            Grouped paper dictionaries in first-seen order.
        """
        grouped: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for paper in papers:
            key = str(paper.get("content_hash") or paper.get("uid") or "")
            if key not in grouped:
                entry = dict(paper)
                entry["locations"] = []
                entry["presentations"] = []
                grouped[key] = entry
                order.append(key)
            entry = grouped[key]
            location = paper.get("location")
            if location and location not in entry["locations"]:
                entry["locations"].append(location)
            entry["presentations"].append(
                {
                    "uid": paper.get("uid"),
                    "session": paper.get("session"),
                    "room_name": paper.get("room_name"),
                    "poster_position": paper.get("poster_position"),
                    "location": location,
                }
            )
        return [grouped[key] for key in order]

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        use_embeddings: bool = False,
        distance_threshold: float = 1.1,
        embeddings_manager: Optional[Any] = None,
        include_total: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Perform a user-query search, optionally using semantic embeddings.

        This is the main search entry point for the web UI.  It parses
        ``field:"value"`` filters from *query* and combines the keyword or
        semantic search primitives.

        Parameters
        ----------
        query : str
            Search query text.  May include ``field:"value"`` filters.
        limit : int, optional
            Maximum number of results, by default 10.
        conferences : list of str, optional
            Filter by conference names.
        years : list of int, optional
            Filter by publication years.
        sessions : list of str, optional
            Filter by paper sessions.
        locations : list of str, optional
            Filter by presentation locations.
        where_filters : dict, optional
            Additional SQL-style filters.
        use_embeddings : bool, optional
            Use semantic search instead of SQL keyword search.
        distance_threshold : float, optional
            Maximum embedding distance for semantic results.
        embeddings_manager : EmbeddingsManager, optional
            Connected embeddings manager.  A temporary manager is created when
            omitted.
        include_total : bool, optional
            When ``True``, return the total number of semantic matches before
            the limit in the second tuple element.  This requires a full match
            query.  When ``False`` (default), only the top results are fetched
            and the total is ``None``.

        Returns
        -------
        tuple
            ``(papers, total_similar)``.  ``total_similar`` is the number of
            semantic matches before limiting, or ``None`` for keyword,
            field-filter-only, and implicit-author searches.

        Raises
        ------
        DatabaseError
            If the search fails.
        """
        from abstracts_explorer.embeddings import EmbeddingsManager

        field_filters, remaining_query = self.parse_field_filters(query)
        filters: _SearchFilterKwargs = {
            "conferences": conferences,
            "years": years,
            "sessions": sessions,
            "locations": locations,
            "where_filters": where_filters,
            "field_filters": field_filters or None,
        }

        if not use_embeddings:
            papers = self.keyword_search(
                keyword=remaining_query if remaining_query else None,
                limit=limit,
                **filters,
            )
            return self.group_papers_by_content(papers), None

        if field_filters and not remaining_query:
            uids = self._resolve_filter_ids(**filters) or []
            papers = self.get_papers_by_uids(uids[:limit] if limit else uids)
            return self.group_papers_by_content(papers), None

        if not field_filters:
            author_filters: _SearchFilterKwargs = {
                **filters,
                "field_filters": {"authors": remaining_query},
            }
            author_uids = self._resolve_filter_ids(**author_filters) or []
            if author_uids:
                return self.group_papers_by_content(self.get_papers_by_uids(author_uids[:limit])), None

        owns_manager = embeddings_manager is None
        em = embeddings_manager or EmbeddingsManager()
        try:
            from abstracts_explorer.embeddings.query import QueryEmbedding

            query_embedding = QueryEmbedding(remaining_query, em)
            matches = self.find_paper_ids_within_distance(
                query_embedding,
                distance_threshold,
                limit=None if include_total else limit,
                **filters,
            )
            total_similar = len(matches) if include_total else None
            return self.group_papers_by_content(self.get_result_details(matches, slice(0, limit))), total_similar
        finally:
            if owns_manager:
                em.close()
