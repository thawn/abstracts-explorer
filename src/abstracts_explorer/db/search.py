"""
Keyword and semantic search methods.
"""

import logging
from typing import Any, Dict, List, Optional

# Import Pydantic models from plugin framework

# Import SQLAlchemy models

from abstracts_explorer.db.errors import DatabaseError
from abstracts_explorer.db.papers import PapersMixin

logger = logging.getLogger(__name__)


class SearchMixin(PapersMixin):
    """Keyword and semantic search methods."""

    def find_papers_within_distance(
        self,
        embeddings_manager: Any,
        query: str,
        distance_threshold: float = 1.1,
        conferences: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sessions: Optional[List[str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Return full paper records within a distance threshold.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Connected embeddings manager.
        query : str
            Search query text.
        distance_threshold : float, optional
            Euclidean distance radius, by default 1.1.
        conferences : list of str, optional
            Restrict results to these conferences.
        years : list of int, optional
            Restrict results to these years.
        sessions : list of str, optional
            Restrict results to these sessions.
        where_filters : dict, optional
            Additional SQL-style filters.
        query_embedding : list of float, optional
            Pre-computed query embedding.

        Returns
        -------
        dict
            Dictionary with ``query``, ``query_embedding``, ``distance``,
            ``papers``, ``count``, and ``total_considered``.
        """
        has_filters = bool(conferences or years or sessions or where_filters)
        ids = (
            self.resolve_paper_uids(
                conferences=conferences,
                years=years,
                sessions=sessions,
                where_filters=where_filters,
            )
            if has_filters
            else None
        )
        if ids is not None and not ids:
            return {
                "query": query,
                "query_embedding": query_embedding,
                "distance": distance_threshold,
                "papers": [],
                "count": 0,
                "total_considered": 0,
            }

        result = embeddings_manager.query_ids_within_distance(
            query=query,
            distance_threshold=distance_threshold,
            ids=ids,
            query_embedding=query_embedding,
        )
        papers_by_uid = {paper["uid"]: paper for paper in self.get_papers_by_uids(result["ids"])}
        papers: List[Dict[str, Any]] = []
        for uid, distance in zip(result["ids"], result["distances"]):
            paper = papers_by_uid.get(uid)
            if paper is None:
                continue
            paper["distance"] = distance
            paper["similarity"] = max(0.0, 1.0 - float(distance))
            papers.append(paper)

        return {
            "query": query,
            "query_embedding": result["query_embedding"],
            "distance": distance_threshold,
            "papers": papers,
            "count": len(papers),
            "total_considered": result["total_considered"],
        }

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
            raise self._db_error("Search failed", e)

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

    def search_papers_semantic(
        self,
        query: str,
        limit: int = 10,
        sessions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        conferences: Optional[List[str]] = None,
        distance_threshold: float = 1.1,
        field_filters: Optional[Dict[str, str]] = None,
        where_filters: Optional[Dict[str, Any]] = None,
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
        where_filters : dict, optional
            Additional SQL-style filter.  Merged with the session/year/
            conference filters.

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
            uids = self.resolve_paper_uids(
                conferences=conferences,
                years=years,
                sessions=sessions,
                where_filters=where_filters,
                field_filters=field_filters,
            )
            return self.get_papers_by_uids(uids[:limit] if limit else uids)

        semantic_query = query

        if not field_filters:
            author_uids = self.resolve_paper_uids(
                conferences=conferences,
                years=years,
                sessions=sessions,
                where_filters=where_filters,
                field_filters={"authors": semantic_query},
            )
            if author_uids:
                return self.get_papers_by_uids(author_uids[:limit])

        has_filters = bool(conferences or years or sessions or where_filters or field_filters)
        matching_uids = (
            self.resolve_paper_uids(
                conferences=conferences,
                years=years,
                sessions=sessions,
                where_filters=where_filters,
                field_filters=field_filters,
            )
            if has_filters
            else None
        )
        if has_filters and not matching_uids:
            return []

        owns_manager = embeddings_manager is None
        em = embeddings_manager or EmbeddingsManager()
        try:
            results = em.search_similar(
                semantic_query,
                n_results=limit * 2,
                ids=matching_uids,
            )
            try:
                papers = format_search_results(results, self)
            except PaperFormattingError:
                return []
        finally:
            if owns_manager:
                em.close()

        papers = [paper for paper in papers if paper.get("distance", 0.0) <= distance_threshold]
        return papers[:limit]
