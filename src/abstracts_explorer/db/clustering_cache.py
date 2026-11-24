"""
Clustering result and hierarchical label cache repositories.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete, update, func, and_

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import ClusteringCache

from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


class ClusteringCacheMixin(SessionMixin):
    """Clustering result cache repository."""

    def _find_clustering_cache_row(
        self,
        embedding_model: str,
        clustering_method: str,
        n_clusters: Optional[int] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
        reduction_method: Optional[str] = None,
        n_components: Optional[int] = None,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> Optional[ClusteringCache]:
        """Return the most recent matching cache row."""
        if collection_name is not None:
            clustering_params = dict(clustering_params or {})
            clustering_params["__collection__"] = collection_name

        stmt = select(ClusteringCache).where(
            and_(
                ClusteringCache.embedding_model == embedding_model,
                ClusteringCache.clustering_method == clustering_method,
            )
        )

        if reduction_method is not None:
            stmt = stmt.where(ClusteringCache.reduction_method == reduction_method)
        if n_components is not None:
            stmt = stmt.where(ClusteringCache.n_components == n_components)
        if n_clusters is not None:
            stmt = stmt.where(ClusteringCache.n_clusters == n_clusters)

        if conference is not None:
            stmt = stmt.where(ClusteringCache.conference == conference)
        else:
            stmt = stmt.where(ClusteringCache.conference.is_(None))

        if year is not None:
            stmt = stmt.where(ClusteringCache.year == year)
        else:
            stmt = stmt.where(ClusteringCache.year.is_(None))

        stmt = stmt.order_by(ClusteringCache.created_at.desc())
        results = self.session.execute(stmt).scalars().all()
        if not results:
            return None

        if clustering_params is None:
            for result in results:
                if result.clustering_params is not None:
                    continue
                return result
            return None

        params_json = json.dumps(clustering_params, sort_keys=True)
        for result in results:
            if result.clustering_params is None:
                continue
            cached_params = json.loads(str(result.clustering_params))
            cached_params_json = json.dumps(cached_params, sort_keys=True)
            if cached_params_json == params_json:
                return result

        return None

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
        collection_name: Optional[str] = None,
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
        collection_name : str, optional
            ChromaDB collection the cached results were computed from.  When
            provided, the cache key includes the collection name so that
            different collections sharing one SQL database cannot collide.

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
            result = self._find_clustering_cache_row(
                embedding_model=embedding_model,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                clustering_params=clustering_params,
                reduction_method=reduction_method,
                n_components=n_components,
                conference=conference,
                year=year,
                collection_name=collection_name,
            )
            return json.loads(str(result.results_json)) if result is not None else None
        except Exception as e:
            raise self._db_error("Failed to get clustering cache", e)

    def get_clustering_cache_entry(
        self,
        embedding_model: str,
        clustering_method: str,
        n_clusters: Optional[int] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
        reduction_method: Optional[str] = None,
        n_components: Optional[int] = None,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached clustering result together with its row ID.

        This mirrors :meth:`get_clustering_cache` and is used when the caller
        needs to update the same cache row in place.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        clustering_method : str
            Clustering algorithm used.
        n_clusters : int, optional
            Number of clusters to match.
        clustering_params : dict, optional
            Additional clustering parameters.
        reduction_method : str, optional
            Dimensionality reduction method.
        n_components : int, optional
            Number of components after reduction.
        conference : str, optional
            Conference name to filter by.
        year : int, optional
            Conference year to filter by.
        collection_name : str, optional
            ChromaDB collection the cached results were computed from.

        Returns
        -------
        dict or None
            ``{"id": int, "results": dict}`` when found, otherwise ``None``.

        Raises
        ------
        DatabaseError
            If query fails.
        """
        try:
            row = self._find_clustering_cache_row(
                embedding_model=embedding_model,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                clustering_params=clustering_params,
                reduction_method=reduction_method,
                n_components=n_components,
                conference=conference,
                year=year,
                collection_name=collection_name,
            )
            if row is None:
                return None
            return {"id": int(row.id), "results": json.loads(str(row.results_json))}
        except Exception as e:
            raise self._db_error("Failed to get clustering cache entry", e)

    def list_clustering_cache_entries(
        self,
        conference: Optional[str] = None,
        year: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return cached clustering rows with decoded results.

        Parameters
        ----------
        conference : str, optional
            Only return entries for this conference.
        year : int, optional
            Only return entries for this year.
        embedding_model : str, optional
            Only return entries for this embedding model.

        Returns
        -------
        list of dict
            Cache rows with ``id``, scope metadata, clustering parameters,
            and decoded ``results``.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        try:
            stmt = select(ClusteringCache)
            if conference is not None:
                stmt = stmt.where(ClusteringCache.conference == conference)
            if year is not None:
                stmt = stmt.where(ClusteringCache.year == year)
            if embedding_model is not None:
                stmt = stmt.where(ClusteringCache.embedding_model == embedding_model)
            stmt = stmt.order_by(ClusteringCache.conference, ClusteringCache.year, ClusteringCache.created_at)
            rows = self.session.execute(stmt).scalars().all()
            entries: List[Dict[str, Any]] = []
            for row in rows:
                entries.append(
                    {
                        "id": int(row.id),
                        "conference": row.conference,
                        "year": row.year,
                        "embedding_model": row.embedding_model,
                        "clustering_method": row.clustering_method,
                        "clustering_params": (
                            json.loads(str(row.clustering_params)) if row.clustering_params is not None else None
                        ),
                        "results": json.loads(str(row.results_json)),
                    }
                )
            return entries
        except Exception as e:
            raise self._db_error("Failed to list clustering cache entries", e)

    def update_clustering_cache_results(self, cache_id: int, results: Dict[str, Any]) -> None:
        """
        Replace the stored results of an existing cache row.

        Parameters
        ----------
        cache_id : int
            Primary key of the cache row to update.
        results : dict
            New clustering results payload.

        Raises
        ------
        DatabaseError
            If the row does not exist or the update fails.
        """
        try:
            stmt = (
                update(ClusteringCache).where(ClusteringCache.id == cache_id).values(results_json=json.dumps(results))
            )
            result = self.session.execute(stmt)
            rowcount = getattr(result, "rowcount", 0) or 0
            if rowcount == 0:
                raise ValueError(f"Clustering cache entry {cache_id} not found")
            self.session.commit()
            logger.info("Updated clustering cache entry %s with backfilled labels", cache_id)
        except Exception as e:
            raise self._db_error(f"Failed to update clustering cache entry {cache_id}", e)

    def update_clustering_cache_entry(
        self,
        cache_id: int,
        *,
        results: Optional[Dict[str, Any]] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the stored results and/or clustering parameters of a cache row.

        Parameters
        ----------
        cache_id : int
            Primary key of the cache row to update.
        results : dict, optional
            Replacement clustering results payload.
        clustering_params : dict, optional
            Replacement clustering parameter dictionary.

        Raises
        ------
        DatabaseError
            If the row does not exist or the update fails.
        """
        values: Dict[str, Any] = {}
        if results is not None:
            values["results_json"] = json.dumps(results)
        if clustering_params is not None:
            values["clustering_params"] = json.dumps(clustering_params)
        if not values:
            return
        try:
            stmt = update(ClusteringCache).where(ClusteringCache.id == cache_id).values(**values)
            result = self.session.execute(stmt)
            rowcount = getattr(result, "rowcount", 0) or 0
            if rowcount == 0:
                raise ValueError(f"Clustering cache entry {cache_id} not found")
            self.session.commit()
            logger.info("Updated clustering cache entry %s", cache_id)
        except Exception as e:
            raise self._db_error(f"Failed to update clustering cache entry {cache_id}", e)

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
        collection_name: Optional[str] = None,
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
        if collection_name is not None:
            clustering_params = dict(clustering_params or {})
            clustering_params["__collection__"] = collection_name

        try:
            # Always try to fill n_clusters from the results statistics
            if n_clusters is None:
                stats = results.get("statistics", {})
                actual = stats.get("n_clusters")
                if actual is not None:
                    n_clusters = int(actual)

            # Serialize results and params to JSON
            results_json = json.dumps(results)
            params_json = json.dumps(clustering_params) if clustering_params is not None else None

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
            raise self._db_error("Failed to save clustering cache", e)

    def delete_clustering_cache_by_id(self, cache_id: int) -> bool:
        """
        Delete a single clustering cache entry by its primary key.

        Parameters
        ----------
        cache_id : int
            Primary key of the cache row to delete.

        Returns
        -------
        bool
            ``True`` when a row was deleted, ``False`` when it did not exist.

        Raises
        ------
        DatabaseError
            If the deletion fails.
        """
        try:
            stmt = delete(ClusteringCache).where(ClusteringCache.id == cache_id)
            result = self.session.execute(stmt)
            self.session.commit()
            deleted = (getattr(result, "rowcount", 0) or 0) > 0
            if deleted:
                logger.info("Deleted clustering cache entry %s", cache_id)
            return deleted
        except Exception as e:
            raise self._db_error(f"Failed to delete clustering cache entry {cache_id}", e)

    def delete_clustering_cache_ids(self, cache_ids: List[int]) -> int:
        """
        Delete clustering cache entries by primary key.

        Parameters
        ----------
        cache_ids : list of int
            Primary keys of the cache rows to delete.  Duplicates are ignored.

        Returns
        -------
        int
            Number of cache entries deleted.

        Raises
        ------
        DatabaseError
            If the deletion fails.
        """
        ids = sorted({int(cache_id) for cache_id in cache_ids})
        if not ids:
            return 0
        try:
            stmt = delete(ClusteringCache).where(ClusteringCache.id.in_(ids))
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0
            self.session.commit()
            logger.info("Deleted %d clustering cache entries", count)
            return count
        except Exception as e:
            raise self._db_error("Failed to delete clustering cache entries", e)

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
            stmt = delete(ClusteringCache).where(
                and_(ClusteringCache.conference == conference, ClusteringCache.year == year)
            )
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0

            self.session.commit()
            logger.info(f"Deleted {count} clustering cache entries for {conference}/{year}")
            return count

        except Exception as e:
            raise self._db_error(f"Failed to delete clustering cache for {conference}/{year}", e)

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
            raise self._db_error(f"Failed to count clustering cache for {conference}/{year}", e)

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
                stmt = delete(ClusteringCache).where(ClusteringCache.embedding_model == embedding_model)
            else:
                # Delete all
                stmt = delete(ClusteringCache)

            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0

            self.session.commit()

            if embedding_model:
                logger.info(f"Cleared {count} clustering cache entries for model: {embedding_model}")
            else:
                logger.info(f"Cleared all {count} clustering cache entries")

            return count

        except Exception as e:
            raise self._db_error("Failed to clear clustering cache", e)

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
            stmt = (
                update(ClusteringCache)
                .where(ClusteringCache.embedding_model == old_model)
                .values(embedding_model=new_model)
            )
            result = self.session.execute(stmt)
            count = getattr(result, "rowcount", 0) or 0

            self.session.commit()
            logger.info(f"Updated {count} clustering cache entries from model {old_model!r} to {new_model!r}")
            return count

        except Exception as e:
            raise self._db_error("Failed to update clustering cache embedding model", e)
