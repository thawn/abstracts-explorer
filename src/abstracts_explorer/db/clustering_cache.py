"""
Clustering result and hierarchical label cache repositories.
"""

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import select, delete, update, func, and_

# Import Pydantic models from plugin framework

# Import SQLAlchemy models
from abstracts_explorer.db_models import (
    ClusteringCache,
    FlatLabelCache,
    HierarchicalLabelCache,
)

from abstracts_explorer.db.session import SessionMixin

logger = logging.getLogger(__name__)


class ClusteringCacheMixin(SessionMixin):
    """Clustering result and hierarchical label cache repositories."""

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
        if collection_name is not None:
            clustering_params = dict(clustering_params or {})
            clustering_params["__collection__"] = collection_name

        try:
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
            raise self._db_error("Failed to get clustering cache", e)

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

    # ------------------------------------------------------------------
    # Hierarchical label cache
    # ------------------------------------------------------------------

    def get_hierarchical_label_cache(
        self,
        embedding_model: str,
        linkage: str = "ward",
        scope: Optional[str] = None,
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
        scope : str, optional
            Dataset fingerprint.  When provided, only labels saved for the
            same dataset are returned; legacy unscoped entries are ignored.

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
            stmt = (
                select(HierarchicalLabelCache)
                .where(
                    and_(
                        HierarchicalLabelCache.embedding_model == embedding_model,
                        HierarchicalLabelCache.linkage == linkage,
                    )
                )
                .order_by(HierarchicalLabelCache.created_at.desc())
            )
            for result in self.session.execute(stmt).scalars():
                raw = json.loads(result.labels_json)
                if isinstance(raw, dict) and "_labels" in raw:
                    if scope is not None and raw.get("_scope") != scope:
                        continue
                    raw = raw["_labels"]
                elif scope is not None:
                    continue
                return {int(k): v for k, v in raw.items()}
            return None

        except Exception as e:
            raise self._db_error("Failed to get hierarchical label cache", e)

    def delete_hierarchical_label_cache(
        self,
        embedding_model: str,
        linkage: str = "ward",
        scope: Optional[str] = None,
    ) -> int:
        """
        Delete cached hierarchical labels.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        linkage : str, optional
            Agglomerative linkage method (default: ``"ward"``).
        scope : str, optional
            Dataset fingerprint.  When provided, only entries saved for the
            same scope are deleted.  When ``None``, all entries for the
            embedding model and linkage are deleted.

        Returns
        -------
        int
            Number of deleted cache entries.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            stmt = select(HierarchicalLabelCache).where(
                and_(
                    HierarchicalLabelCache.embedding_model == embedding_model,
                    HierarchicalLabelCache.linkage == linkage,
                )
            )
            entries = list(self.session.execute(stmt).scalars())
            deleted = 0
            for entry in entries:
                if scope is not None:
                    raw = json.loads(str(entry.labels_json))
                    if not isinstance(raw, dict) or raw.get("_scope") != scope:
                        continue
                self.session.delete(entry)
                deleted += 1

            if deleted:
                self.session.commit()
                logger.info(
                    f"Deleted {deleted} hierarchical label cache entries: "
                    f"model={embedding_model}, linkage={linkage}, scope={scope}"
                )
            return deleted

        except Exception as e:
            raise self._db_error("Failed to delete hierarchical label cache", e)

    def save_hierarchical_label_cache(
        self,
        embedding_model: str,
        labels: Dict[int, str],
        linkage: str = "ward",
        scope: Optional[str] = None,
        replace_scope: bool = False,
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
        scope : str, optional
            Dataset fingerprint stored with the labels so entries are not
            reused across different datasets.
        replace_scope : bool, optional
            Replace existing entries for the same ``scope`` instead of
            appending a duplicate row, by default ``False``.  Requires a
            non-``None`` scope.

        Raises
        ------
        DatabaseError
            If save fails.
        """
        try:
            if replace_scope and scope is not None:
                self.delete_hierarchical_label_cache(
                    embedding_model=embedding_model,
                    linkage=linkage,
                    scope=scope,
                )

            encoded_labels = {str(k): v for k, v in labels.items()}
            if scope is None:
                labels_json = json.dumps(encoded_labels)
            else:
                labels_json = json.dumps({"_scope": scope, "_labels": encoded_labels})
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
            raise self._db_error("Failed to save hierarchical label cache", e)

    def upsert_hierarchical_label_cache(
        self,
        embedding_model: str,
        labels: Dict[int, str],
        linkage: str = "ward",
        scope: Optional[str] = None,
    ) -> None:
        """
        Merge hierarchical labels into the scoped cache entry.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        labels : dict
            Mapping of ``{node_id: label}`` to merge.
        linkage : str, optional
            Agglomerative linkage method (default: ``"ward"``).
        scope : str, optional
            Dataset fingerprint.

        Raises
        ------
        DatabaseError
            If the update fails.
        """
        try:
            stmt = (
                select(HierarchicalLabelCache)
                .where(
                    and_(
                        HierarchicalLabelCache.embedding_model == embedding_model,
                        HierarchicalLabelCache.linkage == linkage,
                    )
                )
                .order_by(HierarchicalLabelCache.created_at.desc())
            )
            entry = None
            merged: Dict[str, str] = {}
            for candidate in self.session.execute(stmt).scalars():
                raw = json.loads(str(candidate.labels_json))
                if isinstance(raw, dict) and "_labels" in raw:
                    if raw.get("_scope") != scope:
                        continue
                    merged = dict(raw["_labels"])
                    entry = candidate
                    break
                if scope is None:
                    merged = dict(raw)
                    entry = candidate
                    break

            merged.update({str(k): v for k, v in labels.items()})
            if entry is None:
                encoded = (
                    json.dumps({"_scope": scope, "_labels": merged}) if scope is not None else json.dumps(merged)
                )
                entry = HierarchicalLabelCache(
                    embedding_model=embedding_model,
                    linkage=linkage,
                    labels_json=encoded,
                )
                self.session.add(entry)
            else:
                entry.labels_json = (
                    json.dumps({"_scope": scope, "_labels": merged}) if scope is not None else json.dumps(merged)
                )
            self.session.commit()

        except Exception as e:
            raise self._db_error("Failed to upsert hierarchical label cache", e)

    def get_flat_label_cache(
        self,
        embedding_model: str,
        scope: str,
    ) -> Optional[Dict[int, str]]:
        """
        Get cached flat cluster labels for a scope.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        scope : str
            Cache scope.

        Returns
        -------
        dict or None
            Mapping of cluster ID to label, or ``None`` when absent.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        try:
            stmt = (
                select(FlatLabelCache)
                .where(
                    and_(
                        FlatLabelCache.embedding_model == embedding_model,
                        FlatLabelCache.scope == scope,
                    )
                )
                .order_by(FlatLabelCache.created_at.desc())
                .limit(1)
            )
            entry = self.session.execute(stmt).scalars().first()
            if entry is None:
                return None
            raw = json.loads(str(entry.labels_json))
            return {int(k): v for k, v in raw.items()}

        except Exception as e:
            raise self._db_error("Failed to get flat label cache", e)

    def upsert_flat_label_cache(
        self,
        embedding_model: str,
        labels: Dict[int, str],
        scope: str,
    ) -> None:
        """
        Merge flat cluster labels into the scoped cache entry.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        labels : dict
            Mapping of cluster ID to label.
        scope : str
            Cache scope.

        Raises
        ------
        DatabaseError
            If the update fails.
        """
        try:
            stmt = (
                select(FlatLabelCache)
                .where(
                    and_(
                        FlatLabelCache.embedding_model == embedding_model,
                        FlatLabelCache.scope == scope,
                    )
                )
                .order_by(FlatLabelCache.created_at.desc())
                .limit(1)
            )
            entry = self.session.execute(stmt).scalars().first()
            merged: Dict[str, str] = {}
            if entry is not None:
                merged = dict(json.loads(str(entry.labels_json)))

            merged.update({str(k): v for k, v in labels.items()})
            if entry is None:
                entry = FlatLabelCache(
                    embedding_model=embedding_model,
                    scope=scope,
                    labels_json=json.dumps(merged),
                )
                self.session.add(entry)
            else:
                entry.labels_json = json.dumps(merged)
            self.session.commit()

        except Exception as e:
            raise self._db_error("Failed to upsert flat label cache", e)

    def clear_flat_label_cache(
        self,
        embedding_model: str,
        scope: Optional[str] = None,
    ) -> int:
        """
        Delete flat cluster label cache entries.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model.
        scope : str, optional
            When provided, delete only this scope.

        Returns
        -------
        int
            Number of deleted entries.

        Raises
        ------
        DatabaseError
            If deletion fails.
        """
        try:
            conditions = [FlatLabelCache.embedding_model == embedding_model]
            if scope is not None:
                conditions.append(FlatLabelCache.scope == scope)
            result = self.session.execute(delete(FlatLabelCache).where(and_(*conditions)))
            self.session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

        except Exception as e:
            raise self._db_error("Failed to clear flat label cache", e)
