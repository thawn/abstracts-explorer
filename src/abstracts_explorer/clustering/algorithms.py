"""
Agglomerative clustering and default cluster-count selection.
"""

import logging
from typing import Any, Optional

import numpy as np

from abstracts_explorer.clustering.errors import ClusteringError

logger = logging.getLogger(__name__)


def calculate_default_clusters(n_papers: int, min_clusters: int = 2, max_clusters: int = 500) -> int:
    """
    Calculate default number of clusters based on the number of papers.

    Uses the rule: n_clusters = n_papers / 100, clamped to [min_clusters, max_clusters].

    Parameters
    ----------
    n_papers : int
        Number of papers to cluster
    min_clusters : int, optional
        Minimum number of clusters, by default 2
    max_clusters : int, optional
        Maximum number of clusters, by default 500

    Returns
    -------
    int
        Recommended number of clusters

    Examples
    --------
    >>> calculate_default_clusters(50)
    2
    >>> calculate_default_clusters(500)
    5
    >>> calculate_default_clusters(100000)
    500
    """
    if n_papers <= 0:
        return min_clusters

    # Calculate based on n_papers / 100
    n_clusters = max(min_clusters, min(max_clusters, n_papers // 100))

    return n_clusters


class AlgorithmMixin:
    """Agglomerative clustering with a distance threshold or fixed count."""

    embeddings: Any
    scaler: Any
    reduced_embeddings: Any
    clusterer: Any
    cluster_labels: Any
    _extract_cluster_hierarchy: Any
    _scaled_embeddings: Any

    def cluster(
        self,
        n_clusters: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Cluster embeddings using Agglomerative clustering.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. Can be None if distance_threshold is provided.
            If both are None, automatically calculated as n_papers / 100,
            clamped to [2, 500]. By default None.
        **kwargs
            Additional arguments passed to AgglomerativeClustering, e.g.
            distance_threshold (float) and linkage (str).

        Returns
        -------
        np.ndarray
            Cluster labels array of shape (n_samples,)

        Raises
        ------
        ClusteringError
            If embeddings not loaded or clustering fails

        Examples
        --------
        >>> # Agglomerative with distance threshold
        >>> cm.cluster(distance_threshold=150.0, linkage="ward")
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        # Extract distance_threshold if provided
        distance_threshold = kwargs.pop("distance_threshold", None)

        # Calculate default n_clusters if not provided and not using distance_threshold
        if n_clusters is None and distance_threshold is None:
            n_clusters = calculate_default_clusters(len(self.embeddings))
            logger.info(f"Auto-calculated n_clusters={n_clusters} based on {len(self.embeddings)} papers")

        data_to_cluster = self._scaled_embeddings()
        logger.info(f"Clustering using original embeddings of shape {data_to_cluster.shape}")

        try:
            from sklearn.cluster import AgglomerativeClustering  # lazy import to reduce initial load time

            if distance_threshold is not None:
                self.clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    compute_full_tree=True,  # Required for hierarchy
                    **kwargs,
                )
                logger.info(f"Applying Agglomerative clustering with distance_threshold={distance_threshold}")
            else:
                self.clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    compute_full_tree=True,  # Store for potential hierarchy extraction
                    **kwargs,
                )
                logger.info(f"Applying Agglomerative clustering with {n_clusters} clusters")

            self.cluster_labels = self.clusterer.fit_predict(data_to_cluster)

            # Extract hierarchical structure
            self._extract_cluster_hierarchy()

            # Count unique clusters
            unique_labels = np.unique(self.cluster_labels)
            n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise label -1
            n_noise = np.sum(self.cluster_labels == -1)

            logger.info(f"Found {n_clusters_found} clusters")
            if n_noise > 0:
                logger.info(f"Noise points: {n_noise}")

            return self.cluster_labels

        except Exception as e:
            raise ClusteringError(f"Failed to cluster embeddings: {str(e)}") from e
