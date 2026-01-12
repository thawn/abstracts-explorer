"""
Clustering Module
=================

This module provides functionality to cluster and visualize paper embeddings
using dimensionality reduction and clustering algorithms from scikit-learn.

Features:
- Dimensionality reduction using PCA and t-SNE
- Clustering using K-Means, DBSCAN, and Agglomerative clustering
- Export clustering results to JSON for visualization
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .embeddings import EmbeddingsManager
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    """Exception raised for clustering operations."""
    pass


class ClusteringManager:
    """
    Manager for clustering and dimensionality reduction of embeddings.

    This class handles:
    - Loading embeddings from ChromaDB
    - Dimensionality reduction (PCA, t-SNE)
    - Clustering (K-Means, DBSCAN, Agglomerative)
    - Export of results for visualization

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance to load embeddings from
    database : DatabaseManager, optional
        Database manager for fetching paper metadata

    Attributes
    ----------
    embeddings_manager : EmbeddingsManager
        The embeddings manager instance
    database : DatabaseManager or None
        The database manager instance
    embeddings : np.ndarray or None
        The loaded embeddings array
    paper_ids : list or None
        The paper IDs corresponding to embeddings
    metadatas : list or None
        The paper metadata corresponding to embeddings

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.connect()
    >>> em.create_collection()
    >>> cm = ClusteringManager(em)
    >>> cm.load_embeddings()
    >>> reduced = cm.reduce_dimensions(method='pca', n_components=2)
    >>> labels = cm.cluster(method='kmeans', n_clusters=5)
    >>> results = cm.get_clustering_results()
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        database: Optional[DatabaseManager] = None,
    ):
        """
        Initialize the ClusteringManager.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Embeddings manager instance to load embeddings from
        database : DatabaseManager, optional
            Database manager for fetching paper metadata
        """
        self.embeddings_manager = embeddings_manager
        self.database = database
        self.embeddings: Optional[np.ndarray] = None
        self.paper_ids: Optional[List[str]] = None
        self.metadatas: Optional[List[Dict[str, Any]]] = None
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None

    def load_embeddings(self, limit: Optional[int] = None) -> int:
        """
        Load embeddings from ChromaDB collection.

        Parameters
        ----------
        limit : int, optional
            Maximum number of embeddings to load. If None, load all.

        Returns
        -------
        int
            Number of embeddings loaded

        Raises
        ------
        ClusteringError
            If loading fails or collection is empty
        """
        if not self.embeddings_manager.collection:
            raise ClusteringError("Collection not initialized in embeddings manager")

        try:
            # Get all embeddings from the collection
            results = self.embeddings_manager.collection.get(
                limit=limit,
                include=["embeddings", "metadatas"]
            )

            if not results["ids"] or len(results["ids"]) == 0:
                raise ClusteringError("No embeddings found in collection")

            self.paper_ids = results["ids"]
            self.embeddings = np.array(results["embeddings"])
            self.metadatas = results["metadatas"]

            logger.info(f"Loaded {len(self.paper_ids)} embeddings with dimension {self.embeddings.shape[1]}")
            return len(self.paper_ids)

        except Exception as e:
            raise ClusteringError(f"Failed to load embeddings: {str(e)}") from e

    def reduce_dimensions(
        self,
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionality of embeddings.

        Parameters
        ----------
        method : str, optional
            Dimensionality reduction method: 'pca' or 'tsne', by default 'pca'
        n_components : int, optional
            Number of components to reduce to, by default 2
        random_state : int, optional
            Random state for reproducibility, by default 42
        **kwargs
            Additional arguments passed to the reduction algorithm

        Returns
        -------
        np.ndarray
            Reduced embeddings array of shape (n_samples, n_components)

        Raises
        ------
        ClusteringError
            If embeddings not loaded or reduction fails
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        try:
            # Standardize embeddings before reduction
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_embeddings = self.scaler.fit_transform(self.embeddings)
            else:
                scaled_embeddings = self.scaler.transform(self.embeddings)

            if method.lower() == "pca":
                reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
                logger.info(f"Applying PCA to reduce to {n_components} dimensions")
            elif method.lower() == "tsne":
                # t-SNE parameters
                perplexity = kwargs.pop("perplexity", min(30, len(self.embeddings) - 1))
                max_iter = kwargs.pop("max_iter", 1000)
                reducer = TSNE(
                    n_components=n_components,
                    random_state=random_state,
                    perplexity=perplexity,
                    max_iter=max_iter,
                    **kwargs
                )
                logger.info(f"Applying t-SNE to reduce to {n_components} dimensions (perplexity={perplexity})")
            else:
                raise ClusteringError(f"Unknown reduction method: {method}. Use 'pca' or 'tsne'.")

            self.reduced_embeddings = reducer.fit_transform(scaled_embeddings)
            logger.info(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")

            return self.reduced_embeddings

        except Exception as e:
            raise ClusteringError(f"Failed to reduce dimensions: {str(e)}") from e

    def cluster(
        self,
        method: str = "kmeans",
        n_clusters: int = 5,
        random_state: int = 42,
        use_reduced: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Cluster embeddings using specified algorithm.

        Parameters
        ----------
        method : str, optional
            Clustering method: 'kmeans', 'dbscan', or 'agglomerative', by default 'kmeans'
        n_clusters : int, optional
            Number of clusters (for kmeans and agglomerative), by default 5
        random_state : int, optional
            Random state for reproducibility, by default 42
        use_reduced : bool, optional
            Whether to cluster reduced embeddings or original embeddings, by default False
        **kwargs
            Additional arguments passed to the clustering algorithm

        Returns
        -------
        np.ndarray
            Cluster labels array of shape (n_samples,)

        Raises
        ------
        ClusteringError
            If embeddings not loaded or clustering fails
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        # Choose embeddings to cluster
        if use_reduced and self.reduced_embeddings is not None:
            data_to_cluster = self.reduced_embeddings
            logger.info(f"Clustering using reduced embeddings of shape {data_to_cluster.shape}")
        else:
            # Standardize original embeddings
            if self.scaler is None:
                self.scaler = StandardScaler()
                data_to_cluster = self.scaler.fit_transform(self.embeddings)
            else:
                data_to_cluster = self.scaler.transform(self.embeddings)
            logger.info(f"Clustering using original embeddings of shape {data_to_cluster.shape}")

        try:
            if method.lower() == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
                logger.info(f"Applying K-Means clustering with {n_clusters} clusters")
            elif method.lower() == "dbscan":
                eps = kwargs.pop("eps", 0.5)
                min_samples = kwargs.pop("min_samples", 5)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
                logger.info(f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}")
            elif method.lower() == "agglomerative":
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
                logger.info(f"Applying Agglomerative clustering with {n_clusters} clusters")
            else:
                raise ClusteringError(
                    f"Unknown clustering method: {method}. Use 'kmeans', 'dbscan', or 'agglomerative'."
                )

            self.cluster_labels = clusterer.fit_predict(data_to_cluster)

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

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clustering results.

        Returns
        -------
        dict
            Dictionary containing cluster statistics:
            - n_clusters: Number of clusters
            - n_noise: Number of noise points (for DBSCAN)
            - cluster_sizes: Dictionary mapping cluster labels to sizes
            - cluster_centers: Cluster centers (if available)

        Raises
        ------
        ClusteringError
            If clustering has not been performed
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")

        try:
            unique_labels = np.unique(self.cluster_labels)
            n_noise = int(np.sum(self.cluster_labels == -1))
            n_clusters = int(len(unique_labels[unique_labels >= 0]))

            # Count papers in each cluster
            cluster_sizes = {}
            for label in unique_labels:
                if label >= 0:
                    cluster_sizes[int(label)] = int(np.sum(self.cluster_labels == label))

            stats = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "cluster_sizes": cluster_sizes,
                "total_papers": int(len(self.cluster_labels)),
            }

            logger.info(f"Cluster statistics: {n_clusters} clusters, {n_noise} noise points")
            return stats

        except Exception as e:
            raise ClusteringError(f"Failed to compute cluster statistics: {str(e)}") from e

    def get_clustering_results(
        self,
        include_metadata: bool = True,
        max_title_length: int = 100,
    ) -> Dict[str, Any]:
        """
        Get complete clustering results for visualization.

        Parameters
        ----------
        include_metadata : bool, optional
            Whether to include paper metadata, by default True
        max_title_length : int, optional
            Maximum length for paper titles, by default 100

        Returns
        -------
        dict
            Dictionary containing:
            - points: List of points with coordinates, cluster labels, and metadata
            - statistics: Cluster statistics
            - n_dimensions: Number of dimensions in reduced embeddings

        Raises
        ------
        ClusteringError
            If required data not available
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")
        if self.reduced_embeddings is None:
            raise ClusteringError("No dimensionality reduction performed. Call reduce_dimensions() first.")
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if self.paper_ids is None:
            raise ClusteringError("No paper IDs available.")

        try:
            points = []
            for i, paper_id in enumerate(self.paper_ids):
                point = {
                    "id": paper_id,
                    "x": float(self.reduced_embeddings[i, 0]),
                    "y": float(self.reduced_embeddings[i, 1]),
                    "cluster": int(self.cluster_labels[i]),
                }

                # Add z-coordinate if available (3D visualization)
                if self.reduced_embeddings.shape[1] > 2:
                    point["z"] = float(self.reduced_embeddings[i, 2])

                # Add metadata if requested
                if include_metadata and self.metadatas:
                    metadata = self.metadatas[i]
                    title = metadata.get("title", "")
                    if len(title) > max_title_length:
                        title = title[:max_title_length] + "..."
                    
                    point["title"] = title
                    point["year"] = metadata.get("year", "")
                    point["conference"] = metadata.get("conference", "")
                    point["session"] = metadata.get("session", "")

                points.append(point)

            stats = self.get_cluster_statistics()

            results = {
                "points": points,
                "statistics": stats,
                "n_dimensions": int(self.reduced_embeddings.shape[1]),
            }

            logger.info(f"Generated clustering results with {len(points)} points")
            return results

        except Exception as e:
            raise ClusteringError(f"Failed to generate clustering results: {str(e)}") from e

    def export_to_json(
        self,
        output_path: Union[str, Path],
        include_metadata: bool = True,
    ) -> None:
        """
        Export clustering results to JSON file.

        Parameters
        ----------
        output_path : str or Path
            Path to output JSON file
        include_metadata : bool, optional
            Whether to include paper metadata, by default True

        Raises
        ------
        ClusteringError
            If export fails
        """
        try:
            results = self.get_clustering_results(include_metadata=include_metadata)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"Exported clustering results to {output_path}")

        except Exception as e:
            raise ClusteringError(f"Failed to export to JSON: {str(e)}") from e


def perform_clustering(
    embeddings_path: Union[str, Path],
    collection_name: str = "papers",
    reduction_method: str = "pca",
    n_components: int = 2,
    clustering_method: str = "kmeans",
    n_clusters: int = 5,
    output_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    limit: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform complete clustering pipeline and optionally export results.

    This is a convenience function that handles the full clustering workflow:
    1. Load embeddings from ChromaDB
    2. Cluster on full embeddings
    3. Apply dimensionality reduction for visualization
    4. Export results if requested

    Parameters
    ----------
    embeddings_path : str or Path
        Path to ChromaDB embeddings database
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers"
    reduction_method : str, optional
        Dimensionality reduction method ('pca' or 'tsne') for visualization, by default 'pca'
    n_components : int, optional
        Number of components for dimensionality reduction, by default 2
    clustering_method : str, optional
        Clustering method ('kmeans', 'dbscan', or 'agglomerative'), by default 'kmeans'
    n_clusters : int, optional
        Number of clusters (for kmeans and agglomerative), by default 5
    output_path : str or Path, optional
        Path to export JSON results. If None, don't export.
    random_state : int, optional
        Random state for reproducibility, by default 42
    limit : int, optional
        Maximum number of embeddings to process. If None, process all.
    **kwargs
        Additional arguments passed to clustering algorithm

    Returns
    -------
    dict
        Clustering results dictionary

    Raises
    ------
    ClusteringError
        If any step fails

    Examples
    --------
    >>> results = perform_clustering(
    ...     embeddings_path="chroma_db",
    ...     reduction_method="tsne",
    ...     clustering_method="kmeans",
    ...     n_clusters=5,
    ...     output_path="clusters.json"
    ... )
    >>> print(f"Found {results['statistics']['n_clusters']} clusters")
    """
    try:
        # Initialize embeddings manager
        em = EmbeddingsManager(
            chroma_path=embeddings_path,
            collection_name=collection_name,
        )
        em.connect()
        em.create_collection()

        # Initialize clustering manager
        cm = ClusteringManager(em)

        # Load embeddings
        logger.info("Loading embeddings...")
        cm.load_embeddings(limit=limit)

        # Perform clustering on full embeddings
        logger.info(f"Clustering using {clustering_method} on full embeddings...")
        cm.cluster(
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=random_state,
            use_reduced=False,  # Cluster on full embeddings
            **kwargs
        )

        # Reduce dimensions for visualization
        logger.info(f"Reducing dimensions using {reduction_method} for visualization...")
        cm.reduce_dimensions(
            method=reduction_method,
            n_components=n_components,
            random_state=random_state,
        )

        # Get results
        results = cm.get_clustering_results()

        # Export if requested
        if output_path:
            logger.info(f"Exporting results to {output_path}...")
            cm.export_to_json(output_path)

        em.close()
        return results

    except Exception as e:
        raise ClusteringError(f"Clustering pipeline failed: {str(e)}") from e
