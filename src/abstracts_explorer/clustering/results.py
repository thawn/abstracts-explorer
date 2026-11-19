"""
Cluster statistics, results assembly, and JSON export.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.utils.json_utils import NumpyEncoder

logger = logging.getLogger(__name__)


class ResultsMixin:
    """Assemble cluster statistics and visualization-ready results."""

    embeddings: Any
    reduced_embeddings: Any
    paper_ids: Any
    metadatas: Any
    cluster_labels: Any
    cluster_label_names: Any
    cluster_keywords: Any
    cluster_hierarchy: Any

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clustering results.

        Returns
        -------
        dict
            Dictionary containing cluster statistics:
            - n_clusters: Number of clusters
            - n_noise: Number of noise points (label -1)
            - cluster_sizes: Dictionary mapping cluster labels to sizes
            - total_papers: Total number of clustered papers

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
            - cluster_labels: Human-readable names for clusters (if generated)
            - cluster_keywords: Keywords for each cluster (if extracted)

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

            # Calculate cluster centers in reduced space
            cluster_centers = self._calculate_cluster_centers()

            results = {
                "points": points,
                "statistics": stats,
                "n_dimensions": int(self.reduced_embeddings.shape[1]),
                "cluster_centers": cluster_centers,
            }

            # Add cluster labels if available
            if self.cluster_label_names:
                results["cluster_labels"] = self.cluster_label_names

            # Add cluster keywords if available
            if self.cluster_keywords:
                results["cluster_keywords"] = self.cluster_keywords

            # Add cluster hierarchy if available (for agglomerative)
            if self.cluster_hierarchy:
                results["cluster_hierarchy"] = self.cluster_hierarchy

            logger.info(f"Generated clustering results with {len(points)} points")
            return results

        except Exception as e:
            raise ClusteringError(f"Failed to generate clustering results: {str(e)}") from e

    def _calculate_cluster_centers(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate cluster centers in the reduced embedding space.

        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary mapping cluster IDs to center coordinates.
            Each center has 'x', 'y', and optionally 'z' coordinates.

        Raises
        ------
        ClusteringError
            If required data is not available.
        """
        if self.reduced_embeddings is None:
            raise ClusteringError("No reduced embeddings available")
        if self.cluster_labels is None:
            raise ClusteringError("No cluster labels available")

        try:
            centers = {}
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            for cluster_id in cluster_ids:
                # Get indices of points in this cluster
                cluster_mask = self.cluster_labels == cluster_id
                cluster_points = self.reduced_embeddings[cluster_mask]

                # Calculate centroid
                centroid = cluster_points.mean(axis=0)

                center = {
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                }

                # Add z coordinate if available
                if len(centroid) > 2:
                    center["z"] = float(centroid[2])

                centers[cluster_id] = center

            return centers

        except Exception as e:
            raise ClusteringError(f"Failed to calculate cluster centers: {str(e)}") from e

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

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"Exported clustering results to {output_path}")

        except Exception as e:
            raise ClusteringError(f"Failed to export to JSON: {str(e)}") from e
