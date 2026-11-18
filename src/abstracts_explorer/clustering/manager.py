"""
Clustering manager facade.

:class:`ClusteringManager` composes the mixins that implement loading,
dimensionality reduction, agglomerative clustering, hierarchy extraction,
labeling, and result assembly/export.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from abstracts_explorer.clustering.algorithms import AlgorithmMixin
from abstracts_explorer.clustering.dimensionality import DimensionalityMixin
from abstracts_explorer.clustering.hierarchy import HierarchyMixin
from abstracts_explorer.clustering.labeling import LabelingMixin
from abstracts_explorer.clustering.loading import LoadingMixin
from abstracts_explorer.clustering.results import ResultsMixin
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


class ClusteringManager(
    LabelingMixin,
    ResultsMixin,
    HierarchyMixin,
    AlgorithmMixin,
    DimensionalityMixin,
    LoadingMixin,
):
    """
    Manager for clustering and dimensionality reduction of embeddings.

    This class handles:

    - Loading embeddings from ChromaDB
    - Clustering using Agglomerative clustering (ward linkage)
    - Dimensionality reduction using t-SNE
    - Automatic cluster labeling using TF-IDF and LLM
    - Keyword extraction for clusters
    - Hierarchical cluster structure tracking
    - Export of results for visualization

    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance to load embeddings from
    database : DatabaseManager
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
    reduced_embeddings : np.ndarray or None
        The reduced dimensionality embeddings
    cluster_labels : np.ndarray or None
        The cluster assignment labels
    cluster_label_names : dict or None
        Human-readable names for each cluster
    cluster_keywords : dict or None
        Keywords extracted for each cluster
    cluster_summaries : dict or None
        Summaries generated for each cluster
    cluster_hierarchy : dict or None
        Hierarchical structure of clusters (for agglomerative)

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.create_collection()
    >>> cm = ClusteringManager(em)
    >>> cm.load_embeddings()
    >>> labels = cm.cluster(distance_threshold=150.0)
    >>> cm.reduce_dimensions(n_components=2)
    >>> cm.extract_cluster_keywords()
    >>> cm.generate_cluster_labels(use_llm=True)
    >>> results = cm.get_clustering_results()
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        database: DatabaseManager,
    ):
        """
        Initialize the ClusteringManager.

        Parameters
        ----------
        embeddings_manager : EmbeddingsManager
            Embeddings manager instance to load embeddings from
        database : DatabaseManager
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
        self.cluster_label_names: Optional[Dict[int, str]] = None
        self.cluster_keywords: Optional[Dict[int, List[str]]] = None
        self.cluster_summaries: Optional[Dict[int, str]] = None
        self.cluster_hierarchy: Optional[Dict[str, Any]] = None
        self.clusterer: Optional[Any] = None
