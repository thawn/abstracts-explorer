"""
Clustering Module
=================

This module provides functionality to cluster and visualize paper embeddings
using dimensionality reduction and clustering algorithms from scikit-learn.

Features:
- Dimensionality reduction using PCA, t-SNE, and UMAP
- Clustering using K-Means, DBSCAN, Agglomerative, Fuzzy C-Means, and Spectral clustering
- **NEW: Automatic cluster labeling using TF-IDF and LLM-based methods**
- **NEW: Keyword extraction for each cluster**
- **NEW: Representative paper selection based on cluster centroids**
- **NEW: Hierarchical cluster structure for agglomerative clustering**
- Export clustering results to JSON for visualization

Cluster Labeling
----------------
The module now includes state-of-the-art cluster labeling functionality that:
1. Extracts distinctive keywords for each cluster using TF-IDF analysis
2. Generates human-readable labels using LLM (Large Language Model) integration
3. Identifies representative papers closest to each cluster's centroid

Hierarchical Clustering
-----------------------
When using agglomerative clustering with distance_threshold, the module tracks
the hierarchical structure of clusters, allowing exploration of sub-clusters.

Example
-------
>>> from abstracts_explorer.clustering import ClusteringManager
>>> from abstracts_explorer.embeddings import EmbeddingsManager
>>> 
>>> # Initialize managers
>>> em = EmbeddingsManager()
>>> em.connect()
>>> em.create_collection()
>>> cm = ClusteringManager(em)
>>> 
>>> # Load and cluster embeddings
>>> cm.load_embeddings()
>>> cm.cluster(method='kmeans', n_clusters=5)
>>> cm.reduce_dimensions(method='pca', n_components=2)
>>> 
>>> # Generate cluster labels
>>> cm.extract_cluster_keywords(n_keywords=10)
>>> cm.generate_cluster_labels(use_llm=True)
>>> 
>>> # Get results with labels
>>> results = cm.get_clustering_results()
>>> print(results['cluster_labels'])  # Shows generated labels
>>> print(results['cluster_keywords'])  # Shows extracted keywords
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
try:
    import skfuzzy as fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

from .embeddings import EmbeddingsManager
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    """Exception raised for clustering operations."""
    pass


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


class ClusteringManager:
    """
    Manager for clustering and dimensionality reduction of embeddings.

    This class handles:
    - Loading embeddings from ChromaDB
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Clustering (K-Means, DBSCAN, Agglomerative, Fuzzy C-Means, Spectral)
    - **Automatic cluster labeling using TF-IDF and LLM**
    - **Keyword extraction for clusters**
    - **Representative paper selection**
    - **Hierarchical cluster structure tracking**
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
    fuzzy_memberships : np.ndarray or None
        Fuzzy membership values (for fuzzy c-means)

    Examples
    --------
    >>> em = EmbeddingsManager()
    >>> em.connect()
    >>> em.create_collection()
    >>> cm = ClusteringManager(em)
    >>> cm.load_embeddings()
    >>> reduced = cm.reduce_dimensions(method='pca', n_components=2)
    >>> labels = cm.cluster(method='kmeans', n_clusters=5)
    >>> cm.extract_cluster_keywords()
    >>> cm.generate_cluster_labels(use_llm=True)
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
        self.cluster_label_names: Optional[Dict[int, str]] = None
        self.cluster_keywords: Optional[Dict[int, List[str]]] = None
        self.cluster_summaries: Optional[Dict[int, str]] = None
        self.cluster_hierarchy: Optional[Dict[str, Any]] = None
        self.fuzzy_memberships: Optional[np.ndarray] = None
        self.clusterer: Optional[Any] = None  # Store the clusterer for hierarchy access

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
            Dimensionality reduction method: 'pca', 'tsne', or 'umap', by default 'pca'
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
            elif method.lower() == "umap":
                # UMAP parameters
                # UMAP requires a sufficient number of samples for spectral initialization
                # Practical minimum is around max(n_components + 1, 4) for reliable operation
                min_samples_required = max(n_components + 1, 4)
                if len(self.embeddings) < min_samples_required:
                    raise ClusteringError(
                        f"UMAP requires at least {min_samples_required} samples for {n_components} components, "
                        f"but only {len(self.embeddings)} samples available. Try PCA or t-SNE instead."
                    )
                n_neighbors = kwargs.pop("n_neighbors", min(15, len(self.embeddings) - 1))
                # Ensure n_neighbors is at least 2 and less than n_samples
                n_neighbors = max(2, min(n_neighbors, len(self.embeddings) - 1))
                min_dist = kwargs.pop("min_dist", 0.1)
                metric = kwargs.pop("metric", "cosine")
                reducer = UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    **kwargs
                )
                logger.info(f"Applying UMAP to reduce to {n_components} dimensions (n_neighbors={n_neighbors}, min_dist={min_dist})")
            else:
                raise ClusteringError(f"Unknown reduction method: {method}. Use 'pca', 'tsne', or 'umap'.")

            self.reduced_embeddings = reducer.fit_transform(scaled_embeddings)
            logger.info(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")

            return self.reduced_embeddings

        except Exception as e:
            raise ClusteringError(f"Failed to reduce dimensions: {str(e)}") from e

    def cluster(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        use_reduced: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Cluster embeddings using specified algorithm.

        Parameters
        ----------
        method : str, optional
            Clustering method: 'kmeans', 'dbscan', 'agglomerative', 'fuzzy_cmeans', or 'spectral'.
            By default 'kmeans'.
        n_clusters : int, optional
            Number of clusters (for kmeans, agglomerative, fuzzy_cmeans, and spectral).
            For agglomerative, can be None if distance_threshold is provided.
            If None, automatically calculated as n_papers / 100, clamped to [2, 500].
            By default None.
        random_state : int, optional
            Random state for reproducibility, by default 42
        use_reduced : bool, optional
            Whether to cluster reduced embeddings or original embeddings, by default False
        **kwargs
            Additional arguments passed to the clustering algorithm.
            For agglomerative: distance_threshold (float), linkage (str), affinity (str)
            For dbscan: eps (float), min_samples (int)
            For fuzzy_cmeans: m (float, fuzziness parameter), error (float), maxiter (int)
            For spectral: affinity (str), n_neighbors (int)

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
        >>> cm.cluster(method='agglomerative', distance_threshold=0.5, n_clusters=None)
        
        >>> # Fuzzy C-Means
        >>> cm.cluster(method='fuzzy_cmeans', n_clusters=5, m=2.0)
        
        >>> # Spectral clustering
        >>> cm.cluster(method='spectral', n_clusters=5)
        """
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded. Call load_embeddings() first.")

        # Extract distance_threshold for agglomerative if provided
        distance_threshold = kwargs.pop("distance_threshold", None)
        
        # Calculate default n_clusters if not provided and not using distance_threshold
        if n_clusters is None and distance_threshold is None:
            n_clusters = calculate_default_clusters(len(self.embeddings))
            logger.info(f"Auto-calculated n_clusters={n_clusters} based on {len(self.embeddings)} papers")

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
                self.clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
                logger.info(f"Applying K-Means clustering with {n_clusters} clusters")
                self.cluster_labels = self.clusterer.fit_predict(data_to_cluster)
                
            elif method.lower() == "dbscan":
                eps = kwargs.pop("eps", 0.5)
                min_samples = kwargs.pop("min_samples", 5)
                self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
                logger.info(f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}")
                self.cluster_labels = self.clusterer.fit_predict(data_to_cluster)
                
            elif method.lower() == "agglomerative":
                # Handle agglomerative with distance_threshold or n_clusters
                if distance_threshold is not None:
                    self.clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=distance_threshold,
                        compute_full_tree=True,  # Required for hierarchy
                        **kwargs
                    )
                    logger.info(f"Applying Agglomerative clustering with distance_threshold={distance_threshold}")
                else:
                    self.clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        compute_full_tree=True,  # Store for potential hierarchy extraction
                        **kwargs
                    )
                    logger.info(f"Applying Agglomerative clustering with {n_clusters} clusters")
                    
                self.cluster_labels = self.clusterer.fit_predict(data_to_cluster)
                
                # Extract hierarchical structure
                self._extract_cluster_hierarchy()
                
            elif method.lower() == "fuzzy_cmeans" or method.lower() == "fuzzy-cmeans":
                if not HAS_FUZZY:
                    raise ClusteringError(
                        "Fuzzy C-Means requires scikit-fuzzy. Install with: pip install scikit-fuzzy"
                    )
                if n_clusters is None:
                    raise ClusteringError("n_clusters must be specified for fuzzy c-means")
                    
                # Fuzzy C-Means parameters
                m = kwargs.pop("m", 2.0)  # Fuzziness parameter
                error = kwargs.pop("error", 0.005)
                maxiter = kwargs.pop("maxiter", 1000)
                
                logger.info(f"Applying Fuzzy C-Means clustering with {n_clusters} clusters (m={m})")
                
                # Fuzzy C-Means expects features as rows, samples as columns (transpose)
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data_to_cluster.T,
                    c=n_clusters,
                    m=m,
                    error=error,
                    maxiter=maxiter,
                    init=None
                )
                
                # Store fuzzy memberships (shape: n_clusters x n_samples)
                self.fuzzy_memberships = u
                
                # Get hard cluster assignments (highest membership)
                self.cluster_labels = np.argmax(u, axis=0)
                
                logger.info(f"Fuzzy C-Means completed with FPC={fpc:.4f}")
                
            elif method.lower() == "spectral":
                # Spectral clustering parameters
                affinity = kwargs.pop("affinity", "rbf")
                n_neighbors = kwargs.pop("n_neighbors", 10)
                
                if affinity == "nearest_neighbors":
                    self.clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        affinity=affinity,
                        n_neighbors=n_neighbors,
                        **kwargs
                    )
                    logger.info(f"Applying Spectral clustering with {n_clusters} clusters (affinity={affinity}, n_neighbors={n_neighbors})")
                else:
                    self.clusterer = SpectralClustering(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        affinity=affinity,
                        **kwargs
                    )
                    logger.info(f"Applying Spectral clustering with {n_clusters} clusters (affinity={affinity})")
                    
                self.cluster_labels = self.clusterer.fit_predict(data_to_cluster)
                
            else:
                raise ClusteringError(
                    f"Unknown clustering method: {method}. "
                    f"Use 'kmeans', 'dbscan', 'agglomerative', 'fuzzy_cmeans', or 'spectral'."
                )

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

    def _extract_cluster_hierarchy(self) -> None:
        """
        Extract hierarchical cluster structure from agglomerative clustering.
        
        This method extracts the dendrogram information from scikit-learn's
        AgglomerativeClustering to build a hierarchy that can be used for
        hierarchical visualization.
        
        The hierarchy is stored in self.cluster_hierarchy as a dictionary
        mapping cluster IDs to their children and parent information.
        """
        if not isinstance(self.clusterer, AgglomerativeClustering):
            return
            
        if not hasattr(self.clusterer, 'children_'):
            logger.warning("Clusterer does not have children_ attribute, hierarchy not available")
            return
            
        if self.cluster_labels is None:
            logger.warning("Cluster labels not available, cannot extract hierarchy")
            return
            
        try:
            n_samples = len(self.cluster_labels)
            children = self.clusterer.children_
            
            # Build hierarchy dictionary
            # Each merge creates a new cluster node
            merges: List[Dict[str, Any]] = []
            
            # Each row in children represents a merge
            # children[i] = [left, right] where left and right are indices
            # Indices < n_samples are original samples
            # Indices >= n_samples are merged clusters (index - n_samples gives merge step)
            for i, (left, right) in enumerate(children):
                merge_info: Dict[str, Any] = {
                    'merge_id': i,
                    'left': int(left),
                    'right': int(right),
                    'cluster_id': n_samples + i  # New cluster ID
                }
                
                # Add distance if available
                if hasattr(self.clusterer, 'distances_'):
                    merge_info['distance'] = float(self.clusterer.distances_[i])
                    
                merges.append(merge_info)
            
            # Build tree structure with levels
            tree = self._build_hierarchy_tree(n_samples, children)
            
            self.cluster_hierarchy = {
                'n_samples': n_samples,
                'n_clusters': len(np.unique(self.cluster_labels)),
                'merges': merges,
                'tree': tree
            }
            logger.info(f"Extracted hierarchy with {len(children)} merges")
            
        except Exception as e:
            logger.warning(f"Failed to extract cluster hierarchy: {e}")
            self.cluster_hierarchy = None
    
    def _build_hierarchy_tree(self, n_samples: int, children: np.ndarray) -> Dict[str, Any]:
        """
        Build a tree structure from agglomerative clustering merges.
        
        Parameters
        ----------
        n_samples : int
            Number of original samples
        children : np.ndarray
            Children array from AgglomerativeClustering
            
        Returns
        -------
        dict
            Tree structure with nodes and their relationships
        """
        # Build node information
        nodes = {}
        
        # Leaf nodes (original samples)
        for i in range(n_samples):
            nodes[i] = {
                'node_id': i,
                'is_leaf': True,
                'children': [],
                'samples': [i],
                'level': 0
            }
        
        # Internal nodes (merges)
        for i, (left, right) in enumerate(children):
            node_id = n_samples + i
            left_node = nodes[int(left)]
            right_node = nodes[int(right)]
            
            left_samples: List[int] = left_node['samples']  # type: ignore
            right_samples: List[int] = right_node['samples']  # type: ignore
            left_level: int = left_node['level']  # type: ignore
            right_level: int = right_node['level']  # type: ignore
            
            nodes[node_id] = {
                'node_id': node_id,
                'is_leaf': False,
                'children': [int(left), int(right)],
                'samples': left_samples + right_samples,
                'level': max(left_level, right_level) + 1
            }
        
        # Root is the last merge
        root_id = n_samples + len(children) - 1
        
        return {
            'nodes': {k: v for k, v in nodes.items()},
            'root': root_id,
            'max_level': nodes[root_id]['level']
        }
    
    def get_hierarchy_level_clusters(self, level: int = 0) -> Dict[str, Any]:
        """
        Get clusters at a specific hierarchy level for agglomerative clustering.
        
        Parameters
        ----------
        level : int, optional
            Hierarchy level (0 = leaf level, higher = more merged), by default 0
            
        Returns
        -------
        dict
            Dictionary containing:
            - clusters: List of cluster information at the level
            - level: The requested level
            - max_level: Maximum available level
            
        Raises
        ------
        ClusteringError
            If hierarchy not available
        """
        if self.cluster_hierarchy is None or 'tree' not in self.cluster_hierarchy:
            raise ClusteringError("Cluster hierarchy not available. Use agglomerative clustering.")
        
        tree = self.cluster_hierarchy['tree']
        max_level = tree['max_level']
        
        # Clamp level to valid range
        level = max(0, min(level, max_level))
        
        # Find all nodes at the requested level
        clusters_at_level = []
        for node_id, node_info in tree['nodes'].items():
            if node_info['level'] == level:
                clusters_at_level.append({
                    'cluster_id': node_id,
                    'node_id': node_id,
                    'samples': node_info['samples'],
                    'is_leaf': node_info['is_leaf'],
                    'children': node_info['children'],
                    'size': len(node_info['samples'])
                })
        
        return {
            'clusters': clusters_at_level,
            'level': level,
            'max_level': max_level
        }
    
    def generate_hierarchical_labels(
        self,
        use_llm: bool = True,
        max_keywords: int = 5,
    ) -> Dict[int, str]:
        """
        Generate labels for all levels of the hierarchy.
        
        For leaf clusters, uses existing cluster labels or keywords.
        For parent clusters, generates labels by summarizing child labels using LLM.
        
        Parameters
        ----------
        use_llm : bool, optional
            Whether to use LLM for label generation, by default True
        max_keywords : int, optional
            Maximum number of keywords to use in label generation, by default 5
            
        Returns
        -------
        Dict[int, str]
            Dictionary mapping node IDs to labels
            
        Raises
        ------
        ClusteringError
            If hierarchy not available
        """
        if self.cluster_hierarchy is None or 'tree' not in self.cluster_hierarchy:
            raise ClusteringError("Cluster hierarchy not available. Use agglomerative clustering.")
        
        # First generate labels for leaf clusters if not already done
        if self.cluster_label_names is None:
            self.generate_cluster_labels(use_llm=use_llm, max_keywords=max_keywords)
        
        tree = self.cluster_hierarchy['tree']
        hierarchical_labels = {}
        
        # Start with leaf labels (map sample indices to cluster labels)
        n_samples = self.cluster_hierarchy['n_samples']
        for i in range(n_samples):
            if self.cluster_labels is not None:
                cluster_id = int(self.cluster_labels[i])
                if self.cluster_label_names and cluster_id in self.cluster_label_names:
                    hierarchical_labels[i] = self.cluster_label_names[cluster_id]
                else:
                    hierarchical_labels[i] = f"Sample {i}"
        
        # Generate labels for internal nodes bottom-up
        for level in range(1, tree['max_level'] + 1):
            for node_id, node_info in tree['nodes'].items():
                if node_info['level'] == level:
                    child_labels = [
                        hierarchical_labels.get(child, f"Node {child}")
                        for child in node_info['children']
                    ]
                    
                    if use_llm and self.embeddings_manager:
                        try:
                            label = self._generate_parent_label_llm(child_labels, node_info['samples'])
                            hierarchical_labels[node_id] = label
                        except Exception as e:
                            logger.warning(f"LLM label generation failed for node {node_id}: {e}")
                            hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
                    else:
                        hierarchical_labels[node_id] = self._generate_parent_label_fallback(child_labels)
        
        return hierarchical_labels
    
    def _generate_parent_label_llm(self, child_labels: List[str], sample_indices: List[int]) -> str:
        """
        Generate a parent cluster label by summarizing child labels using LLM.
        
        Parameters
        ----------
        child_labels : List[str]
            Labels of child clusters
        sample_indices : List[int]
            Indices of samples in this parent cluster
            
        Returns
        -------
        str
            Generated parent label
        """
        from .config import get_config
        config = get_config()
        
        # Get sample titles from the parent cluster
        sample_titles = []
        if self.metadatas and len(sample_indices) > 0:
            sample_size = min(5, len(sample_indices))
            sampled_indices = np.random.choice(sample_indices, size=sample_size, replace=False)
            for idx in sampled_indices:
                title = self.metadatas[idx].get('title', '')
                if title:
                    sample_titles.append(title)
        
        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles) if sample_titles else "N/A"
        child_labels_str = "\n".join(f"- {label}" for label in child_labels)
        
        prompt = f"""Given a parent cluster that contains the following sub-clusters:

{child_labels_str}

Sample paper titles from this parent cluster:
{sample_titles_str}

Generate a concise, descriptive label (3-5 words) that captures the overarching theme of this parent cluster.
The label should generalize the themes of the child clusters.
Only respond with the label, nothing else. Do not add formatting."""

        try:
            if not hasattr(self.embeddings_manager, 'openai_client'):
                raise AttributeError("OpenAI client not available")
            
            response = self.embeddings_manager.openai_client.chat.completions.create(
                model=config.chat_model,
                messages=[
                    {"role": "system", "content": "You are a research paper categorization expert. Generate concise labels that generalize child cluster themes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            label = response.choices[0].message.content.strip()
            label = label.strip('"\'')
            return label
            
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return self._generate_parent_label_fallback(child_labels)
    
    def _generate_parent_label_fallback(self, child_labels: List[str]) -> str:
        """
        Generate a parent label by combining child labels (fallback method).
        
        Parameters
        ----------
        child_labels : List[str]
            Labels of child clusters
            
        Returns
        -------
        str
            Generated parent label
        """
        # Simple concatenation with "&" separator
        return " & ".join(child_labels[:3])

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

    def extract_cluster_keywords(
        self,
        n_keywords: int = 10,
        min_df: int = 2,
    ) -> Dict[int, List[str]]:
        """
        Extract distinctive keywords for each cluster using TF-IDF.

        Parameters
        ----------
        n_keywords : int, optional
            Number of top keywords to extract per cluster, by default 10
        min_df : int, optional
            Minimum document frequency for a term to be considered, by default 2

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of keywords

        Raises
        ------
        ClusteringError
            If clustering has not been performed or metadata is missing

        Examples
        --------
        >>> cm = ClusteringManager(em)
        >>> cm.load_embeddings()
        >>> cm.cluster(method='kmeans', n_clusters=5)
        >>> keywords = cm.extract_cluster_keywords(n_keywords=10)
        >>> print(f"Cluster 0 keywords: {keywords[0]}")
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if self.metadatas is None:
            raise ClusteringError("No metadata available.")

        try:
            # Get unique cluster labels (excluding noise)
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            self.cluster_keywords = {}

            for cluster_id in cluster_ids:
                # Get indices of papers in this cluster
                cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

                # Collect documents (titles and abstracts) for this cluster
                cluster_docs = []
                for idx in cluster_indices:
                    doc_text = EmbeddingsManager.embedding_text_from_paper(self.metadatas[idx])
                    cluster_docs.append(doc_text)

                if not cluster_docs:
                    logger.warning(f"No documents found for cluster {cluster_id}")
                    self.cluster_keywords[cluster_id] = []
                    continue

                # Use TF-IDF to extract keywords
                # Compare cluster documents against all documents to find distinctive terms
                all_docs = []
                for metadata in self.metadatas:
                    doc_text = EmbeddingsManager.embedding_text_from_paper(metadata)
                    all_docs.append(doc_text)

                # Fit TF-IDF on all documents
                tfidf = TfidfVectorizer(
                    max_features=1000,
                    min_df=min_df,
                    stop_words='english',
                    ngram_range=(1, 2)  # Include unigrams and bigrams
                )
                tfidf_matrix = tfidf.fit_transform(all_docs)
                feature_names = tfidf.get_feature_names_out()

                # Calculate mean TF-IDF for cluster documents
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1

                # Get top keywords by TF-IDF score
                top_indices = cluster_tfidf.argsort()[-n_keywords:][::-1]
                keywords = [feature_names[i] for i in top_indices if cluster_tfidf[i] > 0]

                self.cluster_keywords[cluster_id] = keywords[:n_keywords]
                logger.debug(f"Cluster {cluster_id}: {len(keywords)} keywords extracted")

            logger.info(f"Extracted keywords for {len(self.cluster_keywords)} clusters")
            return self.cluster_keywords

        except Exception as e:
            raise ClusteringError(f"Failed to extract cluster keywords: {str(e)}") from e

    def generate_cluster_labels(
        self,
        use_llm: bool = True,
        max_keywords: int = 5,
    ) -> Dict[int, str]:
        """
        Generate descriptive labels for clusters.

        This method can either use an LLM to generate meaningful labels based on
        cluster keywords and representative papers, or simply concatenate keywords.

        Parameters
        ----------
        use_llm : bool, optional
            Whether to use LLM for label generation, by default True
        max_keywords : int, optional
            Maximum number of keywords to use in label generation, by default 5

        Returns
        -------
        Dict[int, str]
            Dictionary mapping cluster labels to descriptive names

        Raises
        ------
        ClusteringError
            If clustering or keyword extraction has not been performed

        Examples
        --------
        >>> cm = ClusteringManager(em)
        >>> cm.load_embeddings()
        >>> cm.cluster(method='kmeans', n_clusters=5)
        >>> cm.extract_cluster_keywords()
        >>> labels = cm.generate_cluster_labels(use_llm=True)
        >>> print(f"Cluster 0 label: {labels[0]}")
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")

        # Extract keywords if not already done
        if self.cluster_keywords is None:
            logger.info("Extracting cluster keywords first...")
            self.extract_cluster_keywords()

        try:
            self.cluster_label_names = {}
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            for cluster_id in cluster_ids:
                keywords = (self.cluster_keywords or {}).get(cluster_id, [])[:max_keywords]

                if not keywords:
                    self.cluster_label_names[cluster_id] = f"Cluster {cluster_id}"
                    continue

                if use_llm and self.embeddings_manager:
                    try:
                        # Generate label using LLM
                        label = self._generate_llm_label(cluster_id, keywords)
                        self.cluster_label_names[cluster_id] = label
                    except Exception as e:
                        logger.warning(f"LLM label generation failed for cluster {cluster_id}: {e}")
                        # Fallback to keyword-based label
                        self.cluster_label_names[cluster_id] = ", ".join(keywords[:3])
                else:
                    # Use keyword-based label
                    self.cluster_label_names[cluster_id] = ", ".join(keywords[:3])

            logger.info(f"Generated labels for {len(self.cluster_label_names)} clusters")
            return self.cluster_label_names

        except Exception as e:
            raise ClusteringError(f"Failed to generate cluster labels: {str(e)}") from e

    def _generate_llm_label(self, cluster_id: int, keywords: List[str]) -> str:
        """
        Generate a cluster label using LLM.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier
        keywords : List[str]
            List of keywords for the cluster

        Returns
        -------
        str
            Generated label
        """
        # Get a few representative paper titles from the cluster
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        sample_size = min(5, len(cluster_indices))

        # Use replacement if there are fewer papers than sample size
        replace = len(cluster_indices) < sample_size
        sample_indices = np.random.choice(
            cluster_indices, 
            size=sample_size, 
            replace=replace
        )

        sample_titles = []
        if self.metadatas:
            for idx in sample_indices:
                title = self.metadatas[idx].get('title', '')
                if title:
                    sample_titles.append(title)

        # Construct prompt for LLM
        sample_titles_str = "\n".join(f"- {title}" for title in sample_titles)
        prompt = f"""Given a cluster of research papers with the following characteristics:

Top keywords: {', '.join(keywords)}

Sample paper titles:
{sample_titles_str}

Generate a concise, descriptive label (3-5 words) that captures the main theme of this cluster. 
Only respond with the label, nothing else. Do not add formatting."""

        try:
            # Check if OpenAI client is available
            if not hasattr(self.embeddings_manager, 'openai_client'):
                raise AttributeError("OpenAI client not available in embeddings manager")

            # Use the embeddings manager's OpenAI client
            from .config import get_config
            config = get_config()

            response = self.embeddings_manager.openai_client.chat.completions.create(
                model=config.chat_model,
                messages=[
                    {"role": "system", "content": "You are a research paper categorization expert. Generate concise, descriptive labels for clusters of papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            label = response.choices[0].message.content.strip()
            # Remove quotes if present
            label = label.strip('"\'')
            return label

        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            # Fallback to keyword-based label
            return ", ".join(keywords[:3])

    def get_cluster_representative_papers(
        self,
        n_papers: int = 5,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Find representative papers for each cluster.

        Representative papers are those closest to the cluster centroid
        in the embedding space.

        Parameters
        ----------
        n_papers : int, optional
            Number of representative papers per cluster, by default 5

        Returns
        -------
        Dict[int, List[Dict[str, Any]]]
            Dictionary mapping cluster labels to lists of representative paper metadata

        Raises
        ------
        ClusteringError
            If clustering has not been performed

        Examples
        --------
        >>> cm = ClusteringManager(em)
        >>> cm.load_embeddings()
        >>> cm.cluster(method='kmeans', n_clusters=5)
        >>> representatives = cm.get_cluster_representative_papers(n_papers=3)
        >>> print(f"Cluster 0 representatives: {representatives[0]}")
        """
        if self.cluster_labels is None:
            raise ClusteringError("No clustering performed. Call cluster() first.")
        if self.embeddings is None:
            raise ClusteringError("No embeddings loaded.")

        try:
            representatives: Dict[int, List[Dict[str, Any]]] = {}
            unique_labels = np.unique(self.cluster_labels)
            cluster_ids = [int(label) for label in unique_labels if label >= 0]

            for cluster_id in cluster_ids:
                # Get indices of papers in this cluster
                cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

                if len(cluster_indices) == 0:
                    representatives[cluster_id] = []
                    continue

                # Get embeddings for this cluster
                cluster_embeddings = self.embeddings[cluster_indices]

                # Calculate cluster centroid
                centroid = cluster_embeddings.mean(axis=0)

                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

                # Get indices of papers closest to centroid
                n_repr = min(n_papers, len(cluster_indices))
                closest_indices = distances.argsort()[:n_repr]

                # Collect representative paper metadata
                repr_papers = []
                if self.metadatas and self.paper_ids:
                    for idx in closest_indices:
                        paper_idx = cluster_indices[idx]
                        paper_meta = self.metadatas[paper_idx].copy()
                        paper_meta['paper_id'] = self.paper_ids[paper_idx]
                        paper_meta['distance_to_centroid'] = float(distances[idx])
                        repr_papers.append(paper_meta)

                representatives[cluster_id] = repr_papers
                logger.debug(f"Found {len(repr_papers)} representative papers for cluster {cluster_id}")

            logger.info(f"Found representative papers for {len(representatives)} clusters")
            return representatives

        except Exception as e:
            raise ClusteringError(f"Failed to find representative papers: {str(e)}") from e

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
                
            # Add fuzzy memberships if available (for fuzzy c-means)
            if self.fuzzy_memberships is not None:
                # Convert to list format for JSON serialization
                results["fuzzy_memberships"] = self.fuzzy_memberships.tolist()

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
    collection_name: str = "papers",
    reduction_method: str = "pca",
    n_components: int = 2,
    clustering_method: str = "kmeans",
    n_clusters: Optional[int] = None,
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
    collection_name : str, optional
        Name of the ChromaDB collection, by default "papers"
    reduction_method : str, optional
        Dimensionality reduction method ('pca', 'tsne', or 'umap') for visualization, by default 'pca'
    n_components : int, optional
        Number of components for dimensionality reduction, by default 2
    clustering_method : str, optional
        Clustering method ('kmeans', 'dbscan', 'agglomerative', 'fuzzy_cmeans', or 'spectral'), by default 'kmeans'
    n_clusters : int, optional
        Number of clusters (for kmeans and agglomerative).
        If None, automatically calculated as n_papers / 100, clamped to [2, 500].
        By default None.
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
    ...     reduction_method="tsne",
    ...     clustering_method="kmeans",
    ...     n_clusters=5,
    ...     output_path="clusters.json"
    ... )
    >>> print(f"Found {results['statistics']['n_clusters']} clusters")
    """
    try:
        # Initialize embeddings manager - gets config from environment
        em = EmbeddingsManager(
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

        # Generate cluster labels
        logger.info("Generating cluster labels...")
        try:
            cm.extract_cluster_keywords(n_keywords=10)
            cm.generate_cluster_labels(use_llm=True, max_keywords=5)
        except Exception as e:
            logger.warning(f"Failed to generate cluster labels: {e}")
            # Continue without labels

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


def compute_clusters_with_cache(
    embeddings_manager: EmbeddingsManager,
    database: DatabaseManager,
    embedding_model: str,
    reduction_method: str = "pca",
    n_components: int = 2,
    clustering_method: str = "kmeans",
    n_clusters: Optional[int] = None,
    limit: Optional[int] = None,
    force: bool = False,
    **clustering_kwargs
) -> Dict[str, Any]:
    """
    Compute clusters with caching support.
    
    This function checks the cache first and returns cached results if available.
    If cache miss or forced recompute, it performs clustering and saves to cache.
    
    Parameters
    ----------
    embeddings_manager : EmbeddingsManager
        Embeddings manager instance
    database : DatabaseManager
        Database manager for cache operations
    embedding_model : str
        Current embedding model name
    reduction_method : str, optional
        Dimensionality reduction method, by default "pca"
    n_components : int, optional
        Number of components for reduction, by default 2
    clustering_method : str, optional
        Clustering method to use, by default "kmeans"
    n_clusters : int, optional
        Number of clusters. If None, auto-calculated based on data size
    limit : int, optional
        Maximum number of embeddings to process
    force : bool, optional
        Force recompute even if cache exists, by default False
    **clustering_kwargs
        Additional clustering parameters (e.g., eps, min_samples for DBSCAN)
        
    Returns
    -------
    dict
        Clustering results with points, statistics, and metadata
        
    Raises
    ------
    ClusteringError
        If clustering fails
        
    Examples
    --------
    >>> results = compute_clusters_with_cache(
    ...     em, db, "text-embedding-model",
    ...     clustering_method="kmeans",
    ...     n_clusters=5
    ... )
    """
    # Get embeddings count to calculate default n_clusters if needed
    collection_stats = embeddings_manager.get_collection_stats()
    n_papers = collection_stats["count"]
    
    # Calculate default n_clusters if not provided
    if n_clusters is None:
        n_clusters = calculate_default_clusters(n_papers)
        logger.info(f"Auto-calculated n_clusters={n_clusters} based on {n_papers} papers")
    
    # Check if cache exists and is valid
    if not force and not limit:  # Only use cache if not limiting results
        # For agglomerative with distance_threshold, don't pass n_clusters
        cache_n_clusters: Optional[int] = n_clusters
        cache_params = clustering_kwargs.copy() if clustering_kwargs else {}
        
        # Special handling for agglomerative with distance_threshold
        if clustering_method.lower() == "agglomerative" and "distance_threshold" in cache_params:
            cache_n_clusters = None  # Don't use n_clusters as cache key when using distance_threshold
        elif clustering_method.lower() == "dbscan":
            cache_n_clusters = None  # DBSCAN doesn't use n_clusters
        
        cached_results = database.get_clustering_cache(
            embedding_model=embedding_model,
            reduction_method=reduction_method,
            n_components=n_components,
            clustering_method=clustering_method,
            n_clusters=cache_n_clusters,
            clustering_params=cache_params if cache_params else None,
        )
        
        if cached_results:
            logger.info("Using cached clustering results")
            return cached_results
    
    # Cache miss or forced recompute - compute clusters
    logger.info("Computing new clustering results...")
    
    # Create clustering manager
    cm = ClusteringManager(embeddings_manager)
    
    # Load embeddings
    logger.info(f"Loading embeddings (limit={limit})...")
    cm.load_embeddings(limit=limit)
    
    # Perform clustering on full embeddings first
    logger.info(f"Clustering using {clustering_method} on full embeddings...")
    cm.cluster(
        method=clustering_method,
        n_clusters=n_clusters,
        use_reduced=False,  # Cluster on full embeddings
        **clustering_kwargs
    )
    
    # Reduce dimensions for visualization
    logger.info(f"Reducing dimensions using {reduction_method} for visualization...")
    cm.reduce_dimensions(
        method=reduction_method,
        n_components=n_components,
    )
    
    # Generate cluster labels
    logger.info("Generating cluster labels...")
    try:
        cm.extract_cluster_keywords(n_keywords=10)
        cm.generate_cluster_labels(use_llm=True, max_keywords=5)
    except Exception as e:
        logger.warning(f"Failed to generate cluster labels: {e}")
        # Continue without labels
    
    # Get results
    results = cm.get_clustering_results()
    
    # Save to cache if no limit was applied
    if not limit:
        try:
            # Use same logic as cache lookup for consistency
            save_n_clusters: Optional[int] = n_clusters
            save_params = clustering_kwargs.copy() if clustering_kwargs else {}
            
            # Special handling for agglomerative with distance_threshold
            if clustering_method.lower() == "agglomerative" and "distance_threshold" in save_params:
                save_n_clusters = None  # Don't use n_clusters as cache key when using distance_threshold
            elif clustering_method.lower() == "dbscan":
                save_n_clusters = None  # DBSCAN doesn't use n_clusters
            
            database.save_clustering_cache(
                embedding_model=embedding_model,
                reduction_method=reduction_method,
                n_components=n_components,
                clustering_method=clustering_method,
                results=results,
                n_clusters=save_n_clusters,
                clustering_params=save_params if save_params else None,
            )
        except Exception as e:
            logger.warning(f"Failed to save clustering cache: {e}")
    
    return results
