"""
Tests for clustering module.

This module tests the clustering and dimensionality reduction functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from abstracts_explorer.clustering import (
    ClusteringManager,
    ClusteringError,
    calculate_default_clusters,
)


class TestCalculateDefaultClusters:
    """Test suite for calculate_default_clusters function."""
    
    def test_calculate_default_clusters_small(self):
        """Test calculation with small number of papers."""
        # Less than 100 papers should give minimum clusters
        assert calculate_default_clusters(50) == 2
        assert calculate_default_clusters(100) == 2
        assert calculate_default_clusters(199) == 2
    
    def test_calculate_default_clusters_medium(self):
        """Test calculation with medium number of papers."""
        assert calculate_default_clusters(500) == 5
        assert calculate_default_clusters(1000) == 10
        assert calculate_default_clusters(2500) == 25
    
    def test_calculate_default_clusters_large(self):
        """Test calculation with large number of papers."""
        # Should be capped at max_clusters (default 50)
        assert calculate_default_clusters(50000) == 500
        assert calculate_default_clusters(100000) == 500
        assert calculate_default_clusters(1000000) == 500
    
    def test_calculate_default_clusters_custom_limits(self):
        """Test calculation with custom min/max limits."""
        assert calculate_default_clusters(500, min_clusters=5, max_clusters=20) == 5
        assert calculate_default_clusters(3000, min_clusters=5, max_clusters=20) == 20
    
    def test_calculate_default_clusters_edge_cases(self):
        """Test edge cases."""
        assert calculate_default_clusters(0) == 2
        assert calculate_default_clusters(-10) == 2


class TestClusteringManager:
    """Test suite for ClusteringManager class."""

    def test_init(self, mock_embeddings_manager):
        """Test ClusteringManager initialization."""
        cm = ClusteringManager(mock_embeddings_manager)
        
        assert cm.embeddings_manager is mock_embeddings_manager
        assert cm.database is None
        assert cm.embeddings is None
        assert cm.paper_ids is None
        assert cm.metadatas is None

    def test_load_embeddings_no_collection(self, mock_embeddings_manager):
        """Test loading embeddings fails without collection."""
        mock_embeddings_manager.collection = None
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="Collection not initialized"):
            cm.load_embeddings()

    def test_load_embeddings_empty_collection(self, mock_embeddings_manager, mock_collection_empty):
        """Test loading embeddings from empty collection."""
        mock_embeddings_manager.collection = mock_collection_empty
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="No embeddings found"):
            cm.load_embeddings()

    def test_load_embeddings_success(self, mock_embeddings_manager, mock_collection_with_data):
        """Test successfully loading embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        
        count = cm.load_embeddings()
        
        assert count == 10
        assert cm.embeddings.shape == (10, 128)
        assert len(cm.paper_ids) == 10
        assert len(cm.metadatas) == 10

    def test_reduce_dimensions_no_embeddings(self, mock_embeddings_manager):
        """Test dimensionality reduction fails without embeddings."""
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="No embeddings loaded"):
            cm.reduce_dimensions()

    def test_reduce_dimensions_pca(self, mock_embeddings_manager, mock_collection_with_data):
        """Test PCA dimensionality reduction."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        reduced = cm.reduce_dimensions(method='pca', n_components=2)
        
        assert reduced.shape == (10, 2)
        assert cm.reduced_embeddings is not None
        assert cm.reduced_embeddings.shape == (10, 2)

    def test_reduce_dimensions_tsne(self, mock_embeddings_manager, mock_collection_with_data):
        """Test t-SNE dimensionality reduction."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        reduced = cm.reduce_dimensions(method='tsne', n_components=2)
        
        assert reduced.shape == (10, 2)
        assert cm.reduced_embeddings is not None

    def test_reduce_dimensions_umap(self, mock_embeddings_manager, mock_collection_with_data):
        """Test UMAP dimensionality reduction."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        reduced = cm.reduce_dimensions(method='umap', n_components=2)
        
        assert reduced.shape == (10, 2)
        assert cm.reduced_embeddings is not None

    def test_reduce_dimensions_umap_insufficient_samples(self, mock_embeddings_manager):
        """Test UMAP with insufficient samples raises appropriate error."""
        # Create a mock collection with only 2 samples (need at least 4 for 2 components)
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': ['id1', 'id2'],
            'embeddings': np.random.randn(2, 128).tolist(),
            'metadatas': [{'title': 'Paper 1'}, {'title': 'Paper 2'}]
        }
        mock_embeddings_manager.collection = mock_collection
        
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        with pytest.raises(ClusteringError, match="UMAP requires at least 4 samples"):
            cm.reduce_dimensions(method='umap', n_components=2)

    def test_reduce_dimensions_invalid_method(self, mock_embeddings_manager, mock_collection_with_data):
        """Test invalid dimensionality reduction method."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        with pytest.raises(ClusteringError, match="Unknown reduction method"):
            cm.reduce_dimensions(method='invalid')

    def test_cluster_no_embeddings(self, mock_embeddings_manager):
        """Test clustering fails without embeddings."""
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="No embeddings loaded"):
            cm.cluster()

    def test_cluster_auto_calculate_n_clusters(self, mock_embeddings_manager, mock_collection_with_data):
        """Test automatic calculation of n_clusters when None is provided."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Cluster with n_clusters=None should auto-calculate
        labels = cm.cluster(method='kmeans', n_clusters=None, use_reduced=False)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None
        # With 10 papers, should calculate 2 clusters (10 / 100 = 0, min is 2)
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 2

    def test_cluster_kmeans(self, mock_embeddings_manager, mock_collection_with_data):
        """Test K-Means clustering on full embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Cluster on full embeddings (not reduced)
        labels = cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None
        # Check that we have valid cluster labels
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 3
        assert np.all(labels >= 0)

    def test_cluster_dbscan(self, mock_embeddings_manager, mock_collection_with_data):
        """Test DBSCAN clustering on full embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Cluster on full embeddings (not reduced)
        labels = cm.cluster(method='dbscan', eps=0.5, min_samples=2, use_reduced=False)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None

    def test_cluster_agglomerative(self, mock_embeddings_manager, mock_collection_with_data):
        """Test Agglomerative clustering on full embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Cluster on full embeddings (not reduced)
        labels = cm.cluster(method='agglomerative', n_clusters=3, use_reduced=False)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None

    def test_cluster_invalid_method(self, mock_embeddings_manager, mock_collection_with_data):
        """Test invalid clustering method."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        with pytest.raises(ClusteringError, match="Unknown clustering method"):
            cm.cluster(method='invalid')

    def test_get_cluster_statistics(self, mock_embeddings_manager, mock_collection_with_data):
        """Test getting cluster statistics."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        # Cluster first on full embeddings
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        stats = cm.get_cluster_statistics()
        
        assert 'n_clusters' in stats
        assert 'n_noise' in stats
        assert 'cluster_sizes' in stats
        assert 'total_papers' in stats
        assert stats['total_papers'] == 10

    def test_get_clustering_results(self, mock_embeddings_manager, mock_collection_with_data):
        """Test getting complete clustering results."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        # Cluster first, then reduce for visualization
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(method='pca', n_components=2)
        
        results = cm.get_clustering_results()
        
        assert 'points' in results
        assert 'statistics' in results
        assert 'n_dimensions' in results
        assert 'cluster_centers' in results
        assert len(results['points']) == 10
        assert results['n_dimensions'] == 2
        
        # Verify cluster centers are calculated correctly
        assert isinstance(results['cluster_centers'], dict)
        # Should have centers for each non-noise cluster
        n_clusters = results['statistics']['n_clusters']
        assert len(results['cluster_centers']) <= n_clusters
        
        # Each center should have x and y coordinates
        for cluster_id, center in results['cluster_centers'].items():
            assert 'x' in center
            assert 'y' in center
            assert isinstance(center['x'], (int, float))
            assert isinstance(center['y'], (int, float))

    def test_export_to_json(self, mock_embeddings_manager, mock_collection_with_data, tmp_path):
        """Test exporting clustering results to JSON."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        # Cluster first, then reduce for visualization
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(method='pca', n_components=2)
        
        output_path = tmp_path / "clusters.json"
        cm.export_to_json(output_path)
        
        assert output_path.exists()
        
        # Verify JSON can be loaded
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'points' in data
        assert 'statistics' in data


# Fixtures

@pytest.fixture
def mock_embeddings_manager(mocker):
    """Create a mock EmbeddingsManager."""
    em = mocker.MagicMock()
    em.collection = None
    return em


@pytest.fixture
def mock_collection_empty(mocker):
    """Create a mock empty ChromaDB collection."""
    collection = mocker.MagicMock()
    collection.get.return_value = {
        'ids': [],
        'embeddings': [],
        'metadatas': []
    }
    return collection


@pytest.fixture
def mock_collection_with_data(mocker):
    """Create a mock ChromaDB collection with test data."""
    collection = mocker.MagicMock()
    
    # Create 10 sample embeddings (128-dimensional)
    embeddings = np.random.randn(10, 128).tolist()
    ids = [f'paper_{i}' for i in range(10)]
    
    # Create more realistic metadata with varied abstracts
    abstracts = [
        "This paper presents a novel deep learning approach for image classification using convolutional neural networks.",
        "We propose a new transformer architecture for natural language processing tasks with improved efficiency.",
        "An analysis of reinforcement learning algorithms for robotic control and autonomous navigation systems.",
        "This work introduces a generative adversarial network for high-resolution image synthesis and style transfer.",
        "We study the application of convolutional neural networks in computer vision and object detection tasks.",
        "A comprehensive survey of attention mechanisms in transformer models for sequence-to-sequence learning.",
        "This paper explores policy gradient methods in reinforcement learning for continuous control problems.",
        "We present a novel architecture combining transformers and attention for machine translation applications.",
        "An investigation of deep Q-networks and actor-critic methods for game playing and decision making.",
        "This work proposes improvements to generative models for realistic image generation and manipulation."
    ]
    
    metadatas = [
        {
            'title': f'Paper {i}',
            'abstract': abstracts[i],
            'year': '2025',
            'conference': 'TestConf',
            'session': 'Session A'
        }
        for i in range(10)
    ]
    
    collection.get.return_value = {
        'ids': ids,
        'embeddings': embeddings,
        'metadatas': metadatas
    }
    
    return collection


class TestClusterLabeling:
    """Test suite for cluster labeling functionality."""

    def test_extract_cluster_keywords(self, mock_embeddings_manager, mock_collection_with_data):
        """Test extracting keywords for each cluster."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        keywords = cm.extract_cluster_keywords(n_keywords=5)
        
        assert isinstance(keywords, dict)
        assert len(keywords) > 0
        # Check that each cluster has keywords
        for cluster_id, kw_list in keywords.items():
            assert isinstance(kw_list, list)
            assert len(kw_list) <= 5

    def test_extract_cluster_keywords_no_clustering(self, mock_embeddings_manager):
        """Test that keyword extraction fails without clustering."""
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="No clustering performed"):
            cm.extract_cluster_keywords()

    def test_generate_cluster_labels_keyword_based(self, mock_embeddings_manager, mock_collection_with_data):
        """Test generating cluster labels without LLM."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        labels = cm.generate_cluster_labels(use_llm=False)
        
        assert isinstance(labels, dict)
        assert len(labels) > 0
        # Check that each cluster has a label
        for cluster_id, label in labels.items():
            assert isinstance(label, str)
            assert len(label) > 0

    def test_generate_cluster_labels_with_llm(self, mock_embeddings_manager, mock_collection_with_data, mocker):
        """Test generating cluster labels with LLM."""
        mock_embeddings_manager.collection = mock_collection_with_data
        
        # Mock the OpenAI client response
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Machine Learning Models"
        
        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_embeddings_manager.openai_client = mock_openai_client
        
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        labels = cm.generate_cluster_labels(use_llm=True)
        
        assert isinstance(labels, dict)
        assert len(labels) > 0
        # At least one cluster should have the LLM-generated label
        assert any("Machine Learning" in label for label in labels.values())

    def test_get_cluster_representative_papers(self, mock_embeddings_manager, mock_collection_with_data):
        """Test finding representative papers for each cluster."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        
        representatives = cm.get_cluster_representative_papers(n_papers=3)
        
        assert isinstance(representatives, dict)
        assert len(representatives) > 0
        # Check that each cluster has representative papers
        for cluster_id, papers in representatives.items():
            assert isinstance(papers, list)
            assert len(papers) <= 3
            # Check that each paper has required fields
            for paper in papers:
                assert 'paper_id' in paper
                assert 'distance_to_centroid' in paper
                assert 'title' in paper

    def test_get_cluster_representative_papers_no_clustering(self, mock_embeddings_manager):
        """Test that finding representatives fails without clustering."""
        cm = ClusteringManager(mock_embeddings_manager)
        
        with pytest.raises(ClusteringError, match="No clustering performed"):
            cm.get_cluster_representative_papers()

    def test_clustering_results_with_labels(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that clustering results include labels when available."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(method='pca', n_components=2)
        cm.extract_cluster_keywords(n_keywords=5)
        cm.generate_cluster_labels(use_llm=False)
        
        results = cm.get_clustering_results()
        
        assert 'cluster_labels' in results
        assert 'cluster_keywords' in results
        assert isinstance(results['cluster_labels'], dict)
        assert isinstance(results['cluster_keywords'], dict)


class TestNewClusteringMethods:
    """Test suite for new clustering methods: spectral, fuzzy c-means, and enhanced agglomerative."""

    def test_spectral_clustering_basic(self, mock_embeddings_manager, mock_collection_with_data):
        """Test basic spectral clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        labels = cm.cluster(method='spectral', n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 10
        assert len(np.unique(labels)) <= 3
        
    def test_spectral_clustering_with_affinity(self, mock_embeddings_manager, mock_collection_with_data):
        """Test spectral clustering with different affinity parameters."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Test with nearest_neighbors affinity
        labels = cm.cluster(method='spectral', n_clusters=3, affinity='nearest_neighbors', n_neighbors=5)
        
        assert labels is not None
        assert len(labels) == 10
        
    def test_spectral_clustering_requires_n_clusters(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that spectral clustering requires n_clusters to be specified."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Should use auto-calculated n_clusters when None is provided
        labels = cm.cluster(method='spectral', n_clusters=None)
        
        assert labels is not None
        assert len(labels) == 10
        
    def test_fuzzy_cmeans_basic(self, mock_embeddings_manager, mock_collection_with_data):
        """Test basic fuzzy c-means clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        labels = cm.cluster(method='fuzzy_cmeans', n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 10
        assert len(np.unique(labels)) <= 3
        # Check that fuzzy memberships are stored
        assert cm.fuzzy_memberships is not None
        assert cm.fuzzy_memberships.shape == (3, 10)  # n_clusters x n_samples
        
    def test_fuzzy_cmeans_with_parameters(self, mock_embeddings_manager, mock_collection_with_data):
        """Test fuzzy c-means with custom fuzziness parameter."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        labels = cm.cluster(method='fuzzy_cmeans', n_clusters=3, m=2.5, maxiter=100)
        
        assert labels is not None
        assert len(labels) == 10
        assert cm.fuzzy_memberships is not None
        
    def test_fuzzy_cmeans_in_results(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that fuzzy memberships are included in clustering results."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='fuzzy_cmeans', n_clusters=3)
        cm.reduce_dimensions(method='pca', n_components=2)
        
        results = cm.get_clustering_results()
        
        assert 'fuzzy_memberships' in results
        assert isinstance(results['fuzzy_memberships'], list)
        
    def test_agglomerative_with_distance_threshold(self, mock_embeddings_manager, mock_collection_with_data):
        """Test agglomerative clustering with distance threshold instead of n_clusters."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        labels = cm.cluster(method='agglomerative', distance_threshold=5.0, n_clusters=None)
        
        assert labels is not None
        assert len(labels) == 10
        # Number of clusters should be determined by distance threshold
        n_clusters_found = len(np.unique(labels))
        assert n_clusters_found >= 1
        
    def test_agglomerative_hierarchy_extraction(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that hierarchical structure is extracted for agglomerative clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        cm.cluster(method='agglomerative', n_clusters=3)
        
        # Check that hierarchy was extracted
        assert cm.cluster_hierarchy is not None
        assert 'n_samples' in cm.cluster_hierarchy
        assert 'n_clusters' in cm.cluster_hierarchy
        assert 'merges' in cm.cluster_hierarchy
        assert isinstance(cm.cluster_hierarchy['merges'], list)
        # Should have n_samples - 1 merges to form the tree
        assert len(cm.cluster_hierarchy['merges']) == 9  # 10 samples - 1
        
    def test_agglomerative_hierarchy_in_results(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that hierarchy is included in clustering results."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.cluster(method='agglomerative', n_clusters=3)
        cm.reduce_dimensions(method='pca', n_components=2)
        
        results = cm.get_clustering_results()
        
        assert 'cluster_hierarchy' in results
        assert isinstance(results['cluster_hierarchy'], dict)
        assert 'merges' in results['cluster_hierarchy']
        
    def test_agglomerative_with_linkage(self, mock_embeddings_manager, mock_collection_with_data):
        """Test agglomerative clustering with different linkage methods."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Test with different linkage methods
        for linkage in ['ward', 'complete', 'average']:
            cm_test = ClusteringManager(mock_embeddings_manager)
            cm_test.load_embeddings()
            labels = cm_test.cluster(method='agglomerative', n_clusters=3, linkage=linkage)
            
            assert labels is not None
            assert len(labels) == 10
            
    def test_clustering_method_names_case_insensitive(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that clustering method names are case-insensitive."""
        mock_embeddings_manager.collection = mock_collection_with_data
        
        # Test different case variations
        for method in ['SPECTRAL', 'Spectral', 'sPeCtRaL']:
            cm = ClusteringManager(mock_embeddings_manager)
            cm.load_embeddings()
            labels = cm.cluster(method=method, n_clusters=3)
            assert labels is not None
            
    def test_fuzzy_cmeans_alternative_name(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that fuzzy c-means accepts both 'fuzzy_cmeans' and 'fuzzy-cmeans'."""
        mock_embeddings_manager.collection = mock_collection_with_data
        
        # Test with hyphen
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        labels = cm.cluster(method='fuzzy-cmeans', n_clusters=3)
        
        assert labels is not None
        assert cm.fuzzy_memberships is not None
        
    def test_unknown_clustering_method_error(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that unknown clustering methods raise ClusteringError."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        with pytest.raises(ClusteringError, match="Unknown clustering method"):
            cm.cluster(method='invalid_method', n_clusters=3)
            
    def test_spectral_on_reduced_embeddings(self, mock_embeddings_manager, mock_collection_with_data):
        """Test spectral clustering on reduced embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=5)
        
        labels = cm.cluster(method='spectral', n_clusters=3, use_reduced=True)
        
        assert labels is not None
        assert len(labels) == 10
        
    def test_fuzzy_cmeans_on_reduced_embeddings(self, mock_embeddings_manager, mock_collection_with_data):
        """Test fuzzy c-means clustering on reduced embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=5)
        
        labels = cm.cluster(method='fuzzy_cmeans', n_clusters=3, use_reduced=True)
        
        assert labels is not None
        assert len(labels) == 10
        assert cm.fuzzy_memberships is not None


class TestHierarchicalClustering:
    """Test suite for hierarchical clustering visualization features."""
    
    def test_build_hierarchy_tree(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that hierarchy tree is built correctly."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        cm.cluster(method='agglomerative', n_clusters=3)
        
        assert cm.cluster_hierarchy is not None
        assert 'tree' in cm.cluster_hierarchy
        
        tree = cm.cluster_hierarchy['tree']
        assert 'nodes' in tree
        assert 'root' in tree
        assert 'max_level' in tree
        
        # Check that we have leaf and internal nodes
        assert len(tree['nodes']) == 19  # 10 leaves + 9 internal nodes
        
    def test_get_hierarchy_level_clusters(self, mock_embeddings_manager, mock_collection_with_data):
        """Test getting clusters at a specific hierarchy level."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        cm.cluster(method='agglomerative', n_clusters=3)
        
        # Get leaf level (level 0)
        level0 = cm.get_hierarchy_level_clusters(level=0)
        assert level0['level'] == 0
        assert len(level0['clusters']) == 10  # All original samples
        
        # Get a higher level
        level2 = cm.get_hierarchy_level_clusters(level=2)
        assert level2['level'] == 2
        assert len(level2['clusters']) < 10  # Should have fewer clusters
        
    def test_get_hierarchy_level_clusters_no_hierarchy(self, mock_embeddings_manager, mock_collection_with_data):
        """Test that getting hierarchy levels fails without hierarchy."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        # Use non-hierarchical clustering
        cm.cluster(method='kmeans', n_clusters=3)
        
        with pytest.raises(ClusteringError, match="Cluster hierarchy not available"):
            cm.get_hierarchy_level_clusters(level=0)
            
    def test_generate_hierarchical_labels_fallback(self, mock_embeddings_manager, mock_collection_with_data):
        """Test hierarchical label generation with fallback (no LLM)."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        cm.cluster(method='agglomerative', n_clusters=3)
        
        # Generate labels without LLM
        labels = cm.generate_hierarchical_labels(use_llm=False)
        
        assert labels is not None
        assert len(labels) > 0
        
        # Check that we have labels for both leaf and internal nodes
        tree = cm.cluster_hierarchy['tree']
        for node_id in tree['nodes'].keys():
            assert node_id in labels
            
    def test_parent_label_fallback(self, mock_embeddings_manager, mock_collection_with_data):
        """Test fallback parent label generation."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        cm.cluster(method='agglomerative', n_clusters=3)
        
        # Test the fallback method directly
        child_labels = ["Machine Learning", "Deep Learning", "Neural Networks"]
        parent_label = cm._generate_parent_label_fallback(child_labels)
        
        assert parent_label is not None
        assert isinstance(parent_label, str)
        assert len(parent_label) > 0


class TestClusteringCache:
    """Test suite for clustering cache with different parameters."""
    
    def test_cache_with_distance_threshold(self, tmp_path):
        """Test that cache considers distance_threshold parameter."""
        from abstracts_explorer.database import DatabaseManager
        from tests.conftest import set_test_db
        
        # Create temporary database
        db_path = tmp_path / "test_cache.db"
        set_test_db(db_path)
        
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        
        # Create mock results
        results1 = {
            'points': [{'x': 0, 'y': 0, 'cluster': 0}],
            'statistics': {'n_clusters': 2}
        }
        results2 = {
            'points': [{'x': 1, 'y': 1, 'cluster': 1}],
            'statistics': {'n_clusters': 3}
        }
        
        # Save first result with distance_threshold=1.0
        db.save_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='agglomerative',
            results=results1,
            n_clusters=None,
            clustering_params={'distance_threshold': 1.0, 'linkage': 'ward'}
        )
        
        # Save second result with distance_threshold=2.0
        db.save_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='agglomerative',
            results=results2,
            n_clusters=None,
            clustering_params={'distance_threshold': 2.0, 'linkage': 'ward'}
        )
        
        # Retrieve with distance_threshold=1.0 should get results1
        cached = db.get_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='agglomerative',
            n_clusters=None,
            clustering_params={'distance_threshold': 1.0, 'linkage': 'ward'}
        )
        assert cached is not None
        assert cached['statistics']['n_clusters'] == 2
        
        # Retrieve with distance_threshold=2.0 should get results2
        cached = db.get_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='agglomerative',
            n_clusters=None,
            clustering_params={'distance_threshold': 2.0, 'linkage': 'ward'}
        )
        assert cached is not None
        assert cached['statistics']['n_clusters'] == 3
        
        # Retrieve with distance_threshold=3.0 should return None (cache miss)
        cached = db.get_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='agglomerative',
            n_clusters=None,
            clustering_params={'distance_threshold': 3.0, 'linkage': 'ward'}
        )
        assert cached is None
        
        db.close()
    
    def test_cache_without_params(self, tmp_path):
        """Test that cache works when no params are provided."""
        from abstracts_explorer.database import DatabaseManager
        from tests.conftest import set_test_db
        
        # Create temporary database
        db_path = tmp_path / "test_cache2.db"
        set_test_db(db_path)
        
        db = DatabaseManager()
        db.connect()
        db.create_tables()
        
        # Save result without params
        results = {
            'points': [{'x': 0, 'y': 0, 'cluster': 0}],
            'statistics': {'n_clusters': 5}
        }
        
        db.save_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='kmeans',
            results=results,
            n_clusters=5,
            clustering_params=None
        )
        
        # Retrieve without params should work
        cached = db.get_clustering_cache(
            embedding_model='model1',
            reduction_method='pca',
            n_components=2,
            clustering_method='kmeans',
            n_clusters=5,
            clustering_params=None
        )
        assert cached is not None
        assert cached['statistics']['n_clusters'] == 5
        
        db.close()
