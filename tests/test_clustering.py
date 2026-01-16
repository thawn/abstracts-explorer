"""
Tests for clustering module.

This module tests the clustering and dimensionality reduction functionality.
"""

import pytest
import numpy as np
from abstracts_explorer.clustering import (
    ClusteringManager,
    ClusteringError,
)


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

    def test_reduce_dimensions_3d(self, mock_embeddings_manager, mock_collection_with_data):
        """Test 3D dimensionality reduction with PCA."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        
        reduced = cm.reduce_dimensions(method='pca', n_components=3)
        
        assert reduced.shape == (10, 3)
        assert cm.reduced_embeddings is not None
        assert cm.reduced_embeddings.shape == (10, 3)

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

    def test_get_clustering_results_3d(self, mock_embeddings_manager, mock_collection_with_data):
        """Test getting complete clustering results with 3D visualization."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        # Cluster first, then reduce to 3D for visualization
        cm.cluster(method='kmeans', n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(method='pca', n_components=3)
        
        results = cm.get_clustering_results()
        
        assert 'points' in results
        assert 'statistics' in results
        assert 'n_dimensions' in results
        assert 'cluster_centers' in results
        assert len(results['points']) == 10
        assert results['n_dimensions'] == 3
        
        # Verify points have z-coordinates
        for point in results['points']:
            assert 'x' in point
            assert 'y' in point
            assert 'z' in point
            assert isinstance(point['x'], (int, float))
            assert isinstance(point['y'], (int, float))
            assert isinstance(point['z'], (int, float))
        
        # Verify cluster centers have z-coordinates
        for cluster_id, center in results['cluster_centers'].items():
            assert 'x' in center
            assert 'y' in center
            assert 'z' in center
            assert isinstance(center['x'], (int, float))
            assert isinstance(center['y'], (int, float))
            assert isinstance(center['z'], (int, float))

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
