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
        """Test K-Means clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=2)
        
        labels = cm.cluster(method='kmeans', n_clusters=3)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None
        # Check that we have valid cluster labels
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 3
        assert np.all(labels >= 0)

    def test_cluster_dbscan(self, mock_embeddings_manager, mock_collection_with_data):
        """Test DBSCAN clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=2)
        
        labels = cm.cluster(method='dbscan', eps=0.5, min_samples=2)
        
        assert labels.shape == (10,)
        assert cm.cluster_labels is not None

    def test_cluster_agglomerative(self, mock_embeddings_manager, mock_collection_with_data):
        """Test Agglomerative clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=2)
        
        labels = cm.cluster(method='agglomerative', n_clusters=3)
        
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
        cm.reduce_dimensions(method='pca', n_components=2)
        cm.cluster(method='kmeans', n_clusters=3)
        
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
        cm.reduce_dimensions(method='pca', n_components=2)
        cm.cluster(method='kmeans', n_clusters=3)
        
        results = cm.get_clustering_results()
        
        assert 'points' in results
        assert 'statistics' in results
        assert 'n_dimensions' in results
        assert len(results['points']) == 10
        assert results['n_dimensions'] == 2

    def test_export_to_json(self, mock_embeddings_manager, mock_collection_with_data, tmp_path):
        """Test exporting clustering results to JSON."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager)
        cm.load_embeddings()
        cm.reduce_dimensions(method='pca', n_components=2)
        cm.cluster(method='kmeans', n_clusters=3)
        
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
    metadatas = [
        {
            'title': f'Paper {i}',
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
