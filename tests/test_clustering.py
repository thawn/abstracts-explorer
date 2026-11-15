"""
Tests for clustering module.

This module tests the clustering and dimensionality reduction functionality.
"""

import pytest
import numpy as np
from abstracts_explorer.clustering import (
    ClusteringManager,
    ClusteringError,
    calculate_default_clusters,
    _compute_tfidf_keywords,
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

    def test_init(self, mock_embeddings_manager, mock_database):
        """Test ClusteringManager initialization."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        assert cm.embeddings_manager is mock_embeddings_manager
        assert cm.database is mock_database
        assert cm.embeddings is None
        assert cm.paper_ids is None
        assert cm.metadatas is None

    def test_load_embeddings_no_collection(self, mock_embeddings_manager, mock_database):
        """Test loading embeddings fails without collection."""
        mock_embeddings_manager.collection = None
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        with pytest.raises(ClusteringError, match="Collection not initialized"):
            cm.load_embeddings()

    def test_load_embeddings_empty_collection(self, mock_embeddings_manager, mock_collection_empty, mock_database):
        """Test loading embeddings from empty collection."""
        mock_embeddings_manager.collection = mock_collection_empty
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        with pytest.raises(ClusteringError, match="No embeddings found"):
            cm.load_embeddings()

    def test_load_embeddings_success(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test successfully loading embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        count = cm.load_embeddings()

        assert count == 10
        assert cm.embeddings.shape == (10, 128)
        assert len(cm.paper_ids) == 10
        assert len(cm.metadatas) == 10

    def test_reduce_dimensions_no_embeddings(self, mock_embeddings_manager, mock_database):
        """Test dimensionality reduction fails without embeddings."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        with pytest.raises(ClusteringError, match="No embeddings loaded"):
            cm.reduce_dimensions()

    def test_reduce_dimensions_tsne(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test t-SNE dimensionality reduction."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        reduced = cm.reduce_dimensions(n_components=2)

        assert reduced.shape == (10, 2)
        assert cm.reduced_embeddings is not None

    def test_cluster_no_embeddings(self, mock_embeddings_manager, mock_database):
        """Test clustering fails without embeddings."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        with pytest.raises(ClusteringError, match="No embeddings loaded"):
            cm.cluster()

    def test_cluster_auto_calculate_n_clusters(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test automatic calculation of n_clusters when None is provided."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        # Cluster with n_clusters=None should auto-calculate
        labels = cm.cluster(n_clusters=None, use_reduced=False)

        assert labels.shape == (10,)
        assert cm.cluster_labels is not None
        # With 10 papers, should calculate 2 clusters (10 / 100 = 0, min is 2)
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 2

    def test_cluster_agglomerative(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test Agglomerative clustering on full embeddings."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        # Cluster on full embeddings (not reduced)
        labels = cm.cluster(n_clusters=3, use_reduced=False)

        assert labels.shape == (10,)
        assert cm.cluster_labels is not None

    def test_get_cluster_statistics(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test getting cluster statistics."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        # Cluster first on full embeddings
        cm.cluster(n_clusters=3, use_reduced=False)

        stats = cm.get_cluster_statistics()

        assert "n_clusters" in stats
        assert "n_noise" in stats
        assert "cluster_sizes" in stats
        assert "total_papers" in stats
        assert stats["total_papers"] == 10

    def test_get_clustering_results(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test getting complete clustering results."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        # Cluster first, then reduce for visualization
        cm.cluster(n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(n_components=2)

        results = cm.get_clustering_results()

        assert "points" in results
        assert "statistics" in results
        assert "n_dimensions" in results
        assert "cluster_centers" in results
        assert len(results["points"]) == 10
        assert results["n_dimensions"] == 2

        # Verify cluster centers are calculated correctly
        assert isinstance(results["cluster_centers"], dict)
        # Should have centers for each non-noise cluster
        n_clusters = results["statistics"]["n_clusters"]
        assert len(results["cluster_centers"]) <= n_clusters

        # Each center should have x and y coordinates
        for cluster_id, center in results["cluster_centers"].items():
            assert "x" in center
            assert "y" in center
            assert isinstance(center["x"], (int, float))
            assert isinstance(center["y"], (int, float))

    def test_export_to_json(self, mock_embeddings_manager, mock_collection_with_data, tmp_path, mock_database):
        """Test exporting clustering results to JSON."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        # Cluster first, then reduce for visualization
        cm.cluster(n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(n_components=2)

        output_path = tmp_path / "clusters.json"
        cm.export_to_json(output_path)

        assert output_path.exists()

        # Verify JSON can be loaded
        import json

        with open(output_path) as f:
            data = json.load(f)

        assert "points" in data
        assert "statistics" in data


# Fixtures


@pytest.fixture
def mock_embeddings_manager(mocker):
    """Create a mock EmbeddingsManager."""
    em = mocker.MagicMock()
    em.collection = None
    return em


@pytest.fixture
def mock_database(mocker, mock_collection_with_data):
    """Create a mock DatabaseManager that hydrates the clustered papers."""
    db = mocker.MagicMock()
    data = mock_collection_with_data.get.return_value
    summaries = []
    for uid, metadata in zip(data["ids"], data["metadatas"]):
        summaries.append(
            {
                "uid": uid,
                "title": metadata["title"],
                "abstract": metadata["abstract"],
                "year": int(metadata["year"]),
                "conference": metadata["conference"],
                "session": metadata["session"],
            }
        )
    by_uid = {summary["uid"]: summary for summary in summaries}
    db.resolve_paper_uids.return_value = [summary["uid"] for summary in summaries]
    db.get_paper_summaries_by_uids.side_effect = lambda uids: [by_uid[uid] for uid in uids if uid in by_uid]
    return db


@pytest.fixture
def mock_collection_empty(mocker):
    """Create a mock empty ChromaDB collection."""
    collection = mocker.MagicMock()
    collection.get.return_value = {"ids": [], "embeddings": [], "metadatas": []}
    return collection


@pytest.fixture
def mock_collection_with_data(mocker):
    """Create a mock ChromaDB collection with test data."""
    collection = mocker.MagicMock()

    # Create 10 sample embeddings (128-dimensional)
    embeddings = np.random.randn(10, 128).tolist()
    ids = [f"paper_{i}" for i in range(10)]

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
        "This work proposes improvements to generative models for realistic image generation and manipulation.",
    ]

    metadatas = [
        {
            "title": f"Paper {i}",
            "abstract": abstracts[i],
            "authors": f"Author A{i};Author B{i}",
            "year": "2025",
            "conference": "TestConf",
            "session": "Session A",
            "poster_position": str(i),
        }
        for i in range(10)
    ]

    collection.get.return_value = {"ids": ids, "embeddings": embeddings, "metadatas": metadatas}

    return collection


class TestClusterLabeling:
    """Test suite for cluster labeling functionality."""

    def test_extract_cluster_keywords(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test extracting keywords for each cluster."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)

        keywords = cm.extract_cluster_keywords(n_keywords=5)

        assert isinstance(keywords, dict)
        assert len(keywords) > 0
        # Check that each cluster has keywords
        for cluster_id, kw_list in keywords.items():
            assert isinstance(kw_list, list)
            assert len(kw_list) <= 5

    def test_extract_cluster_keywords_no_clustering(self, mock_embeddings_manager, mock_database):
        """Test that keyword extraction fails without clustering."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)

        with pytest.raises(ClusteringError, match="No clustering performed"):
            cm.extract_cluster_keywords()

    def test_generate_cluster_labels_keyword_based(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test generating cluster labels without LLM."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)

        labels = cm.generate_cluster_labels(use_llm=False)

        assert isinstance(labels, dict)
        assert len(labels) > 0
        # Check that each cluster has a label
        for cluster_id, label in labels.items():
            assert isinstance(label, str)
            assert len(label) > 0

    def test_generate_cluster_labels_with_llm(
        self, mock_embeddings_manager, mock_collection_with_data, mocker, mock_database
    ):
        """Test generating cluster labels with LLM."""
        mock_embeddings_manager.collection = mock_collection_with_data

        # Mock the OpenAI client response
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Machine Learning Models"

        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_embeddings_manager.openai_client = mock_openai_client

        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)

        labels = cm.generate_cluster_labels(use_llm=True)

        assert isinstance(labels, dict)
        assert len(labels) > 0
        # At least one cluster should have the LLM-generated label
        assert any("Machine Learning" in label for label in labels.values())

    def test_llm_chat_label_returns_stripped_label(self, mock_embeddings_manager, mocker, mock_database):
        """_llm_chat_label strips whitespace and surrounding quotes."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "  'Deep Learning'  "

        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_embeddings_manager.openai_client = mock_openai_client

        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        result = cm._llm_chat_label(
            "system prompt",
            "user prompt",
        )

        assert result == "Deep Learning"
        # The call must use the dedicated label model, not the chat model
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        from abstracts_explorer.config import get_config

        assert call_kwargs["model"] == get_config().cluster_label_model
        # Budget must be large enough to absorb reasoning-model overhead
        assert call_kwargs["max_tokens"] >= 200
        # Reasoning must be suppressed where the back-end supports it
        assert call_kwargs["reasoning_effort"] == "low"

    def test_compute_tfidf_keywords_extracts_distinctive_terms(self):
        """Shared TF-IDF helper ranks terms for the selected documents."""
        documents = [
            "machine learning neural networks",
            "machine learning deep neural networks",
            "quantum physics particle collisions",
            "quantum physics particle accelerator",
        ]

        keywords = _compute_tfidf_keywords(documents, [0, 1], n_keywords=3, min_df=1)

        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        assert any("learning" in keyword for keyword in keywords)
        assert not any("quantum" in keyword for keyword in keywords)

    def test_compute_tfidf_keywords_empty_inputs(self):
        """Shared TF-IDF helper returns an empty list for unusable inputs."""
        assert _compute_tfidf_keywords([], [0], n_keywords=5) == []
        assert _compute_tfidf_keywords(["some text"], [], n_keywords=5) == []
        assert _compute_tfidf_keywords(["some text"], [0], n_keywords=0) == []

    def test_compute_tfidf_keywords_respects_min_df(self):
        """Shared TF-IDF helper filters terms below the minimum document frequency."""
        documents = ["alpha beta", "alpha gamma", "delta epsilon"]

        keywords = _compute_tfidf_keywords(documents, [0], n_keywords=5, min_df=2)

        assert keywords == ["alpha"]

    def test_extract_keywords_for_samples(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """_extract_keywords_for_samples delegates to the shared TF-IDF helper."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        keywords = cm._extract_keywords_for_samples(["paper_0", "paper_1"], max_keywords=3)

        assert isinstance(keywords, list)
        assert len(keywords) <= 3

    def test_extract_keywords_for_samples_unknown_ids(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Unknown paper IDs produce no keywords."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        assert cm._extract_keywords_for_samples(["missing-paper"], max_keywords=3) == []

    def test_call_llm_for_label_returns_llm_label(self, mock_embeddings_manager, mocker, mock_database):
        """_call_llm_for_label returns the stripped LLM response."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "  'Generated Label'  "

        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_embeddings_manager.openai_client = mock_openai_client

        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        result = cm._call_llm_for_label("user prompt", "fallback label")

        assert result == "Generated Label"

    def test_call_llm_for_label_returns_fallback_on_error(self, mock_embeddings_manager, mocker, mock_database):
        """_call_llm_for_label returns the fallback when the LLM call fails."""
        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_embeddings_manager.openai_client = mock_openai_client

        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        result = cm._call_llm_for_label("user prompt", "fallback label")

        assert result == "fallback label"

    def test_generate_llm_label_from_keywords_uses_shared_helper(
        self, mock_embeddings_manager, mocker, mock_database
    ):
        """_generate_llm_label_from_keywords delegates to _call_llm_for_label."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        helper = mocker.patch.object(cm, "_call_llm_for_label", return_value="Keyword Label")

        result = cm._generate_llm_label_from_keywords(["machine", "learning", "neural"])

        assert result == "Keyword Label"
        call_args = helper.call_args.args
        assert "machine, learning, neural" in call_args[0]
        assert call_args[1] == "machine, learning, neural"

    def test_generate_parent_label_llm_uses_shared_helper(self, mock_embeddings_manager, mocker, mock_database):
        """_generate_parent_label_llm delegates to _call_llm_for_label."""
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        helper = mocker.patch.object(cm, "_call_llm_for_label", return_value="Parent Label")
        child_labels = ["Machine Learning", "Deep Learning"]

        result = cm._generate_parent_label_llm(child_labels, [])

        assert result == "Parent Label"
        call_args = helper.call_args.args
        assert call_args[0]
        assert call_args[1] == cm._generate_parent_label_fallback(child_labels)

    def test_generate_llm_label_uses_shared_helper(
        self, mock_embeddings_manager, mock_collection_with_data, mocker, mock_database
    ):
        """_generate_llm_label delegates to _call_llm_for_label."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)
        helper = mocker.patch.object(cm, "_call_llm_for_label", return_value="Cluster Label")

        result = cm._generate_llm_label(0, ["machine", "learning"])

        assert result == "Cluster Label"
        call_args = helper.call_args.args
        assert "machine, learning" in call_args[0]
        assert call_args[1] == "machine, learning"

    def test_generate_cluster_labels_llm_truncated_content_falls_back(
        self, mock_embeddings_manager, mock_collection_with_data, mocker, mock_database
    ):
        """A reasoning model truncated at the token budget returns
        ``content=None`` (finish_reason='length'); each label must then fall
        back to keyword concatenation instead of crashing the whole run."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = None  # reasoning ate the budget

        mock_openai_client = mocker.MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_embeddings_manager.collection = mock_collection_with_data
        mock_embeddings_manager.openai_client = mock_openai_client

        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)

        labels = cm.generate_cluster_labels(use_llm=True)

        assert isinstance(labels, dict)
        assert len(labels) > 0
        # No label should be empty or None: the fallback kicked in everywhere
        for cluster_id, label in labels.items():
            assert isinstance(label, str)
            assert len(label) > 0

    def test_clustering_results_with_labels(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test that clustering results include labels when available."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3, use_reduced=False)
        cm.reduce_dimensions(n_components=2)
        cm.extract_cluster_keywords(n_keywords=5)
        cm.generate_cluster_labels(use_llm=False)

        results = cm.get_clustering_results()

        assert "cluster_labels" in results
        assert "cluster_keywords" in results
        assert isinstance(results["cluster_labels"], dict)
        assert isinstance(results["cluster_keywords"], dict)


class TestAgglomerativeVariants:
    """Test suite for agglomerative clustering variants."""

    def test_agglomerative_with_distance_threshold(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test agglomerative clustering with distance threshold instead of n_clusters."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        labels = cm.cluster(distance_threshold=5.0)

        assert labels is not None
        assert len(labels) == 10
        # Number of clusters should be determined by distance threshold
        n_clusters_found = len(np.unique(labels))
        assert n_clusters_found >= 1

    def test_agglomerative_hierarchy_extraction(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test that hierarchical structure is extracted for agglomerative clustering."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        cm.cluster(n_clusters=3)

        # Check that hierarchy was extracted
        assert cm.cluster_hierarchy is not None
        assert "n_samples" in cm.cluster_hierarchy
        assert "n_clusters" in cm.cluster_hierarchy
        assert "merges" in cm.cluster_hierarchy
        assert isinstance(cm.cluster_hierarchy["merges"], list)
        # Should have n_samples - 1 merges to form the tree
        assert len(cm.cluster_hierarchy["merges"]) == 9  # 10 samples - 1

    def test_agglomerative_hierarchy_in_results(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test that hierarchy is included in clustering results."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()
        cm.cluster(n_clusters=3)
        cm.reduce_dimensions(n_components=2)

        results = cm.get_clustering_results()

        assert "cluster_hierarchy" in results
        assert isinstance(results["cluster_hierarchy"], dict)
        assert "merges" in results["cluster_hierarchy"]

    def test_agglomerative_with_linkage(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test agglomerative clustering with different linkage methods."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        # Test with different linkage methods
        for linkage in ["ward", "complete", "average"]:
            cm_test = ClusteringManager(mock_embeddings_manager, mock_database)
            cm_test.load_embeddings()
            labels = cm_test.cluster(n_clusters=3, linkage=linkage)

            assert labels is not None
            assert len(labels) == 10


class TestHierarchicalClustering:
    """Test suite for hierarchical clustering visualization features."""

    def test_build_hierarchy_tree(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test that hierarchy tree is built correctly."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        cm.cluster(n_clusters=3)

        assert cm.cluster_hierarchy is not None
        assert "tree" in cm.cluster_hierarchy

        tree = cm.cluster_hierarchy["tree"]
        assert "nodes" in tree
        assert "root" in tree
        assert "max_level" in tree

        # Check that we have leaf and internal nodes
        assert len(tree["nodes"]) == 19  # 10 leaves + 9 internal nodes

    def test_generate_hierarchical_labels_fallback(
        self, mock_embeddings_manager, mock_collection_with_data, mock_database
    ):
        """Test hierarchical label generation with fallback (no LLM)."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        cm.cluster(n_clusters=3)

        # Generate labels without LLM
        labels = cm.generate_hierarchical_labels(use_llm=False)

        assert labels is not None
        assert len(labels) > 0

        # Check that we have labels for both leaf and internal nodes
        tree = cm.cluster_hierarchy["tree"]
        for node_id in tree["nodes"].keys():
            assert node_id in labels

    def test_parent_label_fallback(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test fallback parent label generation."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        cm.cluster(n_clusters=3)

        # Test the fallback method directly
        child_labels = ["Machine Learning", "Deep Learning", "Neural Networks"]
        parent_label = cm._generate_parent_label_fallback(child_labels)

        assert parent_label is not None
        assert isinstance(parent_label, str)
        assert len(parent_label) > 0

    def test_hierarchy_labels_in_tree_nodes(self, mock_embeddings_manager, mock_collection_with_data, mock_database):
        """Test that hierarchical labels are added to tree nodes."""
        mock_embeddings_manager.collection = mock_collection_with_data
        cm = ClusteringManager(mock_embeddings_manager, mock_database)
        cm.load_embeddings()

        # Use agglomerative clustering
        cm.cluster(n_clusters=3)

        # Generate hierarchical labels
        hierarchical_labels = cm.generate_hierarchical_labels(use_llm=False)

        # Verify labels are generated
        assert hierarchical_labels is not None
        assert len(hierarchical_labels) > 0

        # Manually update tree nodes with labels (simulating what compute_clusters_with_cache does)
        if "tree" in cm.cluster_hierarchy and "nodes" in cm.cluster_hierarchy["tree"]:
            for node_id, label in hierarchical_labels.items():
                node_id_int = int(node_id)
                if node_id_int in cm.cluster_hierarchy["tree"]["nodes"]:
                    cm.cluster_hierarchy["tree"]["nodes"][node_id_int]["label"] = label

        # Verify that tree nodes now have labels
        tree = cm.cluster_hierarchy["tree"]
        for node_id, node_info in tree["nodes"].items():
            if not node_info.get("is_leaf", False):  # Check internal nodes
                assert "label" in node_info, f"Node {node_id} missing label"
                assert node_info["label"] is not None
                assert len(node_info["label"]) > 0


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
        db.create_tables()

        # Create mock results
        results1 = {
            "points": [{"id": "p1", "x": 0, "y": 0, "cluster": 0}],
            "statistics": {"n_clusters": 2},
        }
        results2 = {
            "points": [{"id": "p1", "x": 1, "y": 1, "cluster": 1}],
            "statistics": {"n_clusters": 3},
        }

        # Save first result with distance_threshold=1.0
        db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            results=results1,
            n_clusters=None,
            clustering_params={"distance_threshold": 1.0, "linkage": "ward"},
        )

        # Save second result with distance_threshold=2.0
        db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            results=results2,
            n_clusters=None,
            clustering_params={"distance_threshold": 2.0, "linkage": "ward"},
        )

        # Retrieve with distance_threshold=1.0 should get results1
        cached = db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            clustering_params={"distance_threshold": 1.0, "linkage": "ward"},
        )
        assert cached is not None
        assert cached["statistics"]["n_clusters"] == 2

        # Retrieve with distance_threshold=2.0 should get results2
        cached = db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            clustering_params={"distance_threshold": 2.0, "linkage": "ward"},
        )
        assert cached is not None
        assert cached["statistics"]["n_clusters"] == 3

        # Retrieve with distance_threshold=3.0 should return None (cache miss)
        cached = db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="agglomerative",
            n_clusters=None,
            clustering_params={"distance_threshold": 3.0, "linkage": "ward"},
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
        db.create_tables()

        # Save result without params
        results = {
            "points": [{"id": "p1", "x": 0, "y": 0, "cluster": 0}],
            "statistics": {"n_clusters": 5},
        }

        db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=5,
            clustering_params=None,
        )

        # Retrieve without params should work
        cached = db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=5,
            clustering_params=None,
        )
        assert cached is not None
        assert cached["statistics"]["n_clusters"] == 5

        db.close()

    def test_different_reduction_methods_reuse_clustering(self, tmp_path):
        """Changing reduction method should find clustering results via clustering-only lookup."""
        from abstracts_explorer.database import DatabaseManager
        from tests.conftest import set_test_db

        db_path = tmp_path / "test_reduction.db"
        set_test_db(db_path)

        db = DatabaseManager()
        db.create_tables()

        results = {
            "points": [{"id": "p1", "x": 1.0, "y": 2.0, "cluster": 0}],
            "statistics": {"n_clusters": 2},
        }

        # Save with pca
        db.save_clustering_cache(
            embedding_model="model1",
            reduction_method="pca",
            n_components=2,
            clustering_method="kmeans",
            results=results,
            n_clusters=2,
        )

        # Exact match with tsne should miss
        cached_exact = db.get_clustering_cache(
            embedding_model="model1",
            reduction_method="tsne",
            n_components=2,
            clustering_method="kmeans",
            n_clusters=2,
        )
        assert cached_exact is None

        # Clustering-only lookup (no reduction_method) should find the pca entry
        cached_any = db.get_clustering_cache(
            embedding_model="model1",
            clustering_method="kmeans",
            n_clusters=2,
        )
        assert cached_any is not None
        assert cached_any["statistics"]["n_clusters"] == 2

        db.close()
