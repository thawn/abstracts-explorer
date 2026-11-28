"""
Tests for MCP server functionality.
"""

import asyncio
import json
import threading
import pytest
from unittest.mock import Mock, patch
import numpy as np

from abstracts_explorer.mcp_server import (
    load_clustering_data,
    analyze_cluster_topics,
    ClusterAnalysisError,
)
from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.utils.conference_utils import parse_conference_year
from abstracts_explorer.clustering import ClusteringManager
from abstracts_explorer.database import DatabaseManager


class TestLoadClusteringData:
    """Tests for load_clustering_data function."""

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.ClusteringManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_load_with_defaults(self, mock_config, mock_cm_class, mock_db_class, mock_em_class):
        """Test loading clustering data with default config values."""
        # Setup mocks
        mock_config_obj = Mock()
        mock_config_obj.embedding_db = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.database_url = "sqlite:///abstracts.db"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_cm = Mock()
        mock_cm_class.return_value = mock_cm

        # Call function
        cm, db = load_clustering_data()

        # Verify calls - EmbeddingsManager now only takes collection_name
        mock_em_class.assert_called_once_with(
            collection_name="papers",
        )

        mock_db_class.assert_called_once_with()

        mock_cm_class.assert_called_once_with(mock_em, mock_db)

        assert cm == mock_cm
        assert db == mock_db

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_load_with_custom_collection(self, mock_config, mock_em_class):
        """Test loading with custom collection name."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em

        with (
            patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager") as mock_db_class,
            patch("abstracts_explorer.mcp.shared_dependencies.ClusteringManager") as mock_cm_class,
        ):
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_cm = Mock()
            mock_cm_class.return_value = mock_cm

            # Call with custom collection name
            cm, db = load_clustering_data(
                collection_name="custom_papers",
            )

            # Verify custom collection name was used
            mock_em_class.assert_called_once_with(
                collection_name="custom_papers",
            )
            mock_db_class.assert_called_once_with()

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_load_failure(self, mock_config, mock_em_class):
        """Test error handling when loading fails."""
        mock_config_obj = Mock()
        mock_config.return_value = mock_config_obj

        mock_em_class.side_effect = Exception("Connection failed")

        with pytest.raises(ClusterAnalysisError) as exc_info:
            load_clustering_data()

        assert "Failed to load clustering data" in str(exc_info.value)


class TestMCPResources:
    """Tests for shared MCP resource context managers."""

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_mcp_resources_reuses_thread_local_instances(self, mock_config, mock_db_class, mock_em_class):
        """The same thread reuses cached database and embeddings managers."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import _get_mcp_resources, close_mcp_resources

        with _get_mcp_resources() as (em, db):
            assert em is mock_em
            assert db is mock_db
        with _get_mcp_resources() as (em, db):
            assert em is mock_em
            assert db is mock_db

        mock_em_class.assert_called_once()
        mock_db_class.assert_called_once()
        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()

        close_mcp_resources()
        mock_em.close.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_mcp_resources_without_embeddings(self, mock_db_class, mock_em_class):
        """Database-only resources skip the embeddings manager."""
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import _get_mcp_resources, close_mcp_resources

        with _get_mcp_resources(with_embeddings=False) as (em, db):
            assert em is None
            assert db is mock_db

        mock_em_class.assert_not_called()
        mock_db.close.assert_not_called()
        close_mcp_resources()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_close_defers_resources_owned_by_live_thread(self, mock_db_class):
        """Default cleanup must not close resources used by another live thread."""
        from abstracts_explorer.mcp_server import _get_mcp_resources, close_mcp_resources

        first_db = Mock()
        second_db = Mock()
        mock_db_class.side_effect = [first_db, second_db]
        started = threading.Event()
        release = threading.Event()
        created = {}

        def worker() -> None:
            with _get_mcp_resources(with_embeddings=False) as (_, db):
                created["first"] = db
            started.set()
            release.wait(timeout=5)
            with _get_mcp_resources(with_embeddings=False) as (_, db):
                created["second"] = db

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            assert started.wait(timeout=5)
            close_mcp_resources()
            first_db.close.assert_not_called()
        finally:
            release.set()
            thread.join(timeout=5)

        assert not thread.is_alive()
        first_db.close.assert_called_once()
        assert created["second"] is second_db
        close_mcp_resources(force=True)
        second_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_force_close_closes_other_thread_resources(self, mock_db_class):
        """Forced cleanup closes resources even when the owner thread is alive."""
        from abstracts_explorer.mcp_server import _get_mcp_resources, close_mcp_resources

        db = Mock()
        mock_db_class.return_value = db
        started = threading.Event()
        release = threading.Event()

        def worker() -> None:
            with _get_mcp_resources(with_embeddings=False):
                started.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            assert started.wait(timeout=5)
            close_mcp_resources(force=True)
            db.close.assert_called_once()
        finally:
            release.set()
            thread.join(timeout=5)

        assert not thread.is_alive()

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_embeddings_are_cached_per_collection(self, mock_config, mock_db_class, mock_em_class):
        """Different collections get distinct embeddings managers."""
        mock_config.return_value = Mock(collection_name="papers")
        db = Mock()
        mock_db_class.return_value = db
        first_em = Mock()
        second_em = Mock()
        mock_em_class.side_effect = [first_em, second_em]

        from abstracts_explorer.mcp_server import _get_mcp_resources

        with _get_mcp_resources("first") as (em, _):
            assert em is first_em
        with _get_mcp_resources("second") as (em, _):
            assert em is second_em

        assert mock_em_class.call_count == 2
        mock_db_class.assert_called_once()

    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_database_is_cached_per_thread(self, mock_db_class):
        """Different threads receive distinct database managers."""
        databases = []
        mock_db_class.side_effect = lambda: databases.append(Mock()) or databases[-1]

        from abstracts_explorer.mcp_server import _get_mcp_resources

        results = []

        def worker():
            with _get_mcp_resources(with_embeddings=False) as (_, db):
                results.append(db)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 2
        assert results[0] is not results[1]
        assert mock_db_class.call_count == 2

    @patch("abstracts_explorer.mcp.resources.load_clustering_data")
    def test_clustering_resources_does_not_close_cached_resources(self, mock_load):
        """Clustering resources leave the cached managers open."""
        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)

        from abstracts_explorer.mcp_server import _get_clustering_resources

        with _get_clustering_resources() as (cm, db):
            assert cm is mock_cm
            assert db is mock_db

        mock_cm.embeddings_manager.close.assert_not_called()
        mock_db.close.assert_not_called()


class TestAnalyzeClusterTopics:
    """Tests for analyze_cluster_topics function."""

    def test_analyze_cluster(self):
        """Test analyzing topics in a cluster."""
        # Create mock clustering manager
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = np.array([0, 0, 1, 0, 1, 2])
        cm.paper_ids = ["p1", "p2", "p3", "p4", "p5", "p6"]
        cm.metadatas = [
            {"title": "Paper 1", "keywords": ["ml", "ai"], "session": "ML Track", "year": 2023},
            {"title": "Paper 2", "keywords": ["dl", "nn"], "session": "ML Track", "year": 2023},
            {"title": "Paper 3", "keywords": ["nlp", "transformers"], "session": "NLP Track", "year": 2024},
            {"title": "Paper 4", "keywords": ["ml", "dl"], "session": "ML Track", "year": 2024},
            {"title": "Paper 5", "keywords": ["nlp", "bert"], "session": "NLP Track", "year": 2024},
            {"title": "Paper 6", "keywords": ["cv", "vision"], "session": "CV Track", "year": 2025},
        ]
        cm.cluster_label_names = None
        cm.cluster_keywords = None

        db = Mock(spec=DatabaseManager)

        # Analyze cluster 0 (papers 0, 1, 3)
        result = analyze_cluster_topics(cm, db, cluster_id=0)

        assert result["paper_count"] == 3
        assert len(result["sample_titles"]) == 3
        assert result["sample_titles"][0] == "Paper 1"
        # topic and keywords are None/empty when not set
        assert result["topic"] is None
        assert result["keywords"] == []

    def test_analyze_cluster_with_label_names(self):
        """Test that topic and keywords are included when available."""
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = np.array([0, 0, 1, 1])
        cm.paper_ids = ["p1", "p2", "p3", "p4"]
        cm.metadatas = [
            {"title": "Paper 1", "keywords": ["ml", "neural"], "session": "ML", "year": 2023},
            {"title": "Paper 2", "keywords": ["deep", "learning"], "session": "ML", "year": 2023},
            {"title": "Paper 3", "keywords": ["nlp", "bert"], "session": "NLP", "year": 2024},
            {"title": "Paper 4", "keywords": ["nlp", "gpt"], "session": "NLP", "year": 2024},
        ]
        cm.cluster_label_names = {0: "Machine Learning", 1: "Natural Language Processing"}
        cm.cluster_descriptions = {
            0: "Papers apply neural networks to learning problems.",
            1: "Papers process language with transformer models.",
        }
        cm.cluster_keywords = {0: ["neural", "deep", "learning"], 1: ["nlp", "bert", "gpt"]}

        db = Mock(spec=DatabaseManager)

        result_0 = analyze_cluster_topics(cm, db, cluster_id=0)
        assert result_0["topic"] == "Papers apply neural networks to learning problems."
        assert result_0["label"] == "Machine Learning"
        assert result_0["keywords"] == ["neural", "deep", "learning"]

        result_1 = analyze_cluster_topics(cm, db, cluster_id=1)
        assert result_1["topic"] == "Papers process language with transformer models."
        assert result_1["label"] == "Natural Language Processing"
        assert result_1["keywords"] == ["nlp", "bert", "gpt"]

    def test_analyze_empty_cluster(self):
        """Test analyzing a cluster with no papers."""
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = np.array([0, 0, 1, 1])
        cm.paper_ids = ["p1", "p2", "p3", "p4"]
        cm.metadatas = [
            {"title": "Paper 1", "keywords": ["ml"], "session": "ML", "year": 2023},
            {"title": "Paper 2", "keywords": ["dl"], "session": "DL", "year": 2023},
            {"title": "Paper 3", "keywords": ["nlp"], "session": "NLP", "year": 2024},
            {"title": "Paper 4", "keywords": ["cv"], "session": "CV", "year": 2024},
        ]
        cm.cluster_label_names = None
        cm.cluster_keywords = None

        db = Mock(spec=DatabaseManager)

        # Analyze cluster 5 (doesn't exist)
        result = analyze_cluster_topics(cm, db, cluster_id=5)

        assert result["paper_count"] == 0
        assert result["topic"] is None
        assert result["keywords"] == []
        assert result["sample_titles"] == []

    def test_analyze_without_clustering(self):
        """Test error when clustering data not loaded."""
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = None
        cm.paper_ids = None
        cm.metadatas = None

        db = Mock(spec=DatabaseManager)

        with pytest.raises(ClusterAnalysisError) as exc_info:
            analyze_cluster_topics(cm, db, cluster_id=0)

        assert "Clustering data not loaded" in str(exc_info.value)

    def test_analyze_with_missing_metadata(self):
        """Test analyzing cluster with some missing metadata fields."""
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = np.array([0, 0])
        cm.paper_ids = ["p1", "p2"]
        cm.metadatas = [
            {"title": "Paper 1"},  # Missing keywords, session, year
            {"title": "Paper 2", "keywords": ["ml"], "year": 2023},  # Missing session
        ]
        cm.cluster_label_names = None
        cm.cluster_keywords = None

        db = Mock(spec=DatabaseManager)

        result = analyze_cluster_topics(cm, db, cluster_id=0)

        assert result["paper_count"] == 2
        assert len(result["sample_titles"]) == 2
        # Should handle missing fields gracefully
        assert result["topic"] is None
        assert isinstance(result["keywords"], list)


class TestMCPTools:
    """Tests for MCP tool functions."""

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_conference_topics(self, mock_config, mock_get_db):
        """Test get_conference_topics tool with cached hierarchy results."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0, "title": "Paper 1"},
                {"id": "p2", "cluster": 0, "x": 1.0, "y": 1.0, "title": "Paper 2"},
                {"id": "p3", "cluster": 1, "x": 2.0, "y": 2.0, "title": "Paper 3"},
                {"id": "p4", "cluster": 1, "x": 3.0, "y": 3.0, "title": "Paper 4"},
            ],
            "statistics": {
                "n_clusters": 2,
                "n_noise": 0,
                "cluster_sizes": {0: 2, 1: 2},
                "total_papers": 4,
            },
            "cluster_labels": {"0": "Machine Learning", "1": "Natural Language Processing"},
            "cluster_descriptions": {
                "0": "Papers apply neural networks to learning problems.",
                "1": "Papers process language with transformer models.",
            },
            "cluster_keywords": {"0": ["neural", "deep"], "1": ["nlp", "bert"]},
            "cluster_centers": {"0": {"x": 0.5, "y": 0.5}, "1": {"x": 2.5, "y": 2.5}},
            "hierarchy": {
                "method": "bertopic",
                "n_topics": 2,
                "tree": [
                    {
                        "id": 10,
                        "topics": [0, 1],
                        "label": "Machine Learning",
                        "description": "Papers apply neural networks to learning problems.",
                        "distance": 0.5,
                        "child_left_id": 0,
                        "child_right_id": 1,
                        "center": {"x": 1.5, "y": 1.5},
                    }
                ],
            },
        }

        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics(conferences=["NeurIPS"])
        result = json.loads(result_str)

        assert result["n_topics"] == 1
        assert result["total_papers"] == 4
        assert result["hierarchy_level"] == 0
        assert result["topic_sizes"] == {"Papers apply neural networks to learning problems.": 4}
        assert len(result["topics"]) == 1
        assert result["topics"][0]["topic"] == "Papers apply neural networks to learning problems."
        assert result["topics"][0]["label"] == "Machine Learning"
        assert result["topics"][0]["keywords"] == ["neural", "deep", "nlp", "bert"]
        assert result["topics"][0]["sample_titles"] == ["Paper 1", "Paper 2", "Paper 3", "Paper 4"]
        assert result["conference"] == "NeurIPS"
        mock_db.get_clustering_cache.assert_called_once()

    @patch("abstracts_explorer.mcp.resources.load_clustering_data")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_conference_topics_no_conference(self, mock_config, mock_load):
        """Test get_conference_topics raises when no conference is specified."""
        from abstracts_explorer.mcp_server import get_conference_topics

        with pytest.raises(ToolError, match="conference must be specified"):
            get_conference_topics()

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_conference_topics_no_cache(self, mock_config, mock_get_db):
        """Test get_conference_topics returns error when no cached results exist."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = None

        from abstracts_explorer.mcp_server import get_conference_topics

        with pytest.raises(ToolError, match="No pre-computed clustering data"):
            get_conference_topics(conferences=["NeurIPS"])

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_conference_topics_with_year_in_name(self, mock_config, mock_get_db):
        """Test get_conference_topics parses year from conference name like 'NeurIPS 2025'."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0, "title": "Paper 1"},
                {"id": "p2", "cluster": 0, "x": 1.0, "y": 1.0, "title": "Paper 2"},
            ],
            "statistics": {"n_clusters": 1, "n_noise": 0, "cluster_sizes": {0: 2}, "total_papers": 2},
            "cluster_labels": {"0": "Deep Learning"},
            "cluster_keywords": {"0": ["neural", "deep"]},
            "hierarchy": {"method": "bertopic", "n_topics": 1, "tree": []},
        }

        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics(conferences=["NeurIPS 2025"])
        result = json.loads(result_str)

        assert "error" not in result
        assert result["conference"] == "NeurIPS"
        assert "topics" in result
        call_args = mock_db.get_clustering_cache.call_args_list[0]
        assert call_args[1]["conference"] == "NeurIPS"
        assert call_args[1]["year"] == 2025


class TestToolResourceReuse:
    """Tests that tool failures raise ToolError without closing cached resources."""

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_raises_on_error(self, mock_config, mock_db_class, mock_em_class):
        """search_papers raises ToolError and keeps cached resources open."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.search.side_effect = RuntimeError("boom")

        from abstracts_explorer.mcp_server import search_papers

        with pytest.raises(ToolError, match="Paper search failed"):
            search_papers(topic_keywords="transformers", conference="NeurIPS")

        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution_raises_on_error(self, mock_config, mock_db_class, mock_em_class):
        """get_topic_evolution raises ToolError and keeps cached resources open."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.side_effect = RuntimeError("boom")

        from abstracts_explorer.mcp_server import get_topic_evolution

        with pytest.raises(ToolError, match="Topic evolution analysis failed"):
            get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS"])

        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_analyze_topic_relevance_raises_on_error(self, mock_config, mock_db_class, mock_em_class):
        """analyze_topic_relevance raises ToolError and keeps cached resources open."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_em.query_ids_within_distance.side_effect = RuntimeError("boom")

        from abstracts_explorer.mcp_server import analyze_topic_relevance

        with pytest.raises(ToolError, match="Topic relevance analysis failed"):
            analyze_topic_relevance(topic="transformers", conferences=["NeurIPS"])

        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()


class TestParseConferenceYear:
    """Tests for parse_conference_year helper."""

    def test_conference_with_year(self):
        assert parse_conference_year("NeurIPS 2025") == ("NeurIPS", 2025)

    def test_conference_without_year(self):
        assert parse_conference_year("ICLR") == ("ICLR", None)

    def test_conference_with_extra_spaces(self):
        assert parse_conference_year("  NeurIPS 2025  ") == ("NeurIPS", 2025)

    def test_conference_name_with_spaces_and_year(self):
        assert parse_conference_year("IEEE VIS 2024") == ("IEEE VIS", 2024)

    def test_conference_with_non_year_number(self):
        # 3-digit numbers shouldn't be parsed as years
        assert parse_conference_year("NeurIPS 123") == ("NeurIPS 123", None)

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_missing_year_cache_does_not_fallback(self, mock_config, mock_get_db):
        """A missing per-year cache must not be replaced by all-years data."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = None

        from abstracts_explorer.mcp_server import _get_conference_topics_for_single_conference

        with pytest.raises(ToolError, match="No pre-computed clustering data"):
            _get_conference_topics_for_single_conference("NeurIPS", years=[2025])

        assert mock_db.get_clustering_cache.call_count == 1


class TestTopicEvolutionAndRelevance:
    """Tests for get_topic_evolution, analyze_topic_relevance, and search_papers."""

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.return_value = [2023, 2024]
        mock_db.resolve_paper_uids.return_value = ["a1", "a2", "a3"]
        mock_db.get_paper_summaries_by_uids.return_value = [
            {"uid": "a1", "title": "Paper 1", "session": "ML", "year": 2023, "conference": "NeurIPS", "abstract": ""},
            {"uid": "a2", "title": "Paper 2", "session": "DL", "year": 2023, "conference": "NeurIPS", "abstract": ""},
            {"uid": "a3", "title": "Paper 3", "session": "ML", "year": 2024, "conference": "NeurIPS", "abstract": ""},
        ]
        mock_em.query_ids_within_distance.return_value = [("a1", 0.1), ("a2", 0.2), ("a3", 0.3)]
        mock_db.get_year_counts.return_value = {
            year: {"total_papers": 100, "presentations": 100} for year in (2023, 2024)
        }

        from abstracts_explorer.mcp_server import get_topic_evolution

        result = json.loads(get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS"]))

        assert result["total_papers"] == 3
        cdata = result["conference_data"]["NeurIPS"]
        assert cdata["year_counts"]["2023"] == 2
        assert cdata["year_counts"]["2024"] == 1
        query_arg = mock_em.query_ids_within_distance.call_args.kwargs["query"]
        assert query_arg.text == "transformers"
        assert mock_em.query_ids_within_distance.call_args.kwargs["distance_threshold"] == 1.1
        assert mock_em.query_ids_within_distance.call_args.kwargs["ids"] == ["a1", "a2", "a3"]

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution_with_year_range(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.return_value = [2022, 2023, 2024, 2025]
        mock_db.resolve_paper_uids.return_value = ["a1", "a2"]
        mock_db.get_paper_summaries_by_uids.return_value = [
            {
                "uid": "a1",
                "title": "Paper 2023",
                "session": "ML",
                "year": 2023,
                "conference": "NeurIPS",
                "abstract": "",
            },
            {
                "uid": "a2",
                "title": "Paper 2024",
                "session": "ML",
                "year": 2024,
                "conference": "NeurIPS",
                "abstract": "",
            },
        ]
        mock_em.query_ids_within_distance.return_value = [("a1", 0.1), ("a2", 0.2)]
        mock_db.get_year_counts.return_value = {
            year: {"total_papers": 50, "presentations": 50} for year in (2022, 2023, 2024, 2025)
        }

        from abstracts_explorer.mcp_server import get_topic_evolution

        result = json.loads(
            get_topic_evolution(
                topic_keywords="transformers",
                conferences=["NeurIPS"],
                start_year=2023,
                end_year=2024,
            )
        )

        cdata = result["conference_data"]["NeurIPS"]
        assert cdata["year_counts"]["2023"] == 1
        assert cdata["year_counts"]["2024"] == 1
        assert "2022" not in cdata["year_counts"]
        assert "2025" not in cdata["year_counts"]
        mock_db.resolve_paper_uids.assert_called_once_with(conferences=["NeurIPS"], years=[2023, 2024])

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution_with_distance_threshold(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.return_value = [2024]
        mock_db.resolve_paper_uids.return_value = ["a1"]
        mock_db.get_paper_summaries_by_uids.return_value = [
            {"uid": "a1", "title": "Paper 1", "session": "ML", "year": 2024, "conference": "NeurIPS", "abstract": ""}
        ]
        mock_em.query_ids_within_distance.return_value = [("a1", 0.3)]
        mock_db.get_year_counts.return_value = {2024: {"total_papers": 200, "presentations": 200}}

        from abstracts_explorer.mcp_server import get_topic_evolution

        result = json.loads(
            get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS"], distance_threshold=0.8)
        )

        assert result["distance_threshold"] == 0.8
        mock_em.query_ids_within_distance.assert_called_once()
        assert mock_em.query_ids_within_distance.call_args.kwargs["distance_threshold"] == 0.8

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution_multiple_conferences(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.return_value = [2023, 2024]
        mock_db.resolve_paper_uids.side_effect = lambda conferences=None, years=None: (
            ["n23", "n24"] if conferences == ["NeurIPS"] else ["i23", "i24"]
        )
        summaries = {
            "n23": {
                "uid": "n23",
                "title": "N23",
                "session": "ML",
                "year": 2023,
                "conference": "NeurIPS",
                "abstract": "",
            },
            "n24": {
                "uid": "n24",
                "title": "N24",
                "session": "ML",
                "year": 2024,
                "conference": "NeurIPS",
                "abstract": "",
            },
            "i23": {
                "uid": "i23",
                "title": "I23",
                "session": "ML",
                "year": 2023,
                "conference": "ICLR",
                "abstract": "",
            },
            "i24": {
                "uid": "i24",
                "title": "I24",
                "session": "ML",
                "year": 2024,
                "conference": "ICLR",
                "abstract": "",
            },
        }
        mock_db.get_paper_summaries_by_uids.side_effect = lambda uids: [summaries[uid] for uid in uids]
        mock_em.query_ids_within_distance.side_effect = lambda ids=None, **kwargs: [(uid, 0.1) for uid in (ids or [])]
        mock_db.get_year_counts.return_value = {
            year: {"total_papers": 100, "presentations": 100} for year in (2023, 2024)
        }

        from abstracts_explorer.mcp_server import get_topic_evolution

        result = json.loads(get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS", "ICLR"]))

        assert result["total_papers"] == 4
        assert result["conference_data"]["NeurIPS"]["year_counts"]["2023"] == 1
        assert result["conference_data"]["ICLR"]["year_counts"]["2024"] == 1

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_topic_evolution_uses_grouped_year_counts(self, mock_config, mock_db_class, mock_em_class):
        """Topic evolution fetches all year totals with one grouped query."""
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db
        mock_db.get_years.return_value = [2023, 2024]
        mock_db.resolve_paper_uids.return_value = []
        mock_db.get_year_counts.return_value = {
            2023: {"total_papers": 10, "presentations": 12},
            2024: {"total_papers": 20, "presentations": 25},
        }

        from abstracts_explorer.mcp_server import get_topic_evolution

        result = json.loads(get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS"]))

        mock_db.get_year_counts.assert_called_once_with(conference="NeurIPS", years=[2023, 2024])
        mock_db.get_stats.assert_not_called()
        cdata = result["conference_data"]["NeurIPS"]
        assert cdata["year_totals"] == {"2023": 10, "2024": 20}

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_analyze_topic_relevance_lightweight_hydration(self, mock_config, mock_db_class, mock_em_class):
        """Relevance counts all matches via light scopes and hydrates only the top five."""
        mock_config.return_value = Mock(collection_name="papers")
        mock_em = Mock()
        mock_db = Mock()
        mock_em_class.return_value = mock_em
        mock_db_class.return_value = mock_db

        candidate_ids = [f"p{i}" for i in range(6)]
        matches = [(f"p{i}", 0.1 * i) for i in range(6)]
        mock_db.resolve_paper_uids.return_value = candidate_ids
        mock_em.query_ids_within_distance.return_value = matches
        mock_db.get_paper_scopes_by_uids.return_value = {
            f"p{i}": {"conference": "NeurIPS" if i < 4 else "ICLR", "year": 2024 if i < 3 else 2025} for i in range(6)
        }
        mock_db.get_result_details.return_value = [
            {"uid": f"p{i}", "title": f"Paper {i}", "year": 2024, "conference": "NeurIPS", "distance": 0.1 * i}
            for i in range(5)
        ]

        from abstracts_explorer.mcp_server import analyze_topic_relevance

        result = json.loads(analyze_topic_relevance(topic="transformers", conferences=["NeurIPS"]))

        mock_db.get_result_details.assert_called_once_with(matches[:5])
        assert result["total_papers"] == 6
        assert result["total_considered"] == 6
        assert result["conferences"] == {"NeurIPS": 4, "ICLR": 2}
        assert result["years"] == {"2024": 3, "2025": 3}
        assert len(result["sample_papers"]) == 5
        assert result["closest_distance"] == 0.0

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_forwards_where_filters(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em_class.return_value = Mock()
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = ([{"uid": "p1", "title": "Paper 1"}], None)

        from abstracts_explorer.mcp_server import search_papers

        filters = {"session": "Oral"}
        result = json.loads(
            search_papers(
                topic_keywords="llm",
                conference="NeurIPS",
                where_filters=filters,
                years=[2024, 2025],
            )
        )

        assert result["papers_found"] == 1
        call_kwargs = mock_db.search.call_args.kwargs
        assert call_kwargs["where_filters"] == filters
        assert call_kwargs["conferences"] == ["NeurIPS"]
        assert call_kwargs["years"] == [2024, 2025]

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_forwards_complex_where_filters(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em_class.return_value = Mock()
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = ([{"uid": "p1", "title": "Paper 1"}], None)

        from abstracts_explorer.mcp_server import search_papers

        filters = {"$and": [{"year": {"$gte": 2024}}, {"session": {"$in": ["Oral Session 1", "Spotlight Session"]}}]}
        result = json.loads(
            search_papers(topic_keywords="deep learning", conference="NeurIPS", where_filters=filters)
        )

        assert result["topic"] == "deep learning"
        assert mock_db.search.call_args.kwargs["where_filters"] == filters

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_forwards_sessions(self, mock_config, mock_db_class, mock_em_class):
        mock_config.return_value = Mock(collection_name="papers")
        mock_em_class.return_value = Mock()
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = ([{"uid": "p1", "title": "Paper 1"}], None)

        from abstracts_explorer.mcp_server import search_papers

        sessions = ["Oral Session 1", "Spotlight Session"]
        result = json.loads(
            search_papers(
                topic_keywords="llm",
                conference="NeurIPS",
                sessions=sessions,
                years=[2024, 2025],
            )
        )

        assert result["papers_found"] == 1
        assert result["sessions_filter"] == sessions
        call_kwargs = mock_db.search.call_args.kwargs
        assert call_kwargs["sessions"] == sessions
        assert call_kwargs["conferences"] == ["NeurIPS"]
        assert call_kwargs["years"] == [2024, 2025]

    @patch("abstracts_explorer.mcp.tools.visualization.derive_hierarchy_levels")
    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_cluster_visualization(self, mock_config, mock_get_db, mock_derive):
        mock_config.return_value = Mock(collection_name="papers", embedding_model="test-model")
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0, "title": "Paper 1"},
                {"id": "p2", "x": 1.0, "y": 1.0, "cluster": 1, "title": "Paper 2"},
            ],
            "statistics": {"n_clusters": 2, "n_noise": 0, "cluster_sizes": {0: 1, 1: 1}, "total_papers": 2},
            "n_dimensions": 2,
            "cluster_centers": {"0": {"x": 0.0, "y": 0.0}, "1": {"x": 1.0, "y": 1.0}},
            "cluster_labels": {"0": "Cluster A", "1": "Cluster B"},
            "cluster_descriptions": {"0": "First cluster", "1": "Second cluster"},
        }

        levels = []
        for index in range(10):
            levels.append(
                {
                    "level": index,
                    "mapping": {"0": index * 10, "1": index * 10 + 1},
                    "clusters": [
                        {
                            "id": index * 10,
                            "label": f"Level {index} A",
                            "description": f"Level {index} first",
                            "size": 1,
                            "center": {"x": float(index), "y": float(index)},
                        },
                        {
                            "id": index * 10 + 1,
                            "label": f"Level {index} B",
                            "description": f"Level {index} second",
                            "size": 1,
                            "center": {"x": float(index + 1), "y": float(index + 1)},
                        },
                    ],
                }
            )
        mock_derive.return_value = {"levels": levels}

        from abstracts_explorer.mcp_server import get_cluster_visualization

        result = json.loads(get_cluster_visualization(conferences=["NeurIPS"]))

        assert result["n_dimensions"] == 2
        assert result["n_points"] == 2
        assert [point["cluster"] for point in result["points"]] == [90, 91]
        assert result["cluster_centers"]["90"] == {"x": 9.0, "y": 9.0}
        assert result["cluster_labels"] == {"90": "Level 9 A", "91": "Level 9 B"}
        assert result["cluster_descriptions"] == {"90": "Level 9 first", "91": "Level 9 second"}
        assert result["statistics"]["n_clusters"] == 2
        mock_db.get_clustering_cache.assert_called_once()

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_cluster_visualization_requires_hierarchy_tree(self, mock_config, mock_get_db):
        mock_config.return_value = Mock(collection_name="papers", embedding_model="test-model")
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = {
            "points": [{"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0}],
            "statistics": {"n_clusters": 1, "total_papers": 1},
            "hierarchy": {"method": "bertopic"},
        }

        from abstracts_explorer.mcp_server import get_cluster_visualization

        with pytest.raises(ToolError, match="missing their hierarchy tree"):
            get_cluster_visualization(conferences=["NeurIPS"])

    @patch("abstracts_explorer.mcp.tools.visualization._select_visualization_level")
    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_cluster_visualization_renumbers_multiple_conferences(self, mock_config, mock_get_db, mock_select):
        mock_config.return_value = Mock(collection_name="papers", embedding_model="test-model")
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.side_effect = [
            {
                "points": [{"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0, "title": "Paper 1"}],
                "statistics": {"n_clusters": 1, "total_papers": 1},
                "cluster_centers": {"0": {"x": 0.0, "y": 0.0}},
                "cluster_labels": {"0": "First"},
                "cluster_descriptions": {"0": "First cluster"},
            },
            {
                "points": [{"id": "p2", "x": 5.0, "y": 5.0, "cluster": 0, "title": "Paper 2"}],
                "statistics": {"n_clusters": 1, "total_papers": 1},
                "cluster_centers": {"0": {"x": 5.0, "y": 5.0}},
                "cluster_labels": {"0": "Second"},
                "cluster_descriptions": {"0": "Second cluster"},
            },
        ]
        mock_select.side_effect = [
            {
                "mapping": {"0": 10},
                "clusters": [
                    {
                        "id": 10,
                        "label": "First",
                        "description": "First cluster",
                        "size": 1,
                        "center": {"x": 0.0, "y": 0.0},
                    }
                ],
            },
            {
                "mapping": {"0": 20},
                "clusters": [
                    {
                        "id": 20,
                        "label": "Second",
                        "description": "Second cluster",
                        "size": 1,
                        "center": {"x": 5.0, "y": 5.0},
                    }
                ],
            },
        ]

        from abstracts_explorer.mcp_server import get_cluster_visualization

        result = json.loads(get_cluster_visualization(conferences=["NeurIPS", "ICLR"]))

        assert [point["cluster"] for point in result["points"]] == [0, 1]
        assert result["cluster_centers"]["0"] == {"x": 0.0, "y": 0.0}
        assert result["cluster_centers"]["1"] == {"x": 5.0, "y": 5.0}
        assert result["cluster_labels"] == {"0": "First", "1": "Second"}
        assert result["cluster_descriptions"] == {"0": "First cluster", "1": "Second cluster"}
        assert result["statistics"]["n_clusters"] == 2
        assert result["statistics"]["total_papers"] == 2

    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_cluster_visualization_no_conference(self, mock_config):
        from abstracts_explorer.mcp_server import get_cluster_visualization

        with pytest.raises(ToolError, match="conference must be specified"):
            get_cluster_visualization()

    @patch("abstracts_explorer.mcp.resources._get_shared_database")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_get_cluster_visualization_no_cache(self, mock_config, mock_get_db):
        mock_config.return_value = Mock(collection_name="papers", embedding_model="test-model")
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        mock_db.get_clustering_cache.return_value = None

        from abstracts_explorer.mcp_server import get_cluster_visualization

        with pytest.raises(ToolError, match="No pre-computed clustering data"):
            get_cluster_visualization(conferences=["NeurIPS"])


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    def test_mcp_server_initialization(self):
        """Test that MCP server initializes correctly."""
        from mcp.server.mcpserver import MCPServer

        from abstracts_explorer.mcp_server import mcp

        assert mcp is not None
        assert isinstance(mcp, MCPServer)

        tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        assert tool_names == {
            "analyze_topic_relevance",
            "get_cluster_visualization",
            "get_conference_topics",
            "get_paper_details",
            "get_topic_evolution",
            "search_papers",
            "summarize_interesting_papers",
            "recommend_related_papers",
            "suggest_authors_to_meet",
            "suggest_search_terms_from_orcid",
        }


class TestSearchPapersPaperCardFields:
    """
    Tests that verify search_papers returns the correct fields for paper card display.

    These integration tests ensure that the data returned by search_papers()
    contains all fields needed to correctly render paper cards in the web UI,
    including the fix for missing 'uid' and 'authors' fields.
    """

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_returns_uid_not_id(self, mock_config, mock_db_class, mock_em_class):
        """
        Test that search_papers returns 'uid' field instead of 'id'.

        Paper cards use paper.uid for star ratings and the detail modal.
        Returning 'id' instead would break these features.
        """
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["abc123"]],
            "metadatas": [
                [{"title": "Test Paper", "year": 2024, "conference": "NeurIPS", "authors": ["Alice", "Bob"]}]
            ],
            "documents": [["Test abstract"]],
            "distances": [[0.1]],
        }
        mock_db = Mock()
        mock_db.search.return_value = (
            [
                {
                    "uid": "abc123",
                    "title": "Test Paper",
                    "year": 2024,
                    "conference": "NeurIPS",
                    "authors": ["Alice", "Bob"],
                    "abstract": "Test abstract",
                }
            ],
            None,
        )
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="test", conference="NeurIPS")
        result = json.loads(result_str)

        assert "papers" in result
        assert len(result["papers"]) == 1

        paper = result["papers"][0]
        # Must have 'uid' field (not 'id') for paper card star ratings and detail modal
        assert "uid" in paper, "Paper card requires 'uid' field for star ratings and detail modal"
        assert paper["uid"] == "abc123"
        assert "id" not in paper, "Should use 'uid' not 'id' - 'id' would break paper cards"

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_returns_authors_as_list(self, mock_config, mock_db_class, mock_em_class):
        """
        Test that search_papers returns authors as a list, not a string.

        The paper card's formatPaperCard() expects paper.authors to be an array.
        Without this, authors would always show as 'Unknown' in chat sidebar.
        """
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["abc123"]],
            "metadatas": [
                [
                    {
                        "title": "Test Paper",
                        "year": 2024,
                        "conference": "NeurIPS",
                        "authors": ["Alice Smith", "Bob Jones", "Carol White"],
                    }
                ]
            ],
            "documents": [["Test abstract"]],
            "distances": [[0.1]],
        }
        mock_db = Mock()
        mock_db.search.return_value = (
            [
                {
                    "uid": "abc123",
                    "title": "Test Paper",
                    "year": 2024,
                    "conference": "NeurIPS",
                    "authors": ["Alice Smith", "Bob Jones", "Carol White"],
                    "abstract": "Test abstract",
                }
            ],
            None,
        )
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="test", conference="NeurIPS")
        result = json.loads(result_str)

        paper = result["papers"][0]
        # authors must be a list for formatPaperCard() to work
        assert "authors" in paper, "Paper card requires 'authors' field"
        assert isinstance(
            paper["authors"], list
        ), "authors must be a list for paper card display; string would show as 'Unknown'"
        assert paper["authors"] == ["Alice Smith", "Bob Jones", "Carol White"]

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_authors_empty_when_missing(self, mock_config, mock_db_class, mock_em_class):
        """Test that missing authors metadata results in an empty list, not 'Unknown'."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["xyz789"]],
            "metadatas": [[{"title": "No Authors Paper", "year": 2024, "conference": "ICLR"}]],
            "documents": [["Abstract text"]],
            "distances": [[0.2]],
        }
        mock_db = Mock()
        mock_db.search.return_value = (
            [
                {
                    "uid": "xyz789",
                    "title": "No Authors Paper",
                    "year": 2024,
                    "conference": "ICLR",
                    "authors": [],
                    "abstract": "Abstract text",
                }
            ],
            None,
        )
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="test", conference="ICLR")
        result = json.loads(result_str)

        paper = result["papers"][0]
        assert isinstance(paper["authors"], list)
        assert paper["authors"] == []

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_returns_conference_field(self, mock_config, mock_db_class, mock_em_class):
        """
        Test that search_papers returns the conference field.

        The paper card displays a conference badge if paper.conference is set.
        """
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["p1"]],
            "metadatas": [
                [
                    {
                        "title": "Conference Paper",
                        "year": 2025,
                        "conference": "NeurIPS",
                        "session": "Oral",
                        "authors": ["Author A"],
                    }
                ]
            ],
            "documents": [["Abstract"]],
            "distances": [[0.05]],
        }
        mock_db = Mock()
        mock_db.search.return_value = (
            [
                {
                    "uid": "p1",
                    "title": "Conference Paper",
                    "year": 2025,
                    "conference": "NeurIPS",
                    "session": "Oral",
                    "authors": ["Author A"],
                    "abstract": "Abstract",
                }
            ],
            None,
        )
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="machine learning", conference="NeurIPS")
        result = json.loads(result_str)

        paper = result["papers"][0]
        # conference field is needed for the conference badge in paper cards
        assert "conference" in paper, "Paper card requires 'conference' field for conference badge"
        assert paper["conference"] == "NeurIPS"

    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    def test_search_papers_all_card_fields_present(self, mock_config, mock_db_class, mock_em_class):
        """
        Test that search_papers returns all fields required for a complete paper card.

        Paper cards need: uid, title, authors (list), conference, session, abstract.
        """
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["p42"]],
            "metadatas": [
                [
                    {
                        "title": "Full Paper",
                        "year": 2024,
                        "conference": "ICML",
                        "session": "Best Paper Session",
                        "authors": ["First Author", "Second Author"],
                    }
                ]
            ],
            "documents": [["This is the abstract text."]],
            "distances": [[0.15]],
        }
        mock_db = Mock()
        mock_db.search.return_value = (
            [
                {
                    "uid": "p42",
                    "title": "Full Paper",
                    "year": 2024,
                    "conference": "ICML",
                    "session": "Best Paper Session",
                    "authors": ["First Author", "Second Author"],
                    "abstract": "This is the abstract text.",
                    "relevance_score": 0.85,
                }
            ],
            None,
        )
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="learning", conference="ICML")
        result = json.loads(result_str)

        paper = result["papers"][0]
        # All fields required by paper card
        assert paper["uid"] == "p42"
        assert paper["title"] == "Full Paper"
        assert paper["authors"] == ["First Author", "Second Author"]
        assert paper["conference"] == "ICML"
        assert paper["session"] == "Best Paper Session"
        assert paper["abstract"] == "This is the abstract text."
        assert "relevance_score" in paper


class TestGetPaperDetails:
    """Tests for the get_paper_details MCP tool."""

    def _make_paper_row(self, **overrides):
        """Return a minimal paper dict suitable for mock DB results.

        Returns authors as a list, mirroring what DatabaseManager._paper_to_dict()
        returns after deserializing the stored semicolon-separated string.
        """
        defaults = {
            "uid": "abc123",
            "original_id": "neurips2023/abc",
            "title": "A Test Paper",
            "authors": ["Smith, John", "Doe, Jane"],
            "abstract": "This paper describes a test.",
            "session": "Poster Session 1",
            "poster_position": "P01",
            "paper_pdf_url": "https://example.com/paper.pdf",
            "poster_image_url": None,
            "url": "https://example.com/paper",
            "room_name": "Hall A",
            "keywords": "deep learning, transformers",
            "starttime": "09:00",
            "endtime": "11:00",
            "award": None,
            "year": 2023,
            "conference": "NeurIPS",
            "created_at": "2024-01-01T00:00:00",
        }
        defaults.update(overrides)
        return defaults

    def test_no_arguments_returns_error(self):
        """get_paper_details raises when no title or paper_id is given."""
        from abstracts_explorer.mcp_server import get_paper_details

        with pytest.raises(ToolError, match="Provide at least one"):
            get_paper_details()

    def _mock_title_lookup(self, mock_db, row=None):
        """Configure a mock DB for the title lookup fallback chain."""
        mock_db.get_paper_by_title.return_value = None
        mock_db.get_paper.return_value = [row] if row is not None else []
        mock_db.keyword_search.return_value = []

    def test_lookup_by_exact_title(self):
        """An exact title match is used without a partial or keyword search."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.get_paper_by_title.return_value = self._make_paper_row()
        self._mock_title_lookup(mock_db)
        mock_db.get_paper_by_title.return_value = self._make_paper_row()

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="A Test Paper"))

        assert result["papers_found"] == 1
        mock_db.get_paper_by_title.assert_called_once_with("A Test Paper", conference=None, year=None)
        mock_db.get_paper.assert_not_called()
        mock_db.keyword_search.assert_not_called()

    def test_lookup_by_partial_title(self):
        """A partial title match uses a title-only contains query."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db, self._make_paper_row())

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Test Paper"))

        assert "error" not in result
        assert result["papers_found"] == 1
        paper = result["papers"][0]
        assert paper["title"] == "A Test Paper"
        assert paper["authors"] == ["Smith, John", "Doe, Jane"]
        assert paper["paper_pdf_url"] == "https://example.com/paper.pdf"
        assert paper["keywords"] == "deep learning, transformers"
        mock_db.get_paper.assert_called_once_with({"title": {"$contains": "Test Paper"}}, limit=5)
        mock_db.keyword_search.assert_not_called()

    def test_lookup_by_paper_id(self):
        """get_paper_details performs exact UID/original_id lookup when paper_id is provided."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.get_paper_by_original_id_or_uid.return_value = self._make_paper_row(award="Best Paper")

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(paper_id="abc123"))

        assert "error" not in result
        assert result["papers_found"] == 1
        paper = result["papers"][0]
        assert paper["uid"] == "abc123"
        assert paper["award"] == "Best Paper"
        assert paper["authors"] == ["Smith, John", "Doe, Jane"]
        mock_db.get_paper_by_original_id_or_uid.assert_called_once_with("abc123")

    def test_lookup_by_id_falls_back_to_title(self):
        """When paper_id lookup finds nothing, falls back to title search."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.get_paper_by_original_id_or_uid.return_value = None  # ID not found
        self._mock_title_lookup(mock_db, self._make_paper_row())

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(paper_id="notfound", title="Test Paper"))

        assert "error" not in result
        assert result["papers_found"] == 1
        mock_db.get_paper_by_original_id_or_uid.assert_called_once()
        mock_db.get_paper.assert_called_once()

    def test_title_only_no_match_falls_back_to_keyword(self):
        """A broad keyword search is only used when the title-only search is empty."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db)
        mock_db.keyword_search.return_value = [self._make_paper_row()]

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert result["papers_found"] == 1
        mock_db.get_paper.assert_called_once_with({"title": {"$contains": "Paper"}}, limit=5)
        mock_db.keyword_search.assert_called_once_with(keyword="Paper", conferences=None, years=None, limit=5)

    def test_authors_with_no_semicolon(self):
        """Single author is returned as one-element list (DatabaseManager already returns list)."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db, self._make_paper_row(authors=["Solo Author"]))

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert result["papers"][0]["authors"] == ["Solo Author"]

    def test_empty_authors_returns_empty_list(self):
        """None or empty authors field is returned as empty list."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db, self._make_paper_row(authors=[]))

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert result["papers"][0]["authors"] == []

    def test_conference_and_year_filters_passed_to_search(self):
        """conference and year parameters are forwarded to title and keyword lookups."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db)

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            get_paper_details(title="Paper", conference="ICLR", year=2024, limit=3)

        mock_db.get_paper_by_title.assert_called_once_with("Paper", conference="ICLR", year=2024)
        mock_db.get_paper.assert_called_once_with(
            {"$and": [{"title": {"$contains": "Paper"}}, {"conference": "ICLR"}, {"year": 2024}]},
            limit=3,
        )
        mock_db.keyword_search.assert_called_once_with(keyword="Paper", conferences=["ICLR"], years=[2024], limit=3)

    def test_limit_parameter(self):
        """limit parameter controls how many results are requested from DB."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db)

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            get_paper_details(title="Paper", limit=10)

        mock_db.get_paper.assert_called_once_with({"title": {"$contains": "Paper"}}, limit=10)

    def test_no_papers_found(self):
        """No matching papers returns papers_found=0 and empty list."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db)

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Nonexistent Paper"))

        assert "error" not in result
        assert result["papers_found"] == 0
        assert result["papers"] == []

    def test_exception_returns_error_json(self):
        """Database exception raises ToolError without leaking details."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        self._mock_title_lookup(mock_db)
        mock_db.keyword_search.side_effect = Exception("DB connection failed")

        with patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager", return_value=mock_db):
            with pytest.raises(ToolError, match="Paper lookup failed"):
                get_paper_details(title="Paper")

        mock_db.close.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestInterestingPaperMCPTools:
    """Tests for the interesting-paper MCP tools."""

    def test_summarize_without_ratings_returns_error(self):
        from abstracts_explorer.mcp_server import summarize_interesting_papers

        with pytest.raises(ToolError, match="No interesting papers"):
            summarize_interesting_papers()

    @patch("abstracts_explorer.mcp.shared_dependencies._summarize_interesting_papers")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_summarize_delegates_to_core(self, mock_db_class, mock_summarize):
        from abstracts_explorer.mcp_server import summarize_interesting_papers

        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_summarize.return_value = {"total_interesting": 1, "papers": []}

        result = json.loads(
            summarize_interesting_papers(
                priorities={"p1": {"priority": 5, "searchTerm": "llms"}},
                conference="ICLR",
                years=[2026],
            )
        )

        assert result["total_interesting"] == 1
        mock_summarize.assert_called_once_with(
            database=mock_db,
            priorities={"p1": {"priority": 5, "searchTerm": "llms"}},
            paper_ids=None,
            conference="ICLR",
            years=[2026],
        )
        mock_db.close.assert_not_called()

    def test_recommend_without_ratings_returns_error(self):
        from abstracts_explorer.mcp_server import recommend_related_papers

        with pytest.raises(ToolError, match="No interesting papers"):
            recommend_related_papers()

    @patch("abstracts_explorer.mcp.shared_dependencies._recommend_related_papers")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_recommend_delegates_to_core(
        self,
        mock_db_class,
        mock_em_class,
        mock_config,
        mock_recommend,
    ):
        from abstracts_explorer.mcp_server import recommend_related_papers

        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_recommend.return_value = {"papers": [], "recommendations_count": 0}

        result = json.loads(
            recommend_related_papers(
                priorities={"p1": {"priority": 5}},
                conferences=["ICLR"],
                years=[2026],
                limit=3,
            )
        )

        assert result["recommendations_count"] == 0
        mock_recommend.assert_called_once_with(
            database=mock_db,
            embeddings_manager=mock_em,
            priorities={"p1": {"priority": 5}},
            paper_ids=None,
            conferences=["ICLR"],
            years=[2026],
            limit=3,
            exclude_ids=None,
        )
        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()


class TestSuggestAuthorsMCPTool:
    """Tests for the suggest-authors-to-meet MCP tool."""

    def test_suggest_authors_without_ratings_returns_error(self):
        from abstracts_explorer.mcp_server import suggest_authors_to_meet

        with pytest.raises(ToolError, match="No interesting papers"):
            suggest_authors_to_meet()

    @patch("abstracts_explorer.mcp.shared_dependencies._suggest_authors_to_meet")
    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_suggest_authors_delegates_to_core(
        self,
        mock_db_class,
        mock_em_class,
        mock_config,
        mock_suggest,
    ):
        from abstracts_explorer.mcp_server import suggest_authors_to_meet

        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_suggest.return_value = {"authors": [], "authors_count": 0}

        result = json.loads(
            suggest_authors_to_meet(
                priorities={"p1": {"priority": 5}},
                conferences=["ICLR"],
                years=[2026],
                limit=3,
                paper_limit=20,
            )
        )

        assert result["authors_count"] == 0
        mock_suggest.assert_called_once_with(
            database=mock_db,
            embeddings_manager=mock_em,
            priorities={"p1": {"priority": 5}},
            paper_ids=None,
            conferences=["ICLR"],
            years=[2026],
            limit=3,
            paper_limit=20,
            exclude_ids=None,
        )
        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()


class TestInterestingPaperLegacyFormatRejection:
    """Legacy scalar ratings are rejected by the interesting-paper tools."""

    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_summarize_rejects_scalar_priorities(self, mock_db_class):
        from abstracts_explorer.mcp_server import summarize_interesting_papers

        mock_db = Mock()
        mock_db_class.return_value = mock_db

        with pytest.raises(ToolError, match="Invalid rating format"):
            summarize_interesting_papers(priorities={"p1": 5})
        mock_db.close.assert_not_called()

    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_recommend_rejects_scalar_priorities(self, mock_db_class, mock_em_class, mock_config):
        from abstracts_explorer.mcp_server import recommend_related_papers

        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        with pytest.raises(ToolError, match="Invalid rating format"):
            recommend_related_papers(priorities={"p1": 5})
        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()

    @patch("abstracts_explorer.mcp.shared_dependencies.get_config")
    @patch("abstracts_explorer.mcp.shared_dependencies.EmbeddingsManager")
    @patch("abstracts_explorer.mcp.shared_dependencies.DatabaseManager")
    def test_suggest_authors_rejects_scalar_priorities(self, mock_db_class, mock_em_class, mock_config):
        from abstracts_explorer.mcp_server import suggest_authors_to_meet

        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        with pytest.raises(ToolError, match="Invalid rating format"):
            suggest_authors_to_meet(priorities={"p1": 5})
        mock_em.close.assert_not_called()
        mock_db.close.assert_not_called()
