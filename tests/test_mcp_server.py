"""
Tests for MCP server functionality.
"""

import json
import pytest
from unittest.mock import Mock, patch
import numpy as np

from abstracts_explorer.mcp_server import (
    load_clustering_data,
    analyze_cluster_topics,
    merge_where_clause_with_conference,
    ClusterAnalysisError,
    _parse_conference_year,
)
from abstracts_explorer.clustering import ClusteringManager
from abstracts_explorer.database import DatabaseManager


class TestMergeWhereClauseWithConference:
    """Tests for merge_where_clause_with_conference helper function."""

    def test_both_none(self):
        """Test merging when both WHERE and conference are None."""
        result = merge_where_clause_with_conference(None, None)
        assert result is None

    def test_only_conference(self):
        """Test merging with only conference filter."""
        result = merge_where_clause_with_conference(None, "NeurIPS")
        assert result == {"conference": "NeurIPS"}

    def test_only_where(self):
        """Test merging with only WHERE clause."""
        where = {"year": 2024}
        result = merge_where_clause_with_conference(where, None)
        assert result == {"year": 2024}
        assert result is not where  # Should be a deep copy

    def test_deep_copy_prevents_mutation(self):
        """Test that deep copy prevents mutations to nested structures."""
        where = {"$and": [{"year": 2024}, {"session": "Oral"}]}
        result = merge_where_clause_with_conference(where, "NeurIPS")

        # Modify result
        result["$and"].append({"modified": True})

        # Original should be unchanged
        assert len(where["$and"]) == 2
        assert {"modified": True} not in where["$and"]

    def test_simple_merge(self):
        """Test merging simple WHERE clause with conference."""
        where = {"year": 2024}
        result = merge_where_clause_with_conference(where, "NeurIPS")
        assert result == {"$and": [{"year": 2024}, {"conference": "NeurIPS"}]}

    def test_merge_with_existing_and(self):
        """Test merging when WHERE already has $and."""
        where = {"$and": [{"year": 2024}, {"session": "Oral"}]}
        result = merge_where_clause_with_conference(where, "NeurIPS")
        assert "$and" in result
        assert len(result["$and"]) == 3
        # Verify all items are present
        assert {"year": 2024} in result["$and"]
        assert {"session": "Oral"} in result["$and"]
        assert {"conference": "NeurIPS"} in result["$and"]

    def test_conference_already_in_where_top_level(self):
        """Test when conference already exists at top level."""
        where = {"conference": "ICML"}
        result = merge_where_clause_with_conference(where, "NeurIPS")
        # Should not add duplicate, just return existing
        assert result == {"conference": "ICML"}

    def test_conference_in_nested_and(self):
        """Test when conference exists in nested $and."""
        where = {"$and": [{"conference": "ICML"}, {"year": 2024}]}
        result = merge_where_clause_with_conference(where, "NeurIPS")
        # Should detect existing conference and not add duplicate
        assert result == {"$and": [{"conference": "ICML"}, {"year": 2024}]}

    def test_conference_in_deeply_nested_structure(self):
        """Test conference detection in deeply nested structure."""
        where = {"$or": [{"$and": [{"conference": "ICML"}, {"year": 2024}]}, {"session": "Oral"}]}
        result = merge_where_clause_with_conference(where, "NeurIPS")
        # Should detect conference in nested structure
        assert result == where

    def test_invalid_where_type(self):
        """Test that invalid WHERE type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            merge_where_clause_with_conference("invalid", "NeurIPS")
        assert "must be a dict" in str(exc_info.value)

    def test_where_is_list(self):
        """Test that list WHERE raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            merge_where_clause_with_conference([{"year": 2024}], "NeurIPS")
        assert "must be a dict" in str(exc_info.value)


class TestLoadClusteringData:
    """Tests for load_clustering_data function."""

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.ClusteringManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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
        mock_em.connect.assert_called_once()
        mock_em.create_collection.assert_called_once()

        mock_db_class.assert_called_once_with()
        mock_db.connect.assert_called_once()

        mock_cm_class.assert_called_once_with(mock_em, mock_db)

        assert cm == mock_cm
        assert db == mock_db

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_load_with_custom_collection(self, mock_config, mock_em_class):
        """Test loading with custom collection name."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        mock_em = Mock()
        mock_em_class.return_value = mock_em

        with (
            patch("abstracts_explorer.mcp_server.DatabaseManager") as mock_db_class,
            patch("abstracts_explorer.mcp_server.ClusteringManager") as mock_cm_class,
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

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_load_failure(self, mock_config, mock_em_class):
        """Test error handling when loading fails."""
        mock_config_obj = Mock()
        mock_config.return_value = mock_config_obj

        mock_em_class.side_effect = Exception("Connection failed")

        with pytest.raises(ClusterAnalysisError) as exc_info:
            load_clustering_data()

        assert "Failed to load clustering data" in str(exc_info.value)


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
        cm.cluster_keywords = {0: ["neural", "deep", "learning"], 1: ["nlp", "bert", "gpt"]}

        db = Mock(spec=DatabaseManager)

        result_0 = analyze_cluster_topics(cm, db, cluster_id=0)
        assert result_0["topic"] == "Machine Learning"
        assert result_0["keywords"] == ["neural", "deep", "learning"]

        result_1 = analyze_cluster_topics(cm, db, cluster_id=1)
        assert result_1["topic"] == "Natural Language Processing"
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

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.analyze_cluster_topics")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_conference_topics(self, mock_config, mock_analyze, mock_load):
        """Test get_conference_topics tool with cached results."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        # Setup mocks
        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_cm.paper_ids = ["p1", "p2", "p3", "p4"]
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)

        mock_cm.load_embeddings.return_value = 100

        # db.get_clustering_cache returns cached results (includes cluster names and keywords)
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0},
                {"id": "p2", "cluster": 0, "x": 1.0, "y": 1.0},
                {"id": "p3", "cluster": 1, "x": 2.0, "y": 2.0},
                {"id": "p4", "cluster": 1, "x": 3.0, "y": 3.0},
            ],
            "statistics": {
                "n_clusters": 2,
                "n_noise": 0,
                "cluster_sizes": {0: 2, 1: 2},
                "total_papers": 4,
            },
            "cluster_labels": {"0": "Machine Learning", "1": "Natural Language Processing"},
            "cluster_keywords": {"0": ["neural", "deep"], "1": ["nlp", "bert"]},
        }

        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 2,
            "n_noise": 0,
            "cluster_sizes": {0: 2, 1: 2},
            "total_papers": 4,
        }

        mock_analyze.side_effect = [
            {
                "topic": "Machine Learning",
                "paper_count": 2,
                "keywords": ["neural", "deep"],
                "sample_titles": ["Paper 1", "Paper 2"],
            },
            {
                "topic": "Natural Language Processing",
                "paper_count": 2,
                "keywords": ["nlp", "bert"],
                "sample_titles": ["Paper 3", "Paper 4"],
            },
        ]

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics(conferences=["NeurIPS"])
        result = json.loads(result_str)

        # Verify result uses topic-centric structure
        assert result["n_topics"] == 2
        assert result["total_papers"] == 4
        # topic_sizes should use topic names (sorted by size desc)
        assert result["topic_sizes"] == {
            "Machine Learning": 2,
            "Natural Language Processing": 2,
        }
        assert "topics" in result
        assert len(result["topics"]) == 2
        assert result["topics"][0]["topic"] == "Machine Learning"
        assert result["topics"][0]["keywords"] == ["neural", "deep"]
        assert result["topics"][1]["topic"] == "Natural Language Processing"
        assert result["conference"] == "NeurIPS"

        # Verify _apply_cached_cluster_labels restored cluster_label_names and cluster_keywords
        assert mock_cm.cluster_label_names == {0: "Machine Learning", 1: "Natural Language Processing"}
        assert mock_cm.cluster_keywords == {0: ["neural", "deep"], 1: ["nlp", "bert"]}

        # Verify cache was queried with correct params
        mock_db.get_clustering_cache.assert_called_once()

        # Verify cleanup
        mock_cm.embeddings_manager.close.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_conference_topics_no_conference(self, mock_config, mock_load):
        """Test get_conference_topics returns error when no conference is specified."""
        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics()
        result = json.loads(result_str)

        assert "error" in result
        assert "conference must be specified" in result["error"]

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_conference_topics_no_cache(self, mock_config, mock_load):
        """Test get_conference_topics returns error when no cached results exist."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)
        mock_db.get_clustering_cache.return_value = None

        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics(conferences=["NeurIPS"])
        result = json.loads(result_str)

        assert "error" in result
        assert "No pre-computed clustering data" in result["error"]
        assert "NeurIPS" in result["error"]

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.analyze_cluster_topics")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_conference_topics_with_year_in_name(self, mock_config, mock_analyze, mock_load):
        """Test get_conference_topics parses year from conference name like 'NeurIPS 2025'."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_cm.paper_ids = ["p1", "p2"]
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)

        mock_cm.load_embeddings.return_value = 100
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0},
                {"id": "p2", "cluster": 0, "x": 1.0, "y": 1.0},
            ],
            "statistics": {"n_clusters": 1, "n_noise": 0, "cluster_sizes": {0: 2}, "total_papers": 2},
            "cluster_labels": {"0": "Deep Learning"},
            "cluster_keywords": {"0": ["neural", "deep"]},
        }

        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 1,
            "n_noise": 0,
            "cluster_sizes": {0: 2},
            "total_papers": 2,
        }
        mock_cm.cluster_label_names = {0: "Deep Learning"}

        mock_analyze.return_value = {
            "topic": "Deep Learning",
            "paper_count": 2,
            "keywords": ["neural", "deep"],
            "sample_titles": ["Paper 1"],
        }

        from abstracts_explorer.mcp_server import get_conference_topics

        result_str = get_conference_topics(conferences=["NeurIPS 2025"])
        result = json.loads(result_str)

        # Should succeed (no error) by parsing "NeurIPS 2025" → conference="NeurIPS", years=[2025]
        assert "error" not in result
        assert result["conference"] == "NeurIPS"
        assert "topics" in result

        # Verify the cache was looked up with the parsed conference name (not "NeurIPS 2025")
        call_args = mock_db.get_clustering_cache.call_args_list[0]
        assert call_args[1]["conference"] == "NeurIPS"
        assert call_args[1]["year"] == 2025


class TestParseConferenceYear:
    """Tests for _parse_conference_year helper."""

    def test_conference_with_year(self):
        assert _parse_conference_year("NeurIPS 2025") == ("NeurIPS", 2025)

    def test_conference_without_year(self):
        assert _parse_conference_year("ICLR") == ("ICLR", None)

    def test_conference_with_extra_spaces(self):
        assert _parse_conference_year("  NeurIPS 2025  ") == ("NeurIPS", 2025)

    def test_conference_name_with_spaces_and_year(self):
        assert _parse_conference_year("IEEE VIS 2024") == ("IEEE VIS", 2024)

    def test_conference_with_non_year_number(self):
        # 3-digit numbers shouldn't be parsed as years
        assert _parse_conference_year("NeurIPS 123") == ("NeurIPS 123", None)

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.analyze_cluster_topics")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_fallback_to_all_years_cache(self, mock_config, mock_analyze, mock_load):
        """When per-year cache is not found, fallback to all-years cache."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_cm.paper_ids = ["p1"]
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)
        mock_cm.load_embeddings.return_value = 10
        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 1,
            "n_noise": 0,
            "cluster_sizes": {0: 1},
            "total_papers": 1,
        }
        mock_cm.cluster_label_names = {0: "AI"}

        mock_analyze.return_value = {
            "topic": "AI",
            "paper_count": 1,
            "keywords": ["artificial"],
            "sample_titles": ["Paper 1"],
        }

        # First call (with years) returns None, second (without years) returns cache
        mock_db.get_clustering_cache.side_effect = [
            None,  # per-year lookup fails
            {  # all-years fallback succeeds
                "points": [{"id": "p1", "cluster": 0, "x": 0.0, "y": 0.0}],
                "statistics": {"n_clusters": 1, "n_noise": 0, "cluster_sizes": {0: 1}, "total_papers": 1},
                "cluster_labels": {"0": "AI"},
                "cluster_keywords": {"0": ["artificial"]},
            },
        ]

        from abstracts_explorer.mcp_server import _get_conference_topics_for_single_conference

        result = _get_conference_topics_for_single_conference("NeurIPS", years=[2025])

        assert "error" not in result
        assert mock_db.get_clustering_cache.call_count == 2
        # Second call should NOT have 'years' in params
        fallback_params = mock_db.get_clustering_cache.call_args_list[1][1]["clustering_params"]
        assert "years" not in fallback_params


class TestTopicEvolutionAndRelevance:
    """Tests for get_topic_evolution and analyze_topic_relevance tools."""

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution tool."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        def _find_papers(database, query, distance_threshold, conferences=None, years=None, query_embedding=None):
            year = years[0] if years else None
            if year == 2023:
                return {
                    "count": 2,
                    "papers": [
                        {"title": "Paper 1", "session": "ML", "distance": 0.1},
                        {"title": "Paper 2", "session": "DL", "distance": 0.2},
                    ],
                    "total_considered": 50,
                }
            elif year == 2024:
                return {
                    "count": 1,
                    "papers": [
                        {"title": "Paper 3", "session": "ML", "distance": 0.3},
                    ],
                    "total_considered": 40,
                }
            return {"count": 0, "papers": [], "total_considered": 0}

        mock_em.find_papers_within_distance.side_effect = _find_papers

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_years_for_conference.return_value = [2023, 2024]
        mock_db.get_stats.side_effect = lambda year, conference: {"total_papers": 100}

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(topic_keywords="transformers", conferences=["NeurIPS"])
        result = json.loads(result_str)

        # Verify result structure
        assert result["topic"] == "transformers"
        assert result["conferences"] == ["NeurIPS"]
        assert result["total_papers"] == 3

        # Per-conference data
        assert "NeurIPS" in result["conference_data"]
        cdata = result["conference_data"]["NeurIPS"]
        assert cdata["year_counts"]["2023"] == 2  # Keys are strings after JSON
        assert cdata["year_counts"]["2024"] == 1
        assert cdata["year_relative"]["2023"] == 2.0  # 2/100 * 100
        assert cdata["year_relative"]["2024"] == 1.0  # 1/100 * 100
        assert "2023" in cdata["papers_by_year"]
        assert "2024" in cdata["papers_by_year"]

        # Verify cleanup
        mock_em.close.assert_called_once()
        mock_db.close.assert_called_once()

        # Verify query was embedded only once, not once per year
        mock_em.generate_embedding.assert_called_once_with("transformers")

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_with_year_range(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution with start_year and end_year filters."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        def _find_papers(database, query, distance_threshold, conferences=None, years=None, query_embedding=None):
            year = years[0] if years else None
            return {
                "count": 1,
                "papers": [{"title": f"Paper {year}", "session": "ML", "distance": 0.1}],
                "total_considered": 30,
            }

        mock_em.find_papers_within_distance.side_effect = _find_papers

        # Setup database mock - DB has years 2022-2025
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_years_for_conference.return_value = [2022, 2023, 2024, 2025]
        mock_db.get_stats.side_effect = lambda year, conference: {"total_papers": 50}

        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(
            topic_keywords="transformers",
            conferences=["NeurIPS"],
            start_year=2023,
            end_year=2024,
        )
        result = json.loads(result_str)

        # Only papers from 2023 and 2024 should be included
        cdata = result["conference_data"]["NeurIPS"]
        assert cdata["year_counts"]["2023"] == 1
        assert cdata["year_counts"]["2024"] == 1
        assert "2022" not in cdata["year_counts"]
        assert "2025" not in cdata["year_counts"]
        assert result["year_range"]["start"] == 2023
        assert result["year_range"]["end"] == 2024

        # Verify find_papers_within_distance was called only for 2023 and 2024
        assert mock_em.find_papers_within_distance.call_count == 2

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_with_distance_threshold(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution passes distance_threshold to find_papers_within_distance."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.find_papers_within_distance.return_value = {
            "count": 2,
            "papers": [
                {"title": "Paper 1", "session": "ML", "distance": 0.3},
                {"title": "Paper 2", "session": "DL", "distance": 0.4},
            ],
            "total_considered": 50,
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_years_for_conference.return_value = [2024]
        mock_db.get_stats.side_effect = lambda year, conference: {"total_papers": 200}

        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(
            topic_keywords="transformers",
            conferences=["NeurIPS"],
            distance_threshold=0.8,
        )
        result = json.loads(result_str)

        # Verify result
        assert result["topic"] == "transformers"
        assert result["distance_threshold"] == 0.8

        # Verify find_papers_within_distance was called with correct threshold
        mock_em.find_papers_within_distance.assert_called_once_with(
            database=mock_db,
            query="transformers",
            distance_threshold=0.8,
            conferences=["NeurIPS"],
            years=[2024],
            query_embedding=mock_em.generate_embedding.return_value,
        )

        # Verify cleanup
        mock_em.close.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_multiple_conferences(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution with multiple conferences."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        def _find_papers(database, query, distance_threshold, conferences=None, years=None, query_embedding=None):
            conf = conferences[0] if conferences else None
            year = years[0] if years else None
            if conf == "NeurIPS" and year == 2023:
                return {"count": 5, "papers": [], "total_considered": 100}
            elif conf == "NeurIPS" and year == 2024:
                return {"count": 10, "papers": [], "total_considered": 100}
            elif conf == "ICLR" and year == 2023:
                return {"count": 3, "papers": [], "total_considered": 80}
            elif conf == "ICLR" and year == 2024:
                return {"count": 8, "papers": [], "total_considered": 80}
            return {"count": 0, "papers": [], "total_considered": 0}

        mock_em.find_papers_within_distance.side_effect = _find_papers

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db
        mock_db.get_years_for_conference.return_value = [2023, 2024]

        def _get_stats(year, conference):
            if conference == "NeurIPS":
                return {"total_papers": 200}
            return {"total_papers": 150}

        mock_db.get_stats.side_effect = _get_stats

        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(
            topic_keywords="transformers",
            conferences=["NeurIPS", "ICLR"],
        )
        result = json.loads(result_str)

        assert result["conferences"] == ["NeurIPS", "ICLR"]
        assert result["total_papers"] == 26  # 5+10+3+8

        # Verify per-conference data
        assert "NeurIPS" in result["conference_data"]
        assert "ICLR" in result["conference_data"]

        neurips = result["conference_data"]["NeurIPS"]
        assert neurips["year_counts"]["2023"] == 5
        assert neurips["year_counts"]["2024"] == 10
        assert neurips["year_relative"]["2023"] == 2.5  # 5/200*100

        iclr = result["conference_data"]["ICLR"]
        assert iclr["year_counts"]["2023"] == 3
        assert iclr["year_counts"]["2024"] == 8
        assert iclr["year_relative"]["2023"] == 2.0  # 3/150*100

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_search_papers_with_where_clause(self, mock_config, mock_db_class, mock_em_class):
        """Test search_papers tool with custom WHERE clause."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        # Mock search results with recent papers
        from datetime import datetime

        current_year = datetime.now().year
        mock_em.search_similar.return_value = {
            "ids": [["p1", "p2"]],
            "metadatas": [
                [
                    {"title": "Paper 1", "year": current_year, "conference": "NeurIPS", "session": "Oral"},
                    {"title": "Paper 2", "year": current_year - 1, "conference": "NeurIPS", "session": "Oral"},
                ]
            ],
            "documents": [["Abstract 1", "Abstract 2"]],
            "distances": [[0.1, 0.2]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool with WHERE clause
        from abstracts_explorer.mcp_server import search_papers

        where_clause = {"session": "Oral"}
        result_str = search_papers(
            topic_keywords="llm", conference="NeurIPS", where=where_clause, years=[current_year, current_year - 1]
        )
        result = json.loads(result_str)

        # Verify result
        assert result["topic"] == "llm"
        assert result["papers_found"] >= 1

        # Verify search_similar was called with merged WHERE clause (session + conference)
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        where_arg = call_args[1]["where"]
        assert "$and" in where_arg
        assert {"session": "Oral"} in where_arg["$and"]
        assert {"conference": "NeurIPS"} in where_arg["$and"]

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_search_papers_complex_where_clause(self, mock_config, mock_db_class, mock_em_class):
        """Test search_papers with complex WHERE clause using $and."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em

        from datetime import datetime

        current_year = datetime.now().year
        mock_em.search_similar.return_value = {
            "ids": [["p1"]],
            "metadatas": [[{"title": "Paper 1", "year": current_year, "conference": "NeurIPS"}]],
            "documents": [["Abstract 1"]],
            "distances": [[0.1]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool with complex WHERE clause
        from abstracts_explorer.mcp_server import search_papers

        where_clause = {
            "$and": [{"year": {"$gte": 2024}}, {"session": {"$in": ["Oral Session 1", "Spotlight Session"]}}]
        }
        result_str = search_papers(topic_keywords="deep learning", conference="NeurIPS", where=where_clause)

        # Verify result is valid JSON
        result = json.loads(result_str)
        assert result["topic"] == "deep learning"

        # Verify search_similar was called with complex WHERE clause
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        where_arg = call_args[1]["where"]

        # Should preserve the complex $and structure and add conference
        assert "$and" in where_arg
        assert {"year": {"$gte": 2024}} in where_arg["$and"]
        assert {"session": {"$in": ["Oral Session 1", "Spotlight Session"]}} in where_arg["$and"]
        assert {"conference": "NeurIPS"} in where_arg["$and"]

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_cluster_visualization(self, mock_config, mock_load):
        """Test get_cluster_visualization tool with cached results."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        # Setup load_clustering_data mock
        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)

        # db.get_clustering_cache returns cached results
        mock_db.get_clustering_cache.return_value = {
            "points": [
                {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0, "title": "Paper 1"},
                {"id": "p2", "x": 1.0, "y": 1.0, "cluster": 1, "title": "Paper 2"},
            ],
            "statistics": {
                "n_clusters": 2,
                "n_noise": 0,
                "cluster_sizes": {0: 1, 1: 1},
                "total_papers": 2,
            },
            "n_dimensions": 2,
        }

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_cluster_visualization

        result_str = get_cluster_visualization(conferences=["NeurIPS"])
        result = json.loads(result_str)

        # Verify result
        assert result["n_dimensions"] == 2
        assert result["n_points"] == 2
        assert "statistics" in result
        assert result["statistics"]["n_clusters"] == 2
        assert len(result["points"]) == 2

        # Verify cache was queried
        mock_db.get_clustering_cache.assert_called_once()

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_cluster_visualization_no_conference(self, mock_config, mock_load):
        """Test get_cluster_visualization returns error when no conference is specified."""
        from abstracts_explorer.mcp_server import get_cluster_visualization

        result_str = get_cluster_visualization()
        result = json.loads(result_str)

        assert "error" in result
        assert "conference must be specified" in result["error"]

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_cluster_visualization_no_cache(self, mock_config, mock_load):
        """Test get_cluster_visualization returns error when no cache."""
        mock_config_obj = Mock()
        mock_config_obj.collection_name = "papers"
        mock_config_obj.embedding_model = "test-model"
        mock_config.return_value = mock_config_obj

        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)
        mock_db.get_clustering_cache.return_value = None

        from abstracts_explorer.mcp_server import get_cluster_visualization

        result_str = get_cluster_visualization(conferences=["NeurIPS"])
        result = json.loads(result_str)

        assert "error" in result
        assert "No pre-computed clustering data" in result["error"]


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    def test_mcp_server_initialization(self):
        """Test that MCP server initializes correctly."""
        from abstracts_explorer.mcp_server import mcp

        assert mcp is not None
        # Verify tools are registered
        # Note: We can't easily test tool registration without running the server


class TestSearchPapersPaperCardFields:
    """
    Tests that verify search_papers returns the correct fields for paper card display.

    These integration tests ensure that the data returned by search_papers()
    contains all fields needed to correctly render paper cards in the web UI,
    including the fix for missing 'uid' and 'authors' fields.
    """

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="test", conference="ICLR")
        result = json.loads(result_str)

        paper = result["papers"][0]
        assert isinstance(paper["authors"], list)
        assert paper["authors"] == []

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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
        mock_db_class.return_value = mock_db

        from abstracts_explorer.mcp_server import search_papers

        result_str = search_papers(topic_keywords="machine learning", conference="NeurIPS")
        result = json.loads(result_str)

        paper = result["papers"][0]
        # conference field is needed for the conference badge in paper cards
        assert "conference" in paper, "Paper card requires 'conference' field for conference badge"
        assert paper["conference"] == "NeurIPS"

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
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
        """get_paper_details returns an error when no title or paper_id given."""
        from abstracts_explorer.mcp_server import get_paper_details

        result = json.loads(get_paper_details())
        assert "error" in result
        assert "title" in result["error"] or "paper_id" in result["error"]

    def test_lookup_by_title(self):
        """get_paper_details searches by title keyword and returns matching papers."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = [self._make_paper_row()]

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Test Paper"))

        assert "error" not in result
        assert result["papers_found"] == 1
        paper = result["papers"][0]
        assert paper["title"] == "A Test Paper"
        # Authors returned as list (formatting done by DatabaseManager)
        assert paper["authors"] == ["Smith, John", "Doe, Jane"]
        assert paper["paper_pdf_url"] == "https://example.com/paper.pdf"
        assert paper["keywords"] == "deep learning, transformers"
        mock_db.search_papers.assert_called_once_with(keyword="Test Paper", conference=None, year=None, limit=5)

    def test_lookup_by_paper_id(self):
        """get_paper_details performs exact UID/original_id lookup when paper_id is provided."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.get_paper_by_original_id_or_uid.return_value = self._make_paper_row(award="Best Paper")

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
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
        mock_db.search_papers.return_value = [self._make_paper_row()]

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(paper_id="notfound", title="Test Paper"))

        assert "error" not in result
        assert result["papers_found"] == 1
        mock_db.get_paper_by_original_id_or_uid.assert_called_once()
        mock_db.search_papers.assert_called_once()

    def test_authors_with_no_semicolon(self):
        """Single author is returned as one-element list (DatabaseManager already returns list)."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = [self._make_paper_row(authors=["Solo Author"])]

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert result["papers"][0]["authors"] == ["Solo Author"]

    def test_empty_authors_returns_empty_list(self):
        """None or empty authors field is returned as empty list."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = [self._make_paper_row(authors=[])]

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert result["papers"][0]["authors"] == []

    def test_conference_and_year_filters_passed_to_search(self):
        """conference and year parameters are forwarded to search_papers."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = []

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            get_paper_details(title="Paper", conference="ICLR", year=2024, limit=3)

        mock_db.search_papers.assert_called_once_with(keyword="Paper", conference="ICLR", year=2024, limit=3)

    def test_limit_parameter(self):
        """limit parameter controls how many results are requested from DB."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = []

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            get_paper_details(title="Paper", limit=10)

        mock_db.search_papers.assert_called_once_with(keyword="Paper", conference=None, year=None, limit=10)

    def test_no_papers_found(self):
        """No matching papers returns papers_found=0 and empty list."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.return_value = []

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Nonexistent Paper"))

        assert "error" not in result
        assert result["papers_found"] == 0
        assert result["papers"] == []

    def test_exception_returns_error_json(self):
        """Database exception returns error JSON without raising."""
        from abstracts_explorer.mcp_server import get_paper_details

        mock_db = Mock()
        mock_db.search_papers.side_effect = Exception("DB connection failed")

        with patch("abstracts_explorer.mcp_server.DatabaseManager", return_value=mock_db):
            result = json.loads(get_paper_details(title="Paper"))

        assert "error" in result
        assert "DB connection failed" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
