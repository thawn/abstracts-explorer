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
        where = {
            "$or": [
                {"$and": [{"conference": "ICML"}, {"year": 2024}]},
                {"session": "Oral"}
            ]
        }
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

        with patch("abstracts_explorer.mcp_server.DatabaseManager") as mock_db_class, \
             patch("abstracts_explorer.mcp_server.ClusteringManager") as mock_cm_class:
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
            {"title": "Paper 1", "keywords": "ml, ai", "session": "ML Track", "year": 2023},
            {"title": "Paper 2", "keywords": "dl, nn", "session": "ML Track", "year": 2023},
            {"title": "Paper 3", "keywords": "nlp, transformers", "session": "NLP Track", "year": 2024},
            {"title": "Paper 4", "keywords": "ml, dl", "session": "ML Track", "year": 2024},
            {"title": "Paper 5", "keywords": "nlp, bert", "session": "NLP Track", "year": 2024},
            {"title": "Paper 6", "keywords": "cv, vision", "session": "CV Track", "year": 2025},
        ]

        db = Mock(spec=DatabaseManager)

        # Analyze cluster 0 (papers 0, 1, 3)
        result = analyze_cluster_topics(cm, db, cluster_id=0)

        assert result["cluster_id"] == 0
        assert result["paper_count"] == 3
        assert len(result["sample_titles"]) == 3
        assert result["sample_titles"][0] == "Paper 1"
        
        # Check keywords
        keyword_dict = {k["keyword"]: k["count"] for k in result["keywords"]}
        assert keyword_dict["ml"] == 2  # appears in papers 0 and 3
        assert keyword_dict["dl"] == 2  # appears in papers 1 and 3

        # Check sessions
        session_dict = {s["session"]: s["count"] for s in result["sessions"]}
        assert session_dict["ML Track"] == 3

        # Check years
        assert result["years"][2023] == 2
        assert result["years"][2024] == 1

    def test_analyze_empty_cluster(self):
        """Test analyzing a cluster with no papers."""
        cm = Mock(spec=ClusteringManager)
        cm.cluster_labels = np.array([0, 0, 1, 1])
        cm.paper_ids = ["p1", "p2", "p3", "p4"]
        cm.metadatas = [
            {"title": "Paper 1", "keywords": "ml", "session": "ML", "year": 2023},
            {"title": "Paper 2", "keywords": "dl", "session": "DL", "year": 2023},
            {"title": "Paper 3", "keywords": "nlp", "session": "NLP", "year": 2024},
            {"title": "Paper 4", "keywords": "cv", "session": "CV", "year": 2024},
        ]

        db = Mock(spec=DatabaseManager)

        # Analyze cluster 5 (doesn't exist)
        result = analyze_cluster_topics(cm, db, cluster_id=5)

        assert result["cluster_id"] == 5
        assert result["paper_count"] == 0
        assert result["keywords"] == []
        assert result["sessions"] == []
        assert result["years"] == {}
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
            {"title": "Paper 2", "keywords": "ml", "year": 2023},  # Missing session
        ]

        db = Mock(spec=DatabaseManager)

        result = analyze_cluster_topics(cm, db, cluster_id=0)

        assert result["cluster_id"] == 0
        assert result["paper_count"] == 2
        assert len(result["sample_titles"]) == 2
        # Should handle missing fields gracefully
        assert isinstance(result["keywords"], list)
        assert isinstance(result["sessions"], list)
        assert isinstance(result["years"], dict)


class TestMCPTools:
    """Tests for MCP tool functions."""

    @patch("abstracts_explorer.mcp_server.load_clustering_data")
    @patch("abstracts_explorer.mcp_server.analyze_cluster_topics")
    def test_get_cluster_topics(self, mock_analyze, mock_load):
        """Test get_cluster_topics tool."""
        # Setup mocks
        mock_cm = Mock()
        mock_cm.embeddings_manager = Mock()
        mock_db = Mock()
        mock_load.return_value = (mock_cm, mock_db)

        mock_cm.load_embeddings.return_value = 100
        mock_cm.cluster.return_value = np.array([0, 0, 1, 1])
        mock_cm.reduce_dimensions.return_value = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mock_cm.get_cluster_statistics.return_value = {
            "n_clusters": 2,
            "n_noise": 0,
            "cluster_sizes": {0: 2, 1: 2},
            "total_papers": 4,
        }

        mock_analyze.side_effect = [
            {
                "cluster_id": 0,
                "paper_count": 2,
                "keywords": [{"keyword": "ml", "count": 2}],
                "sessions": [{"session": "ML Track", "count": 2}],
                "years": {2023: 2},
                "sample_titles": ["Paper 1", "Paper 2"],
            },
            {
                "cluster_id": 1,
                "paper_count": 2,
                "keywords": [{"keyword": "nlp", "count": 2}],
                "sessions": [{"session": "NLP Track", "count": 2}],
                "years": {2024: 2},
                "sample_titles": ["Paper 3", "Paper 4"],
            },
        ]

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_cluster_topics

        result_str = get_cluster_topics(n_clusters=2)
        result = json.loads(result_str)

        # Verify result
        assert "statistics" in result
        assert result["statistics"]["n_clusters"] == 2
        assert "clusters" in result
        assert len(result["clusters"]) == 2
        assert result["clusters"][0]["cluster_id"] == 0
        assert result["clusters"][1]["cluster_id"] == 1

        # Verify cleanup
        mock_cm.embeddings_manager.close.assert_called_once()
        mock_db.close.assert_called_once()

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
        mock_em.search_similar.return_value = {
            "ids": [["p1", "p2", "p3"]],
            "metadatas": [[
                {"title": "Paper 1", "year": 2023, "session": "ML"},
                {"title": "Paper 2", "year": 2023, "session": "DL"},
                {"title": "Paper 3", "year": 2024, "session": "ML"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(topic_keywords="transformers")
        result = json.loads(result_str)

        # Verify result
        assert result["topic"] == "transformers"
        assert result["total_papers"] == 3
        assert result["year_counts"]["2023"] == 2  # Keys are strings after JSON conversion
        assert result["year_counts"]["2024"] == 1
        assert "2023" in result["papers_by_year"]
        assert "2024" in result["papers_by_year"]

        # Verify cleanup
        mock_em.close.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_with_where_clause(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution tool with custom WHERE clause."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["p1", "p2"]],
            "metadatas": [[
                {"title": "Paper 1", "year": 2024, "session": "Oral Session 1"},
                {"title": "Paper 2", "year": 2024, "session": "Oral Session 1"},
            ]],
            "distances": [[0.1, 0.2]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool with WHERE clause
        from abstracts_explorer.mcp_server import get_topic_evolution

        where_clause = {"session": {"$in": ["Oral Session 1"]}}
        result_str = get_topic_evolution(
            topic_keywords="transformers",
            where=where_clause
        )
        result = json.loads(result_str)

        # Verify result
        assert result["topic"] == "transformers"
        assert result["total_papers"] == 2

        # Verify search_similar was called with WHERE clause
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        assert call_args[1]["where"] == where_clause

        # Verify cleanup
        mock_em.close.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_with_where_and_conference(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution merges WHERE clause with conference parameter."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup embeddings manager mock
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_em.search_similar.return_value = {
            "ids": [["p1"]],
            "metadatas": [[{"title": "Paper 1", "year": 2024, "session": "Oral"}]],
            "distances": [[0.1]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool
        from abstracts_explorer.mcp_server import get_topic_evolution

        where_clause = {"year": {"$gte": 2024}}
        result_str = get_topic_evolution(
            topic_keywords="transformers",
            conference="NeurIPS",
            where=where_clause
        )
        
        # Verify result is valid JSON
        result = json.loads(result_str)
        assert result["topic"] == "transformers"

        # Verify search_similar was called with merged WHERE clause
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        where_arg = call_args[1]["where"]
        
        # Should contain $and with both conditions
        assert "$and" in where_arg
        assert {"year": {"$gte": 2024}} in where_arg["$and"]
        assert {"conference": "NeurIPS"} in where_arg["$and"]

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_recent_developments_with_where_clause(self, mock_config, mock_db_class, mock_em_class):
        """Test get_recent_developments tool with custom WHERE clause."""
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
            "metadatas": [[
                {"title": "Paper 1", "year": current_year, "conference": "NeurIPS", "session": "Oral"},
                {"title": "Paper 2", "year": current_year - 1, "conference": "NeurIPS", "session": "Oral"},
            ]],
            "documents": [["Abstract 1", "Abstract 2"]],
            "distances": [[0.1, 0.2]],
        }

        # Setup database mock
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool with WHERE clause
        from abstracts_explorer.mcp_server import get_recent_developments

        where_clause = {"session": "Oral"}
        result_str = get_recent_developments(
            topic_keywords="llm",
            where=where_clause,
            n_years=2
        )
        result = json.loads(result_str)

        # Verify result
        assert result["topic"] == "llm"
        assert result["papers_found"] >= 1

        # Verify search_similar was called with WHERE clause
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        assert call_args[1]["where"] == where_clause

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_recent_developments_complex_where_clause(self, mock_config, mock_db_class, mock_em_class):
        """Test get_recent_developments with complex WHERE clause using $and."""
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
        from abstracts_explorer.mcp_server import get_recent_developments

        where_clause = {
            "$and": [
                {"year": {"$gte": 2024}},
                {"session": {"$in": ["Oral Session 1", "Spotlight Session"]}}
            ]
        }
        result_str = get_recent_developments(
            topic_keywords="deep learning",
            where=where_clause
        )
        
        # Verify result is valid JSON
        result = json.loads(result_str)
        assert result["topic"] == "deep learning"

        # Verify search_similar was called with complex WHERE clause
        mock_em.search_similar.assert_called_once()
        call_args = mock_em.search_similar.call_args
        where_arg = call_args[1]["where"]
        
        # Should preserve the complex $and structure
        assert "$and" in where_arg
        assert {"year": {"$gte": 2024}} in where_arg["$and"]
        assert {"session": {"$in": ["Oral Session 1", "Spotlight Session"]}} in where_arg["$and"]

    @patch("abstracts_explorer.mcp_server.EmbeddingsManager")
    @patch("abstracts_explorer.mcp_server.DatabaseManager")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_topic_evolution_invalid_where_type(self, mock_config, mock_db_class, mock_em_class):
        """Test get_topic_evolution with invalid WHERE clause type."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup mocks
        mock_em = Mock()
        mock_em_class.return_value = mock_em
        mock_db = Mock()
        mock_db_class.return_value = mock_db

        # Import and call the tool with invalid WHERE type
        from abstracts_explorer.mcp_server import get_topic_evolution

        result_str = get_topic_evolution(
            topic_keywords="transformers",
            where="invalid string"  # Invalid: should be dict or None
        )
        result = json.loads(result_str)

        # Should return error
        assert "error" in result
        assert "Invalid WHERE clause" in result["error"]

    @patch("abstracts_explorer.mcp_server.perform_clustering")
    @patch("abstracts_explorer.mcp_server.get_config")
    def test_get_cluster_visualization(self, mock_config, mock_perform):
        """Test get_cluster_visualization tool."""
        # Setup config mock
        mock_config_obj = Mock()
        mock_config_obj.embedding_db_path = "chroma_db"
        mock_config_obj.collection_name = "papers"
        mock_config_obj.paper_db_path = "abstracts.db"
        mock_config.return_value = mock_config_obj

        # Setup clustering mock
        mock_perform.return_value = {
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

        result_str = get_cluster_visualization(n_clusters=2)
        result = json.loads(result_str)

        # Verify result
        assert result["n_dimensions"] == 2
        assert result["n_points"] == 2
        assert "statistics" in result
        assert result["statistics"]["n_clusters"] == 2
        assert len(result["points"]) == 2


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    def test_mcp_server_initialization(self):
        """Test that MCP server initializes correctly."""
        from abstracts_explorer.mcp_server import mcp

        assert mcp is not None
        # Verify tools are registered
        # Note: We can't easily test tool registration without running the server


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
