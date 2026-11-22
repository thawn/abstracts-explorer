"""
Tests for the BERTopic clustering pipeline.
"""

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from abstracts_explorer.clustering import (
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_PARAMS,
    DEFAULT_NR_LEVELS,
    DEFAULT_REDUCTION_METHOD,
    ClusteringError,
    clustering_params_for,
)
from abstracts_explorer.clustering.hierarchy_levels import (
    build_level_mapping,
    build_level_mapping_from_tree,
    build_level_mappings,
    build_level_mappings_from_tree,
    compute_level_max_distances,
    compute_level_max_distances_from_tree,
    level_parent_descriptions,
    level_parent_labels,
)
from abstracts_explorer.clustering.labels import (
    LABEL_SEPARATOR,
    combine_label_description,
    parse_llm_label_response,
    split_label_description,
)
from abstracts_explorer.clustering.llm import (
    _retry_after_seconds,
    generate_llm_label,
    is_retryable_llm_error,
)
from abstracts_explorer.clustering.manager import ClusteringManager
from abstracts_explorer.clustering.pipeline import compute_clusters_with_cache, perform_clustering
from abstracts_explorer.clustering.representation import ProjectLLMRepresentation
from abstracts_explorer.clustering.results import (
    build_cluster_centers,
    build_clustering_results,
    build_hierarchy,
    build_hierarchy_levels,
    build_hierarchy_tree,
    build_points,
    build_statistics,
    derive_hierarchy_levels,
    export_results_to_json,
    results_have_flat_labels,
)
from abstracts_explorer.rate_limiter import RateLimitCooldown


def sample_hierarchy() -> pd.DataFrame:
    """Return a small BERTopic-style hierarchy sorted like BERTopic output."""
    rows = [
        {
            "Parent_ID": 4,
            "Parent_Name": "alpha_beta",
            "Topics": [0, 1],
            "Child_Left_ID": 0,
            "Child_Right_ID": 1,
            "Distance": 0.1,
        },
        {
            "Parent_ID": 5,
            "Parent_Name": "gamma_delta",
            "Topics": [2, 3],
            "Child_Left_ID": 2,
            "Child_Right_ID": 3,
            "Distance": 0.2,
        },
        {
            "Parent_ID": 6,
            "Parent_Name": "alpha_beta_gamma_delta",
            "Topics": [0, 1, 2, 3],
            "Child_Left_ID": 4,
            "Child_Right_ID": 5,
            "Distance": 0.3,
        },
    ]
    return pd.DataFrame(rows).sort_values("Parent_ID", ascending=False)


class TestLabels:
    """Tests for short-label and description helpers."""

    def test_parse_two_line_response(self):
        label, description = parse_llm_label_response(
            "Label: Medical Image Uncertainty\nDescription: Papers quantify uncertainty in medical imaging."
        )
        assert label == "Medical Image Uncertainty"
        assert description == "Papers quantify uncertainty in medical imaging."

    def test_parse_json_response(self):
        label, description = parse_llm_label_response(
            '{"label": "Graph Neural Networks", "description": "Papers study message passing on graphs."}'
        )
        assert label == "Graph Neural Networks"
        assert description == "Papers study message passing on graphs."

    def test_parse_markdown_response(self):
        label, description = parse_llm_label_response(
            "**Label:** Graph Neural Networks\n- **Description:** Papers study graphs."
        )
        assert label == "Graph Neural Networks"
        assert description == "Papers study graphs."

    def test_parse_legacy_single_line_response(self):
        label, description = parse_llm_label_response(" 'Topic: Nice Label' ")
        assert label == "Nice Label"
        assert description == ""

    def test_parse_empty_response(self):
        assert parse_llm_label_response("") == ("", "")

    def test_combine_and_split_roundtrip(self):
        encoded = combine_label_description("Short Label", "A longer descriptive sentence.")
        assert encoded == f"Short Label{LABEL_SEPARATOR}A longer descriptive sentence."
        assert split_label_description(encoded) == ("Short Label", "A longer descriptive sentence.")
        assert split_label_description("Short Label") == ("Short Label", "")


class TestConstants:
    """Tests for clustering constants and cache parameters."""

    def test_defaults(self):
        assert DEFAULT_CLUSTERING_METHOD == "bertopic-agglomerative"
        assert DEFAULT_REDUCTION_METHOD == "umap"
        assert DEFAULT_NR_LEVELS == 10
        assert DEFAULT_CLUSTERING_PARAMS["distance_threshold"] == 0.5
        assert DEFAULT_CLUSTERING_PARAMS["pipeline_version"] == 2

    def test_clustering_params_for(self):
        llm_params = clustering_params_for(True)
        ctfidf_params = clustering_params_for(False)
        assert llm_params["representation"] == "llm"
        assert "nr_levels" not in llm_params
        assert "level_scale" not in llm_params
        assert ctfidf_params["representation"] == "ctfidf"


class TestHierarchyLevels:
    """Tests for the BERTopic hierarchy-level adapter."""

    def test_compute_level_max_distances_linear(self):
        distances = compute_level_max_distances(sample_hierarchy(), 3, "linear")
        assert distances == [0.1, 0.2, 0.3]

    def test_compute_level_max_distances_clamps_levels(self):
        distances = compute_level_max_distances(sample_hierarchy(), 10, "linear")
        assert distances == [0.1, 0.2, 0.3]

    def test_compute_level_max_distances_log(self):
        distances = compute_level_max_distances(sample_hierarchy(), 2, "log")
        assert len(distances) == 2
        assert distances == sorted(distances)

    def test_compute_level_max_distances_empty(self):
        assert compute_level_max_distances(pd.DataFrame({"Distance": []}), 3) == []
        assert compute_level_max_distances(sample_hierarchy(), 0) == []

    def test_compute_level_max_distances_invalid_scale(self):
        with pytest.raises(ValueError):
            compute_level_max_distances(sample_hierarchy(), 3, "quadratic")

    def test_build_level_mapping_transitive(self):
        mapping = build_level_mapping(sample_hierarchy(), 0.3, [0, 1, 2, 3])
        assert mapping == {0: 6, 1: 6, 2: 6, 3: 6}

    def test_build_level_mapping_partial(self):
        mapping = build_level_mapping(sample_hierarchy(), 0.1, [0, 1, 2, 3])
        assert mapping == {0: 4, 1: 4, 2: 2, 3: 3}

    def test_build_level_mapping_empty(self):
        assert build_level_mapping(None, 0.5, [0, 1]) == {0: 0, 1: 1}

    def test_build_level_mappings(self):
        levels = build_level_mappings(sample_hierarchy(), [0, 1, 2, 3], 3, "linear")
        assert [level["level"] for level in levels] == [0, 1, 2]
        assert [level["max_distance"] for level in levels] == [0.1, 0.2, 0.3]

    def test_level_parent_labels(self):
        labels = level_parent_labels(sample_hierarchy())
        assert labels == {4: "alpha_beta", 5: "gamma_delta", 6: "alpha_beta_gamma_delta"}
        assert level_parent_labels(None) == {}

    def test_level_parent_descriptions(self):
        hierarchy = sample_hierarchy().copy()
        hierarchy.loc[hierarchy["Parent_ID"] == 4, "Parent_Name"] = combine_label_description(
            "Alpha Beta", "Papers about alpha and beta."
        )
        descriptions = level_parent_descriptions(hierarchy)
        assert descriptions[4] == "Papers about alpha and beta."
        assert descriptions[5] == ""
        assert level_parent_descriptions(None) == {}

    def test_tree_level_helpers_match_dataframe_helpers(self):
        hierarchy = sample_hierarchy()
        tree = build_hierarchy_tree(hierarchy, level_parent_labels(hierarchy), {})
        assert [node["id"] for node in tree] == [6, 5, 4]
        assert compute_level_max_distances_from_tree(tree, 3, "linear") == compute_level_max_distances(
            hierarchy, 3, "linear"
        )
        assert build_level_mappings_from_tree(tree, [0, 1, 2, 3], 3, "linear") == build_level_mappings(
            hierarchy, [0, 1, 2, 3], 3, "linear"
        )
        assert build_level_mapping_from_tree(tree, 0.1, [0, 1, 2, 3]) == build_level_mapping(
            hierarchy, 0.1, [0, 1, 2, 3]
        )


class TestResults:
    """Tests for result assembly."""

    def test_build_statistics(self):
        stats = build_statistics([0, 0, 1, -1])
        assert stats["n_clusters"] == 2
        assert stats["n_noise"] == 1
        assert stats["cluster_sizes"] == {0: 2, 1: 1}
        assert stats["total_papers"] == 4

    def test_build_points_and_centers(self):
        embeddings = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
        metadatas = [
            {"title": "A", "year": 2024, "conference": "X", "session": "S"},
            {"title": "B" * 150, "year": 2025, "conference": "Y", "session": ""},
            {"title": "C", "year": 2025, "conference": "Y", "session": ""},
        ]
        points = build_points(["p1", "p2", "p3"], embeddings, metadatas, [0, 0, 1], max_title_length=20)
        assert points[0]["x"] == 0.0
        assert points[1]["title"].endswith("...")
        assert points[2]["cluster"] == 1
        assert points[2]["conference"] == "Y"

        centers = build_cluster_centers(embeddings, [0, 0, 1])
        assert centers[0] == {"x": 1.0, "y": 1.0}
        assert centers[1] == {"x": 10.0, "y": 10.0}

    def test_build_hierarchy(self):
        points = [
            {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0},
            {"id": "p2", "x": 2.0, "y": 2.0, "cluster": 1},
            {"id": "p3", "x": 10.0, "y": 10.0, "cluster": 2},
            {"id": "p4", "x": 12.0, "y": 12.0, "cluster": 3},
        ]
        stats = build_statistics([0, 1, 2, 3])
        hierarchy = build_hierarchy(
            hierarchical_topics=sample_hierarchy(),
            flat_topic_ids=[0, 1, 2, 3],
            flat_labels={0: "A", 1: "B", 2: "C", 3: "D"},
            flat_descriptions={0: "Papers about A.", 1: "Papers about B."},
            flat_keywords={0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]},
            statistics=stats,
            points=points,
            parent_labels=level_parent_labels(sample_hierarchy()),
            parent_descriptions={4: "Papers about alpha and beta."},
            nr_levels=3,
        )
        assert hierarchy["method"] == "bertopic"
        assert hierarchy["nr_levels"] == 3
        assert hierarchy["n_topics"] == 4

        level0 = hierarchy["levels"][0]
        assert level0["max_distance"] == 0.1
        assert len(level0["clusters"]) == 3
        merged = next(cluster for cluster in level0["clusters"] if cluster["id"] == 4)
        assert merged["label"] == "alpha beta"
        assert merged["description"] == "Papers about alpha and beta."
        assert merged["size"] == 2
        assert merged["source_topic_count"] == 2
        assert merged["center"] == {"x": 1.0, "y": 1.0}
        assert merged["keywords"] == ["a", "b"]
        assert level0["mapping"] == {"0": 4, "1": 4, "2": 2, "3": 3}

        level2 = hierarchy["levels"][2]
        assert len(level2["clusters"]) == 1
        assert level2["clusters"][0]["id"] == 6

    def test_build_hierarchy_empty(self):
        hierarchy = build_hierarchy(
            hierarchical_topics=pd.DataFrame({"Distance": []}),
            flat_topic_ids=[0, 1],
            flat_labels={0: "A", 1: "B"},
            flat_keywords={0: ["a"], 1: ["b"]},
            statistics=build_statistics([0, 1]),
            points=[
                {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0},
                {"id": "p2", "x": 1.0, "y": 1.0, "cluster": 1},
            ],
            parent_labels={},
            nr_levels=10,
        )
        assert hierarchy["levels"] == []

    def test_build_hierarchy_levels_from_tree_matches_dataframe_wrapper(self):
        points = [
            {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0},
            {"id": "p2", "x": 2.0, "y": 2.0, "cluster": 1},
            {"id": "p3", "x": 10.0, "y": 10.0, "cluster": 2},
            {"id": "p4", "x": 12.0, "y": 12.0, "cluster": 3},
        ]
        stats = build_statistics([0, 1, 2, 3])
        hierarchy = sample_hierarchy()
        parent_labels = level_parent_labels(hierarchy)
        tree = build_hierarchy_tree(hierarchy, parent_labels, {4: "Papers about alpha and beta."})
        from_tree = build_hierarchy_levels(
            tree=tree,
            flat_topic_ids=[0, 1, 2, 3],
            flat_labels={0: "A", 1: "B", 2: "C", 3: "D"},
            flat_descriptions={0: "Papers about A."},
            flat_keywords={0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]},
            statistics=stats,
            points=points,
            nr_levels=3,
        )
        from_dataframe = build_hierarchy(
            hierarchical_topics=hierarchy,
            flat_topic_ids=[0, 1, 2, 3],
            flat_labels={0: "A", 1: "B", 2: "C", 3: "D"},
            flat_descriptions={0: "Papers about A."},
            flat_keywords={0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]},
            statistics=stats,
            points=points,
            parent_labels=parent_labels,
            parent_descriptions={4: "Papers about alpha and beta."},
            nr_levels=3,
        )
        assert from_tree == from_dataframe

    def test_derive_hierarchy_levels_uses_tree(self):
        results = {
            "points": [
                {"id": "p1", "x": 0.0, "y": 0.0, "cluster": 0},
                {"id": "p2", "x": 2.0, "y": 2.0, "cluster": 1},
                {"id": "p3", "x": 10.0, "y": 10.0, "cluster": 2},
                {"id": "p4", "x": 12.0, "y": 12.0, "cluster": 3},
            ],
            "statistics": build_statistics([0, 1, 2, 3]),
            "cluster_labels": {0: "A", 1: "B", 2: "C", 3: "D"},
            "cluster_descriptions": {0: "Papers about A."},
            "cluster_keywords": {0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]},
            "hierarchy": {
                "method": "bertopic",
                "n_topics": 4,
                "tree": build_hierarchy_tree(
                    sample_hierarchy(),
                    level_parent_labels(sample_hierarchy()),
                    {4: "Papers about alpha and beta."},
                ),
            },
        }

        derived = derive_hierarchy_levels(results, nr_levels=3)

        assert derived["nr_levels"] == 3
        assert len(derived["levels"]) == 3
        assert derived["levels"][0]["mapping"] == {"0": 4, "1": 4, "2": 2, "3": 3}

    def test_derive_hierarchy_levels_returns_legacy_hierarchy(self):
        legacy = {"method": "bertopic", "nr_levels": 2, "levels": [{"level": 0}]}
        assert derive_hierarchy_levels({"hierarchy": legacy}) is legacy

    def test_build_clustering_results_schema(self):
        embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        results = build_clustering_results(
            paper_ids=["p1", "p2", "p3", "p4"],
            reduced_embeddings=embeddings,
            metadatas=[{"title": f"T{i}"} for i in range(4)],
            topics=[0, 1, 2, 3],
            flat_labels={0: "A", 1: "B", 2: "C", 3: "D"},
            flat_descriptions={0: "Papers about A."},
            flat_keywords={0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]},
            hierarchical_topics=sample_hierarchy(),
            parent_labels=level_parent_labels(sample_hierarchy()),
            parent_descriptions={4: "Papers about A and B."},
        )
        assert set(results) == {
            "points",
            "statistics",
            "n_dimensions",
            "cluster_centers",
            "cluster_labels",
            "cluster_descriptions",
            "cluster_keywords",
            "hierarchy",
        }
        json.dumps(results)
        assert results["n_dimensions"] == 2
        assert results["cluster_descriptions"] == {0: "Papers about A."}
        assert len(results["points"]) == 4
        assert "levels" not in results["hierarchy"]
        assert len(results["hierarchy"]["tree"]) == 3
        assert results["hierarchy"]["tree"][0]["id"] == 6

    def test_export_results(self, tmp_path):
        path = tmp_path / "nested" / "results.json"
        export_results_to_json({"points": []}, path)
        assert json.loads(path.read_text()) == {"points": []}

    def test_results_have_flat_labels(self):
        assert results_have_flat_labels({"cluster_labels": {"0": "A"}, "statistics": {"cluster_sizes": {"0": 1}}})
        assert not results_have_flat_labels({"cluster_labels": {}, "statistics": {"cluster_sizes": {"0": 1}}})


class TestLLMHelpers:
    """Tests for retry-aware label generation."""

    @pytest.fixture(autouse=True)
    def _reset_llm_cooldown(self, monkeypatch):
        import abstracts_explorer.clustering.llm as llm_module

        monkeypatch.setattr(llm_module, "_rate_limit_cooldown", RateLimitCooldown())

    def test_is_retryable(self):
        assert is_retryable_llm_error(ValueError("empty"))
        assert is_retryable_llm_error(TimeoutError("timeout"))
        assert not is_retryable_llm_error(AttributeError("missing client"))

        error = MockError(status_code=429)
        assert is_retryable_llm_error(error)
        error = MockError(status_code=401)
        assert not is_retryable_llm_error(error)
        error = MockError(response=MockError(status_code=503))
        assert is_retryable_llm_error(error)

    def test_generate_label_success(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=3,
            llm_retry_base_delay=1,
            llm_retry_max_delay=2,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        chat = mocker.patch("abstracts_explorer.clustering.llm.chat_completion_text", return_value=" 'Nice Label' ")
        em = mocker.Mock(openai_client=object())

        assert generate_llm_label(em, "system", "user") == "Nice Label"
        assert chat.call_count == 1

    def test_generate_label_uses_embeddings_client(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=1,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        chat = mocker.patch("abstracts_explorer.clustering.llm.chat_completion_text", return_value="Label")
        client = object()
        em = mocker.Mock(openai_client=client)

        assert generate_llm_label(em, "system", "user") == "Label"
        assert chat.call_args.kwargs["client"] is client

    def test_generate_label_retries_empty_then_succeeds(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=3,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        sleep = mocker.patch("abstracts_explorer.clustering.llm.time.sleep")
        chat = mocker.patch("abstracts_explorer.clustering.llm.chat_completion_text", side_effect=["", "Label"])
        em = mocker.Mock(openai_client=object())

        assert generate_llm_label(em, "system", "user") == "Label"
        assert chat.call_count == 2
        sleep.assert_called_once()

    def test_generate_label_retries_retryable_error(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=2,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_rate_limit_delay=60,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        mocker.patch("abstracts_explorer.clustering.llm.time.sleep")
        cooldown = mocker.Mock()
        mocker.patch("abstracts_explorer.clustering.llm._rate_limit_cooldown", cooldown)
        chat = mocker.patch(
            "abstracts_explorer.clustering.llm.chat_completion_text",
            side_effect=[MockError(status_code=429), "Label"],
        )
        em = mocker.Mock(openai_client=object())

        assert generate_llm_label(em, "system", "user") == "Label"
        assert chat.call_count == 2
        assert cooldown.wait.call_count == 2
        cooldown.defer.assert_called_once_with(60)

    def test_generate_label_honors_retry_after_header(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=2,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_rate_limit_delay=60,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        mocker.patch("abstracts_explorer.clustering.llm.time.sleep")
        cooldown = mocker.Mock()
        mocker.patch("abstracts_explorer.clustering.llm._rate_limit_cooldown", cooldown)
        error = MockError(status_code=429, response=Mock(headers={"retry-after": "12"}))
        mocker.patch(
            "abstracts_explorer.clustering.llm.chat_completion_text",
            side_effect=[error, "Label"],
        )
        em = mocker.Mock(openai_client=object())

        assert generate_llm_label(em, "system", "user") == "Label"
        cooldown.defer.assert_called_once_with(12)

    def test_retry_after_http_date(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        error = MockError(
            status_code=429,
            response=Mock(headers={"retry-after": format_datetime(retry_at, usegmt=True)}),
        )

        delay = _retry_after_seconds(error)
        assert delay is not None
        assert 25 <= delay <= 35

    def test_retry_after_invalid_value(self):
        error = MockError(status_code=429, response=Mock(headers={"retry-after": "not-a-date"}))
        assert _retry_after_seconds(error) is None

    def test_generate_label_uses_configured_rate_limit_delay(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=2,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_rate_limit_delay=45,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        mocker.patch("abstracts_explorer.clustering.llm.time.sleep")
        cooldown = mocker.Mock()
        mocker.patch("abstracts_explorer.clustering.llm._rate_limit_cooldown", cooldown)
        mocker.patch(
            "abstracts_explorer.clustering.llm.chat_completion_text",
            side_effect=[MockError(status_code=429), "Label"],
        )
        em = mocker.Mock(openai_client=object())

        assert generate_llm_label(em, "system", "user") == "Label"
        cooldown.defer.assert_called_once_with(45)

    def test_generate_label_exhausts_retries(self, mocker):
        config = mocker.Mock(
            llm_max_attempts=2,
            llm_retry_base_delay=0,
            llm_retry_max_delay=0,
            llm_request_timeout=5,
            cluster_label_model="model",
        )
        mocker.patch("abstracts_explorer.clustering.llm.get_config", return_value=config)
        mocker.patch("abstracts_explorer.clustering.llm.time.sleep")
        chat = mocker.patch("abstracts_explorer.clustering.llm.chat_completion_text", side_effect=ValueError("empty"))
        em = mocker.Mock(openai_client=object())

        with pytest.raises(ValueError):
            generate_llm_label(em, "system", "user")
        assert chat.call_count == 2

    def test_generate_label_missing_client(self):
        with pytest.raises(AttributeError):
            generate_llm_label(object(), "system", "user")


class MockError(Exception):
    """Exception carrying an HTTP status code or response."""

    def __init__(self, status_code=None, response=None):
        super().__init__("mock error")
        self.status_code = status_code
        self.response = response


class TestRepresentation:
    """Tests for the project LLM representation model."""

    def _topic_model(self, docs=None):
        model = Mock()
        model._extract_representative_docs.return_value = (
            {0: docs or ["doc one", "doc two"], 1: ["doc three"]},
            None,
            None,
            None,
        )
        return model

    def test_extract_topics_uses_llm_label(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock())
        label = mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value=" 'Topic: Nice Label' ",
        )

        topics = {0: [("alpha", 1.0), ("beta", 0.9)], 1: [("gamma", 1.0)]}
        updated = representation.extract_topics(model, ["d"], None, topics)

        assert updated[0] == [("Nice Label", 1)]
        assert updated[1] == [("Nice Label", 1)]
        assert label.call_count == 2

    def test_extract_topics_parses_label_and_description(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock())
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label: Nice Label\nDescription: Papers about nice things.",
        )

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        updated = representation.extract_topics(model, ["d"], None, topics)

        assert updated[0] == [(combine_label_description("Nice Label", "Papers about nice things."), 1)]

    def test_extract_topics_falls_back_to_keywords(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock())
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            side_effect=RuntimeError("boom"),
        )

        topics = {0: [("alpha", 1.0), ("beta", 0.9), ("gamma", 0.8)], 1: [("delta", 1.0)]}
        updated = representation.extract_topics(model, ["d"], None, topics)

        assert updated[0] == [("alpha & beta & gamma", 1)]

    def test_extract_topics_empty_label_falls_back(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock())
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="",
        )

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        updated = representation.extract_topics(model, ["d"], None, topics)

        assert updated[0] == [("alpha", 1)]

    def test_extract_topics_shows_progress_for_multi_topic_batch(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock(), show_progress=True)
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label",
        )
        progress = mocker.Mock()
        tqdm_factory = mocker.patch(
            "abstracts_explorer.clustering.representation.tqdm",
            return_value=progress,
        )

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        representation.extract_topics(model, ["d"], None, topics)

        tqdm_factory.assert_called_once_with(
            total=2,
            desc="Labeling clusters (LLM)",
            unit="topic",
        )
        assert progress.update.call_count == 2
        progress.close.assert_called_once()

    def test_extract_topics_skips_progress_for_single_topic_batch(self, mocker):
        model = Mock()
        model._extract_representative_docs.return_value = ({0: ["doc"]}, None, None, None)
        representation = ProjectLLMRepresentation(mocker.Mock(), show_progress=True)
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label",
        )
        tqdm_factory = mocker.patch("abstracts_explorer.clustering.representation.tqdm")

        representation.extract_topics(model, ["d"], None, {0: [("alpha", 1.0)]})

        tqdm_factory.assert_not_called()

    def test_extract_topics_skips_progress_when_disabled(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock(), show_progress=False)
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label",
        )
        tqdm_factory = mocker.patch("abstracts_explorer.clustering.representation.tqdm")

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        representation.extract_topics(model, ["d"], None, topics)

        tqdm_factory.assert_not_called()

    def test_extract_topics_progress_counts_keyword_fallbacks(self, mocker):
        model = self._topic_model()
        representation = ProjectLLMRepresentation(mocker.Mock(), show_progress=True)
        mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            side_effect=RuntimeError("boom"),
        )
        progress = mocker.Mock()
        mocker.patch(
            "abstracts_explorer.clustering.representation.tqdm",
            return_value=progress,
        )

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        representation.extract_topics(model, ["d"], None, topics)

        assert progress.update.call_count == 2
        progress.close.assert_called_once()


class TestClusteringManager:
    """Tests for the BERTopic manager facade."""

    def _manager(self):
        manager = ClusteringManager(Mock(), Mock())
        manager.paper_ids = ["p1", "p2", "p3", "p4"]
        manager.embeddings = np.zeros((4, 8), dtype=float)
        manager.metadatas = [
            {"title": f"Paper {index}", "year": 2025, "conference": "Test", "session": ""} for index in range(4)
        ]
        return manager

    def _fake_topic_model(self):
        model = Mock()
        model.fit_transform.return_value = (
            np.array([0, 0, 1, 1]),
            np.array([0.9, 0.8, 0.7, 0.6]),
        )
        model.vectorizer_model.get_feature_names_out.return_value = np.array(["alpha", "beta"])
        model.c_tf_idf_ = object()
        model._extract_words_per_topic.return_value = {
            0: [("alpha", 1.0), ("beta", 0.5)],
            1: [("beta", 1.0), ("gamma", 0.5)],
        }
        model.get_topic.side_effect = lambda topic: [("Alpha Topic" if topic == 0 else "Beta Topic", 1.0)]
        model.hierarchical_topics.return_value = pd.DataFrame(
            [
                {
                    "Parent_ID": 2,
                    "Parent_Name": "alpha_beta",
                    "Topics": [0, 1],
                    "Child_Left_ID": 0,
                    "Child_Right_ID": 1,
                    "Distance": 0.1,
                }
            ]
        )
        return model

    def test_fit_builds_results(self, mocker):
        manager = self._manager()
        fake_model = self._fake_topic_model()
        mocker.patch("bertopic.BERTopic", return_value=fake_model)
        mocker.patch.object(
            manager,
            "_make_umap",
            return_value=Mock(
                fit_transform=Mock(return_value=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
            ),
        )

        results = manager.fit(use_llm_labels=True)

        assert results["statistics"]["n_clusters"] == 2
        assert results["cluster_labels"] == {0: "Alpha Topic", 1: "Beta Topic"}
        assert results["cluster_keywords"] == {0: ["alpha", "beta"], 1: ["beta", "gamma"]}
        assert manager.cluster_labels is not None
        assert manager.get_cluster_statistics()["total_papers"] == 4
        assert manager.get_clustering_results() is results
        assert len(results["hierarchy"]["tree"]) == 1
        assert results["hierarchy"]["tree"][0]["id"] == 2

    def test_fit_collects_descriptions(self, mocker):
        manager = self._manager()
        fake_model = self._fake_topic_model()
        fake_model.get_topic.side_effect = lambda topic: [
            (
                combine_label_description(
                    "Alpha Topic" if topic == 0 else "Beta Topic",
                    f"Papers about topic {topic}.",
                ),
                1.0,
            )
        ]
        fake_model.hierarchical_topics.return_value = pd.DataFrame(
            [
                {
                    "Parent_ID": 2,
                    "Parent_Name": combine_label_description("Alpha Beta", "Papers spanning both topics."),
                    "Topics": [0, 1],
                    "Child_Left_ID": 0,
                    "Child_Right_ID": 1,
                    "Distance": 0.1,
                }
            ]
        )
        mocker.patch("bertopic.BERTopic", return_value=fake_model)
        mocker.patch.object(
            manager,
            "_make_umap",
            return_value=Mock(
                fit_transform=Mock(return_value=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
            ),
        )

        results = manager.fit(use_llm_labels=True)

        assert results["cluster_labels"] == {0: "Alpha Topic", 1: "Beta Topic"}
        assert results["cluster_descriptions"] == {
            0: "Papers about topic 0.",
            1: "Papers about topic 1.",
        }
        assert manager.cluster_descriptions == results["cluster_descriptions"]
        merged = results["hierarchy"]["tree"][0]
        assert merged["description"] == "Papers spanning both topics."

    def test_fit_passes_show_progress_to_representation(self, mocker):
        manager = self._manager()
        fake_model = self._fake_topic_model()
        mocker.patch("bertopic.BERTopic", return_value=fake_model)
        mocker.patch.object(
            manager, "_make_umap", return_value=Mock(fit_transform=Mock(return_value=np.zeros((4, 2))))
        )
        representation_class = mocker.patch("abstracts_explorer.clustering.manager.ProjectLLMRepresentation")

        manager.fit(use_llm_labels=True, show_progress=True)

        representation_class.assert_called_once_with(manager.embeddings_manager, show_progress=True)

    def test_fit_without_llm_labels(self, mocker):
        manager = self._manager()
        fake_model = self._fake_topic_model()
        mocker.patch("bertopic.BERTopic", return_value=fake_model)
        mocker.patch.object(
            manager, "_make_umap", return_value=Mock(fit_transform=Mock(return_value=np.zeros((4, 2))))
        )

        results = manager.fit(use_llm_labels=False)

        assert results["cluster_labels"] == {0: "alpha & beta", 1: "beta & gamma"}
        fake_model.get_topic.assert_not_called()

    def test_fit_requires_embeddings(self):
        manager = ClusteringManager(Mock(), Mock())
        with pytest.raises(ClusteringError, match="No embeddings loaded"):
            manager.fit()

    def test_fit_without_documents(self):
        manager = self._manager()
        manager.metadatas = []
        with pytest.raises(ClusteringError, match="No documents"):
            manager.fit()

    def test_extract_flat_keywords_failure(self, mocker):
        manager = self._manager()
        manager.topic_model = Mock()
        manager.topic_model.vectorizer_model.get_feature_names_out.return_value = np.array(["a"])
        manager.topic_model.c_tf_idf_ = object()
        manager.topic_model._extract_words_per_topic.side_effect = RuntimeError("boom")

        assert manager._extract_flat_keywords(["doc"] * 4, np.array([0, 0, 1, 1])) == {}

    def test_get_cluster_statistics_without_fit(self):
        manager = ClusteringManager(Mock(), Mock())
        with pytest.raises(ClusteringError):
            manager.get_cluster_statistics()

    def test_export_requires_fit(self, tmp_path):
        manager = ClusteringManager(Mock(), Mock())
        with pytest.raises(ClusteringError):
            manager.export_to_json(tmp_path / "out.json")


def _fake_results():
    return {
        "points": [{"id": "p1", "x": 0, "y": 0, "cluster": 0}],
        "statistics": {"n_clusters": 1, "n_noise": 0, "cluster_sizes": {0: 1}, "total_papers": 1},
        "cluster_centers": {0: {"x": 0, "y": 0}},
        "cluster_labels": {0: "A"},
        "cluster_keywords": {0: ["a"]},
        "hierarchy": {"levels": [], "n_topics": 1},
    }


class TestPipeline:
    """Tests for cached and uncached pipeline orchestration."""

    def _deps(self):
        embeddings = Mock()
        embeddings.collection_name = "papers"
        database = Mock()
        database.get_clustering_cache.return_value = None
        return embeddings, database

    def test_cache_hit_returns_cached(self, mocker):
        embeddings, database = self._deps()
        cached = _fake_results()
        database.get_clustering_cache.return_value = cached
        manager_class = mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager")

        result = compute_clusters_with_cache(embeddings, database, "model", conferences=["A"], years=[2024])

        assert result is cached
        manager_class.assert_not_called()

    def test_cache_hit_backfills_and_updates_entry(self, mocker):
        embeddings, database = self._deps()
        cached = _fake_results()
        database.get_clustering_cache_entry.return_value = {"id": 42, "results": cached}
        backfill = mocker.patch(
            "abstracts_explorer.clustering.pipeline._backfill_missing_labels",
            return_value={"flat_updated": 1, "hierarchy_updated": 0},
        )
        manager_class = mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager")

        result = compute_clusters_with_cache(
            embeddings,
            database,
            "model",
            conferences=["A"],
            years=[2024],
            backfill_missing_labels=True,
        )

        assert result is cached
        backfill.assert_called_once_with(cached, embeddings, database)
        database.update_clustering_cache_results.assert_called_once_with(42, cached)
        database.save_clustering_cache.assert_not_called()
        manager_class.assert_not_called()

    def test_cache_hit_backfill_noop_does_not_update_entry(self, mocker):
        embeddings, database = self._deps()
        cached = _fake_results()
        database.get_clustering_cache_entry.return_value = {"id": 42, "results": cached}
        mocker.patch(
            "abstracts_explorer.clustering.pipeline._backfill_missing_labels",
            return_value={"flat_updated": 0, "hierarchy_updated": 0},
        )

        compute_clusters_with_cache(embeddings, database, "model", backfill_missing_labels=True)

        database.update_clustering_cache_results.assert_not_called()

    def test_cache_miss_does_not_backfill(self, mocker):
        embeddings, database = self._deps()
        database.get_clustering_cache_entry.return_value = None
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)
        backfill = mocker.patch("abstracts_explorer.clustering.pipeline._backfill_missing_labels")

        compute_clusters_with_cache(embeddings, database, "model", backfill_missing_labels=True)

        backfill.assert_not_called()
        database.update_clustering_cache_results.assert_not_called()
        database.save_clustering_cache.assert_called_once()

    def test_ctfidf_cache_hit_does_not_backfill(self, mocker):
        embeddings, database = self._deps()
        cached = _fake_results()
        database.get_clustering_cache.return_value = cached
        backfill = mocker.patch("abstracts_explorer.clustering.pipeline._backfill_missing_labels")

        result = compute_clusters_with_cache(
            embeddings,
            database,
            "model",
            use_llm_labels=False,
            backfill_missing_labels=True,
        )

        assert result is cached
        backfill.assert_not_called()
        database.get_clustering_cache_entry.assert_not_called()

    def test_cache_miss_fits_and_saves(self, mocker):
        embeddings, database = self._deps()
        results = _fake_results()
        manager = mocker.Mock()
        manager.fit.return_value = results
        manager_class = mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)

        result = compute_clusters_with_cache(embeddings, database, "model", conferences=["A"], years=[2024])

        assert result is results
        manager_class.assert_called_once()
        manager.load_embeddings.assert_called_once()
        database.save_clustering_cache.assert_called_once()
        _, kwargs = database.save_clustering_cache.call_args
        assert kwargs["conference"] == "A"
        assert kwargs["year"] == 2024
        assert kwargs["clustering_params"]["representation"] == "llm"
        assert kwargs["clustering_params"]["distance_threshold"] == 0.5

    def test_no_llm_labels_uses_ctfidf_cache_key(self, mocker):
        embeddings, database = self._deps()
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)

        compute_clusters_with_cache(embeddings, database, "model", use_llm_labels=False)

        _, kwargs = database.save_clustering_cache.call_args
        assert kwargs["clustering_params"]["representation"] == "ctfidf"
        manager.fit.assert_called_once()
        assert manager.fit.call_args.kwargs["use_llm_labels"] is False

    def test_force_recomputes(self, mocker):
        embeddings, database = self._deps()
        database.get_clustering_cache.return_value = _fake_results()
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)

        compute_clusters_with_cache(embeddings, database, "model", force=True)

        manager.fit.assert_called_once()
        database.save_clustering_cache.assert_called_once()

    def test_limit_does_not_cache(self, mocker):
        embeddings, database = self._deps()
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)

        compute_clusters_with_cache(embeddings, database, "model", limit=10)

        database.get_clustering_cache.assert_not_called()
        database.save_clustering_cache.assert_not_called()

    def test_perform_clustering_exports(self, mocker, tmp_path):
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)
        mocker.patch("abstracts_explorer.clustering.pipeline.DatabaseManager")
        mocker.patch("abstracts_explorer.clustering.pipeline.EmbeddingsManager")

        results = perform_clustering(output_path=tmp_path / "out.json", limit=5, conferences=["A"])

        assert results == _fake_results()
        manager.load_embeddings.assert_called_once_with(limit=5, conferences=["A"], years=None)
        manager.export_to_json.assert_called_once_with(tmp_path / "out.json")

    def test_perform_clustering_forwards_requests_per_minute(self, mocker):
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)
        mocker.patch("abstracts_explorer.clustering.pipeline.DatabaseManager")
        embedding_class = mocker.patch("abstracts_explorer.clustering.pipeline.EmbeddingsManager")

        perform_clustering(requests_per_minute=30)

        embedding_class.assert_called_once_with(collection_name="papers", requests_per_minute=30)

    def test_perform_clustering_uses_config_rate_limit_by_default(self, mocker):
        manager = mocker.Mock()
        manager.fit.return_value = _fake_results()
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)
        mocker.patch("abstracts_explorer.clustering.pipeline.DatabaseManager")
        embedding_class = mocker.patch("abstracts_explorer.clustering.pipeline.EmbeddingsManager")

        perform_clustering()

        embedding_class.assert_called_once_with(collection_name="papers", requests_per_minute=None)

    def test_perform_clustering_wraps_errors(self, mocker):
        manager = mocker.Mock()
        manager.load_embeddings.side_effect = RuntimeError("boom")
        mocker.patch("abstracts_explorer.clustering.pipeline.ClusteringManager", return_value=manager)
        mocker.patch("abstracts_explorer.clustering.pipeline.DatabaseManager")
        mocker.patch("abstracts_explorer.clustering.pipeline.EmbeddingsManager")

        with pytest.raises(ClusteringError, match="boom"):
            perform_clustering()
