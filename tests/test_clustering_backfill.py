"""
Tests for backfilling missing cluster descriptions from cached clustering results.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from abstracts_explorer.clustering.backfill import (
    CachedLabelRepresentation,
    backfill_missing_labels,
    find_flat_clusters_missing_descriptions,
    find_hierarchy_clusters_missing_descriptions,
)
from abstracts_explorer.clustering.errors import ClusteringError
from abstracts_explorer.clustering.labels import combine_label_description


def _flat_results():
    return {
        "points": [
            {"id": "p0", "cluster": 0, "title": "Paper zero"},
            {"id": "p1", "cluster": 1, "title": "Paper one"},
        ],
        "statistics": {"cluster_sizes": {0: 1, 1: 1}},
        "cluster_labels": {0: "Fallback Label", 1: "Good Label"},
        "cluster_descriptions": {1: "Papers about good things."},
        "cluster_keywords": {0: ["alpha"], 1: ["beta"]},
        "hierarchy": {"levels": []},
    }


def _hierarchy_results():
    results = _flat_results()
    results["hierarchy"] = {
        "nr_levels": 1,
        "level_scale": "linear",
        "levels": [
            {
                "level": 0,
                "max_distance": 0.1,
                "mapping": {"0": 2, "1": 2},
                "clusters": [
                    {
                        "id": 2,
                        "label": "Fallback Parent",
                        "description": "",
                        "size": 2,
                        "center": {"x": 0.0, "y": 0.0},
                        "keywords": ["alpha", "beta"],
                        "source_topic_count": 2,
                    }
                ],
            }
        ],
    }
    return results


def _tree_results():
    results = _flat_results()
    results["cluster_descriptions"] = {0: "Papers about zero.", 1: "Papers about good things."}
    results["hierarchy"] = {
        "method": "bertopic",
        "n_topics": 2,
        "tree": [
            {
                "id": 2,
                "topics": [0, 1],
                "label": "Old Parent",
                "description": "",
                "distance": 0.1,
                "child_left_id": 0,
                "child_right_id": 1,
            }
        ],
    }
    return results


def _papers(*uids):
    return [
        {"uid": uid, "title": f"Title {uid}", "abstract": f"Abstract {uid}", "year": 2024, "conference": "Test"}
        for uid in uids
    ]


class TestDetection:
    """Tests for missing-description detection."""

    def test_flat_missing_handles_string_keys(self):
        results = _flat_results()
        results["cluster_labels"] = {"0": "Fallback Label", "1": "Good Label"}
        results["cluster_descriptions"] = {"1": "Papers about good things."}

        assert find_flat_clusters_missing_descriptions(results) == [0]

    def test_flat_missing_treats_whitespace_as_missing(self):
        results = _flat_results()
        results["cluster_descriptions"] = {0: "   ", 1: "Good description."}

        assert find_flat_clusters_missing_descriptions(results) == [0]

    def test_flat_nothing_missing(self):
        results = _flat_results()
        results["cluster_descriptions"] = {
            0: "Papers about alpha.",
            1: "Papers about good things.",
        }

        assert find_flat_clusters_missing_descriptions(results) == []

    def test_hierarchy_missing_returns_parent_members(self):
        results = _hierarchy_results()

        assert find_hierarchy_clusters_missing_descriptions(results) == {2: frozenset({0, 1})}

    def test_hierarchy_missing_ignores_flat_level_clusters(self):
        results = _hierarchy_results()
        results["hierarchy"]["levels"][0]["mapping"] = {"0": 0, "1": 1}
        results["hierarchy"]["levels"][0]["clusters"] = [
            {
                "id": 0,
                "label": "Fallback Label",
                "description": "",
                "size": 1,
                "center": {"x": 0.0, "y": 0.0},
                "keywords": ["alpha"],
                "source_topic_count": 1,
            }
        ]

        assert find_hierarchy_clusters_missing_descriptions(results) == {}


class TestCachedLabelRepresentation:
    """Tests for reusing cached labels inside the BERTopic representation model."""

    def _topic_model(self, mappings):
        model = Mock()
        model._extract_representative_docs.return_value = (mappings, None, None, None)
        return model

    def test_reuses_cached_flat_label_without_llm(self, mocker):
        docs = pd.DataFrame({"Document": ["d0", "d1"], "ID": [0, 1], "Topic": [0, 1]})
        model = self._topic_model({1: ["d1"]})
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={0: "Old", 1: "Good"},
            flat_descriptions={1: "Good description."},
            hierarchy_labels={},
            flat_topics=[0, 1],
        )
        llm = mocker.patch("abstracts_explorer.clustering.representation.generate_llm_label")

        topics = {1: [("beta", 1.0)]}
        updated = representation.extract_topics(model, docs, None, topics)

        assert updated[1] == [(combine_label_description("Good", "Good description."), 1)]
        llm.assert_not_called()

    def test_llm_only_for_missing_flat_label(self, mocker):
        docs = pd.DataFrame({"Document": ["d0", "d1"], "ID": [0, 1], "Topic": [0, 1]})
        model = self._topic_model({0: ["d0"], 1: ["d1"]})
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={0: "Old", 1: "Good"},
            flat_descriptions={1: "Good description."},
            hierarchy_labels={},
            flat_topics=[0, 1],
        )
        llm = mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label: New\nDescription: New description.",
        )

        topics = {0: [("alpha", 1.0)], 1: [("beta", 1.0)]}
        updated = representation.extract_topics(model, docs, None, topics)

        assert updated[0] == [(combine_label_description("New", "New description."), 1)]
        assert updated[1] == [(combine_label_description("Good", "Good description."), 1)]
        assert llm.call_count == 1

    def test_reuses_cached_hierarchy_label_by_member_set(self, mocker):
        docs = pd.DataFrame({"Document": ["d0", "d1"], "ID": [0, 1], "Topic": [0, 0]})
        model = self._topic_model({0: ["d0", "d1"]})
        cached = combine_label_description("Parent", "Parent description.")
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={0: "A", 1: "B"},
            flat_descriptions={0: "A desc.", 1: "B desc."},
            hierarchy_labels={frozenset({0, 1}): cached},
            flat_topics=[0, 1],
        )
        llm = mocker.patch("abstracts_explorer.clustering.representation.generate_llm_label")

        updated = representation.extract_topics(model, docs, None, {0: [("alpha", 1.0)]})

        assert updated[0] == [(cached, 1)]
        llm.assert_not_called()

    def test_outlier_topic_never_calls_llm(self, mocker):
        docs = pd.DataFrame({"Document": ["d0"], "ID": [0], "Topic": [-1]})
        model = self._topic_model({-1: ["d0"]})
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={},
            flat_descriptions={},
            hierarchy_labels={},
            flat_topics=[-1],
        )
        llm = mocker.patch("abstracts_explorer.clustering.representation.generate_llm_label")

        updated = representation.extract_topics(model, docs, None, {-1: [("alpha", 1.0)]})

        assert updated[-1] == [("Outliers", 1)]
        llm.assert_not_called()


class TestTreeBackfill:
    """Tests for tree-aware backfill behavior."""

    def test_tree_missing_detection(self):
        results = _tree_results()

        assert find_hierarchy_clusters_missing_descriptions(results) == {2: frozenset({0, 1})}

    def test_tree_missing_ignores_described_nodes(self):
        results = _tree_results()
        results["hierarchy"]["tree"][0]["description"] = "Described parent."

        assert find_hierarchy_clusters_missing_descriptions(results) == {}

    def test_tree_backfill_updates_tree_nodes(self, mocker):
        results = _tree_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0", "p1")

        fake_model = Mock()
        fake_model.hierarchical_topics.return_value = pd.DataFrame(
            [
                {
                    "Topics": [0, 1],
                    "Parent_Name": combine_label_description("New Parent", "New parent description."),
                }
            ]
        )
        mocker.patch("abstracts_explorer.clustering.backfill.BERTopic", return_value=fake_model)

        report = backfill_missing_labels(results, mocker.Mock(), database)

        assert report["hierarchy_updated"] == 1
        assert results["hierarchy"]["tree"][0]["label"] == "New Parent"
        assert results["hierarchy"]["tree"][0]["description"] == "New parent description."

    def test_uncached_non_target_hierarchy_members_skip_llm(self, mocker):
        docs = pd.DataFrame({"Document": ["d0", "d1"], "ID": [0, 1], "Topic": [0, 0]})
        model = Mock()
        model._extract_representative_docs.return_value = ({0: ["d0", "d1"]}, None, None, None)
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={0: "A", 1: "B"},
            flat_descriptions={0: "A desc.", 1: "B desc."},
            hierarchy_labels={},
            flat_topics=[0, 1],
            target_flat_ids=set(),
            target_hierarchy_members={frozenset({0, 1, 2})},
        )
        llm = mocker.patch("abstracts_explorer.clustering.representation.generate_llm_label")

        updated = representation.extract_topics(model, docs, None, {0: [("alpha", 1.0)]})

        assert updated[0] == [(combine_label_description("Merged cluster", ""), 1)]
        llm.assert_not_called()

    def test_target_hierarchy_members_use_llm(self, mocker):
        docs = pd.DataFrame({"Document": ["d0", "d1"], "ID": [0, 1], "Topic": [0, 0]})
        model = Mock()
        model._extract_representative_docs.return_value = ({0: ["d0", "d1"]}, None, None, None)
        representation = CachedLabelRepresentation(
            mocker.Mock(),
            flat_labels={0: "A", 1: "B"},
            flat_descriptions={0: "A desc.", 1: "B desc."},
            hierarchy_labels={},
            flat_topics=[0, 1],
            target_flat_ids=set(),
            target_hierarchy_members={frozenset({0, 1})},
        )
        llm = mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label: New\nDescription: New description.",
        )

        updated = representation.extract_topics(model, docs, None, {0: [("alpha", 1.0)]})

        assert updated[0] == [(combine_label_description("New", "New description."), 1)]
        assert llm.call_count == 1


class TestBackfill:
    """Tests for the backfill orchestration."""

    def test_no_missing_is_a_noop(self, mocker):
        results = _flat_results()
        results["cluster_descriptions"] = {0: "A desc.", 1: "Good description."}
        load = mocker.patch("abstracts_explorer.clustering.backfill._load_documents")
        bertopic = mocker.patch("abstracts_explorer.clustering.backfill.BERTopic")

        report = backfill_missing_labels(results, mocker.Mock(), mocker.Mock())

        assert report["missing"] == 0
        assert report["flat_updated"] == 0
        assert report["hierarchy_updated"] == 0
        load.assert_not_called()
        bertopic.assert_not_called()

    def test_missing_paper_raises(self):
        results = _flat_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0")

        with pytest.raises(ClusteringError, match="missing paper metadata"):
            backfill_missing_labels(results, Mock(), database)

    def test_empty_document_raises(self):
        results = _flat_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = [
            {"uid": "p0", "title": "", "abstract": ""},
            {"uid": "p1", "title": "Title", "abstract": "Abstract"},
        ]

        with pytest.raises(ClusteringError, match="no abstract and no title|cannot create embedding text"):
            backfill_missing_labels(results, Mock(), database)

    def test_updates_flat_and_hierarchy_labels(self, mocker):
        results = _hierarchy_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0", "p1")

        fake_model = Mock()
        fake_model.get_topic.side_effect = lambda topic: (
            [(combine_label_description("New Flat", "New flat description."), 1)] if topic == 0 else []
        )
        fake_model.hierarchical_topics.return_value = pd.DataFrame(
            [
                {
                    "Topics": [0, 1],
                    "Parent_Name": combine_label_description("New Parent", "New parent description."),
                }
            ]
        )
        bertopic = mocker.patch("abstracts_explorer.clustering.backfill.BERTopic", return_value=fake_model)

        report = backfill_missing_labels(results, mocker.Mock(), database)

        bertopic.assert_called_once()
        assert fake_model.update_topics.call_args.kwargs["topics"] == [0, 1]
        assert results["cluster_labels"][0] == "New Flat"
        assert results["cluster_descriptions"][0] == "New flat description."
        assert results["cluster_labels"][1] == "Good Label"
        assert results["cluster_descriptions"][1] == "Papers about good things."
        assert results["hierarchy"]["levels"][0]["clusters"][0]["label"] == "New Parent"
        assert results["hierarchy"]["levels"][0]["clusters"][0]["description"] == "New parent description."
        assert report["flat_updated"] == 1
        assert report["hierarchy_updated"] == 1
        assert report["failed"] == 0

    def test_failed_flat_attempt_keeps_existing_label(self, mocker):
        results = _flat_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0", "p1")

        fake_model = Mock()
        fake_model.get_topic.return_value = [("alpha", 1.0)]
        mocker.patch("abstracts_explorer.clustering.backfill.BERTopic", return_value=fake_model)

        report = backfill_missing_labels(results, mocker.Mock(), database)

        assert results["cluster_labels"][0] == "Fallback Label"
        assert 0 not in results["cluster_descriptions"]
        assert report["flat_updated"] == 0
        assert report["failed"] == 1

    def test_real_bertopic_reuses_cached_flat_labels(self, mocker):
        results = _flat_results()
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0", "p1")
        llm = mocker.patch(
            "abstracts_explorer.clustering.representation.generate_llm_label",
            return_value="Label: New Flat\nDescription: New flat description.",
        )

        report = backfill_missing_labels(results, mocker.Mock(), database)

        assert llm.call_count == 1
        assert results["cluster_labels"][0] == "New Flat"
        assert results["cluster_descriptions"][0] == "New flat description."
        assert results["cluster_labels"][1] == "Good Label"
        assert report["flat_updated"] == 1
        assert report["failed"] == 0

    def test_second_run_is_noop(self, mocker):
        results = _flat_results()
        results["cluster_descriptions"] = {0: "A desc.", 1: "Good description."}
        database = Mock()
        database.get_paper_summaries_by_uids.return_value = _papers("p0", "p1")
        llm = mocker.patch("abstracts_explorer.clustering.representation.generate_llm_label")
        bertopic = mocker.patch("abstracts_explorer.clustering.backfill.BERTopic")

        report = backfill_missing_labels(results, mocker.Mock(), database)

        assert report["missing"] == 0
        llm.assert_not_called()
        bertopic.assert_not_called()
