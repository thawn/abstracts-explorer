"""Tests for interesting-paper summaries and recommendations."""

from unittest.mock import Mock

import pytest

from abstracts_explorer.interesting_papers import (
    load_interesting_papers,
    normalize_priorities,
    recommend_related_papers,
    summarize_interesting_papers,
)


def _paper(
    uid: str,
    title: str = None,
    conference: str = "ICLR",
    year: int = 2026,
    keywords=None,
    authors=None,
    session: str = "Oral",
    abstract: str = "An abstract about transformers and alignment.",
):
    return {
        "uid": uid,
        "title": title or f"Paper {uid}",
        "authors": authors or ["Alice", "Bob"],
        "abstract": abstract,
        "session": session,
        "year": year,
        "conference": conference,
        "keywords": keywords or ["transformers", "alignment"],
    }


def _mock_db(papers):
    db = Mock()
    by_uid = {paper["uid"]: paper for paper in papers}
    db.get_papers_by_uids.side_effect = lambda uids: [by_uid[uid] for uid in uids if uid in by_uid]
    db.search_papers.return_value = [paper for paper in papers if paper["uid"] not in {"p1", "p2"}]
    return db


class TestNormalizePriorities:
    def test_normalizes_dict_ratings(self):
        result = normalize_priorities(
            {
                "a": {"priority": 4},
                "b": {"priority": "5", "searchTerm": "llms"},
                "c": {"priority": -2},
            }
        )

        assert result["a"] == {"priority": 4, "searchTerm": "Unknown"}
        assert result["b"] == {"priority": 5, "searchTerm": "llms"}
        assert result["c"] == {"priority": 0, "searchTerm": "Unknown"}

    @pytest.mark.parametrize("value", [4, None, [5], "5"])
    def test_rejects_legacy_scalar_ratings(self, value):
        with pytest.raises(ValueError, match="Invalid rating format"):
            normalize_priorities({"a": value})


class TestLoadInterestingPapers:
    def test_filters_by_conference_and_year(self):
        papers = [
            _paper("p1", conference="ICLR", year=2026),
            _paper("p2", conference="ICLR", year=2026),
            _paper("p3", conference="NeurIPS", year=2025),
        ]
        db = _mock_db(papers)

        loaded = load_interesting_papers(
            db,
            priorities={"p1": {"priority": 5}, "p2": {"priority": 4}, "p3": {"priority": 3}},
            conference="ICLR 2026",
        )

        assert [paper["uid"] for paper in loaded["papers"]] == ["p1", "p2"]
        assert loaded["filters"] == {"conferences": ["ICLR"], "years": [2026]}


class TestSummarizeInterestingPapers:
    def test_summary_statistics_and_papers(self):
        papers = [
            _paper("p1", keywords=["transformers", "alignment"]),
            _paper("p2", title="Second", keywords=["alignment", "rl"]),
            _paper("p3", conference="NeurIPS", year=2025),
        ]
        db = _mock_db(papers)

        summary = summarize_interesting_papers(
            db,
            priorities={
                "p1": {"priority": 5, "searchTerm": "llms"},
                "p2": {"priority": 4, "searchTerm": "llms"},
                "p3": {"priority": 3},
            },
            conference="ICLR",
            years=[2026],
        )

        assert summary["total_interesting"] == 2
        assert summary["rated_count"] == 2
        assert summary["average_rating"] == 4.5
        assert summary["rating_distribution"]["5"] == 1
        assert summary["rating_distribution"]["4"] == 1
        assert summary["conferences"] == {"ICLR": 2}
        assert summary["years"] == {"2026": 2}
        assert summary["top_rated_papers"][0]["uid"] == "p1"
        assert "alignment" in summary["top_keywords"]
        assert [paper["uid"] for paper in summary["papers"]] == ["p1", "p2"]

    def test_empty_summary_has_warning(self):
        db = _mock_db([])
        summary = summarize_interesting_papers(db, priorities={})

        assert summary["total_interesting"] == 0
        assert summary["warning"]


class TestRecommendRelatedPapers:
    def _mock_embeddings(self, query_result):
        em = Mock()
        em.collection.get.return_value = {
            "ids": ["p1", "p2"],
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        em.collection.count.return_value = 4
        em.collection.query.return_value = query_result
        em.parse_chromadb_metadata.side_effect = lambda metadata: {
            **metadata,
            "year": int(metadata.get("year", 0)),
            "authors": metadata.get("authors", []),
            "keywords": metadata.get("keywords", []),
        }
        return em

    def test_recommends_nearest_unrated_papers(self):
        papers = [_paper("p1"), _paper("p2")]
        db = _mock_db(papers)
        em = self._mock_embeddings(
            {
                "ids": [["p1", "p3", "p4"]],
                "distances": [[0.0, 0.2, 0.4]],
                "metadatas": [
                    [
                        {"uid": "p1", "year": "2026", "conference": "ICLR"},
                        {
                            "uid": "p3",
                            "title": "Recommended",
                            "year": "2026",
                            "conference": "ICLR",
                            "keywords": ["alignment"],
                            "authors": ["Alice"],
                            "session": "Oral",
                        },
                        {"uid": "p4", "title": "Other", "year": "2026", "conference": "ICLR"},
                    ]
                ],
                "documents": [["", "abstract three", "abstract four"]],
            }
        )

        result = recommend_related_papers(
            db,
            em,
            priorities={"p1": {"priority": 5}, "p2": {"priority": 1}},
            conferences=["ICLR"],
            years=[2026],
            n_results=2,
        )

        assert result["seed_count"] == 2
        assert [paper["uid"] for paper in result["papers"]] == ["p3", "p4"]
        assert result["papers"][0]["recommendation_score"] == 0.8
        assert any("alignment" in reason for reason in result["papers"][0]["recommendation_reasons"])
        # Seed papers are excluded from recommendations.
        assert all(paper["uid"] not in {"p1", "p2"} for paper in result["papers"])

    def test_infers_scope_from_seed_papers(self):
        papers = [_paper("p1"), _paper("p2")]
        db = _mock_db(papers)
        em = self._mock_embeddings({"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]})

        recommend_related_papers(db, em, priorities={"p1": {"priority": 5}, "p2": {"priority": 1}}, n_results=5)

        where = em.collection.query.call_args.kwargs["where"]
        assert {"conference": {"$in": ["ICLR"]}} in where["$and"]
        assert {"year": {"$in": ["2026"]}} in where["$and"]

    def test_falls_back_to_metadata_when_embeddings_missing(self):
        papers = [_paper("p1")]
        db = _mock_db(papers)
        db.search_papers.return_value = [_paper("p3", title="Keyword Match")]
        em = Mock()
        em.collection.get.return_value = {"ids": ["p1"], "embeddings": None}

        result = recommend_related_papers(db, em, priorities={"p1": {"priority": 5}}, n_results=5)

        assert result["papers"]
        assert result["papers"][0]["uid"] == "p3"
        assert result["warning"]


class TestConferenceYearParsing:
    def test_apostrophe_year_is_split(self):
        papers = [_paper("p1", conference="ICLR", year=2026)]
        db = _mock_db(papers)

        loaded = load_interesting_papers(db, priorities={"p1": {"priority": 5}}, conference="ICLR'26")

        assert loaded["filters"] == {"conferences": ["ICLR"], "years": [2026]}
        assert [paper["uid"] for paper in loaded["papers"]] == ["p1"]
