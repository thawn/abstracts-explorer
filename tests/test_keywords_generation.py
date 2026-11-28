"""Tests for generated keyword schema and TF-IDF generation."""

from sqlalchemy import update

from abstracts_explorer.db_models import Paper
from abstracts_explorer.keywords.generation import (
    GENERATOR_VERSION,
    generate_keywords_for_scope,
)
from abstracts_explorer.plugin import LightweightPaper
from tests.helpers import raw_query


def _paper(uid, title, abstract, conference="NeurIPS", year=2025, keywords=None):
    """Build a LightweightPaper for generation tests."""
    return LightweightPaper(
        title=title,
        authors=["Alice"],
        abstract=abstract,
        session="Session",
        poster_position=uid,
        year=year,
        conference=conference,
        keywords=keywords,
    )


class TestGeneratedKeywordsSchema:
    """Schema and serialization tests for Paper.generated_keywords."""

    def test_generated_keywords_round_trip(self, connected_db):
        paper = _paper("p1", "Neural Networks", "Optimization of neural networks.")
        uid = connected_db.add_paper(paper)
        assert uid is not None

        connected_db.session.execute(
            update(Paper).where(Paper.uid == uid).values(generated_keywords="neural networks, optimization")
        )
        connected_db.session.commit()

        stored = connected_db.get_paper({"uid": uid}, limit=1)[0]
        assert stored["generated_keywords"] == ["neural networks", "optimization"]

        summary = connected_db.get_paper_summaries_by_uids([uid])[0]
        assert summary["generated_keywords"] == ["neural networks", "optimization"]

    def test_source_keywords_remain_stored(self, connected_db):
        paper = _paper("p1", "Neural Networks", "Optimization.", keywords=["source keyword"])
        uid = connected_db.add_paper(paper)
        stored = connected_db.get_paper({"uid": uid}, limit=1)[0]
        assert stored["keywords"] == ["source keyword"]


class TestGenerateKeywordsForScope:
    """Generation algorithm tests on a temporary SQLite database."""

    def _seed(self, db):
        db.add_papers(
            [
                _paper("p1", "Gradient Descent Optimization", "We optimize neural networks with gradient descent."),
                _paper(
                    "p2", "Neural Network Regularization", "Regularization prevents overfitting in neural networks."
                ),
                _paper("p3", "Gradient Boosting Trees", "Gradient boosting ensembles decision trees."),
            ]
        )

    def test_fills_generated_keywords_for_scope(self, connected_db):
        self._seed(connected_db)

        updated = generate_keywords_for_scope(connected_db, "NeurIPS", 2025)

        assert updated == 3
        rows = raw_query(connected_db, "SELECT generated_keywords FROM papers")
        assert all(row["generated_keywords"] for row in rows)

    def test_filters_generic_blacklisted_terms(self, connected_db):
        connected_db.add_papers(
            [
                _paper(
                    "p1",
                    "A Study of Neural Networks",
                    "This paper presents results for a method. The study of neural networks is a workshop paper.",
                ),
                _paper("p2", "A Study of Transformers", "This paper presents results for a method and workshop."),
                _paper("p3", "A Study of Optimization", "This paper presents results for a method and course."),
            ]
        )

        generate_keywords_for_scope(connected_db, "NeurIPS", 2025)
        rows = raw_query(connected_db, "SELECT generated_keywords FROM papers")

        combined = " ".join(row["generated_keywords"] for row in rows).lower()
        for banned in ("paper", "study", "results", "method", "workshop", "course"):
            assert banned not in combined
        assert all(not token.isdigit() for token in combined.replace(",", " ").split())

    def test_is_idempotent_without_force(self, connected_db):
        self._seed(connected_db)

        first = generate_keywords_for_scope(connected_db, "NeurIPS", 2025)
        second = generate_keywords_for_scope(connected_db, "NeurIPS", 2025)

        assert first == 3
        assert second == 0

    def test_force_regenerates_existing(self, connected_db):
        self._seed(connected_db)
        generate_keywords_for_scope(connected_db, "NeurIPS", 2025)

        updated = generate_keywords_for_scope(connected_db, "NeurIPS", 2025, force=True)

        assert updated == 3

    def test_dry_run_does_not_write(self, connected_db):
        self._seed(connected_db)

        preview = generate_keywords_for_scope(connected_db, "NeurIPS", 2025, dry_run=True)

        assert preview == 3
        rows = raw_query(connected_db, "SELECT generated_keywords FROM papers")
        assert all(row["generated_keywords"] is None for row in rows)

    def test_scope_is_limited_to_conference_and_year(self, connected_db):
        connected_db.add_papers(
            [
                _paper("p1", "Gradient Descent", "Neural network optimization.", year=2025),
                _paper("p2", "Gradient Descent", "Neural network optimization.", year=2024),
            ]
        )

        updated = generate_keywords_for_scope(connected_db, "NeurIPS", 2025)

        assert updated == 1
        rows = raw_query(connected_db, "SELECT year, generated_keywords FROM papers ORDER BY year")
        assert rows[0]["generated_keywords"] is None
        assert rows[1]["generated_keywords"]

    def test_version_constant_exists(self):
        assert isinstance(GENERATOR_VERSION, str)
        assert GENERATOR_VERSION
