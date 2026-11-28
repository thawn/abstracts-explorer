"""Tests for the generated-keyword backfill script."""

import importlib.util
import json
import sqlite3
from pathlib import Path


from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.plugin import LightweightPaper
from tests.conftest import set_test_db
from tests.helpers import raw_query

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "backfill_generated_keywords.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("backfill_generated_keywords", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper(uid, title, abstract, conference="NeurIPS", year=2025):
    return LightweightPaper(
        title=title,
        authors=["Alice"],
        abstract=abstract,
        session="Session",
        poster_position=uid,
        year=year,
        conference=conference,
    )


def _seed(db):
    db.add_papers(
        [
            _paper("p1", "Gradient Descent Optimization", "We optimize neural networks with gradient descent."),
            _paper("p2", "Neural Network Regularization", "Regularization prevents overfitting in neural networks."),
            _paper("p3", "Gradient Boosting Trees", "Gradient boosting ensembles decision trees."),
        ]
    )


class TestEnsureColumn:
    def test_adds_column_to_old_schema(self, tmp_path):
        db_path = tmp_path / "old.db"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE papers (uid VARCHAR(16) PRIMARY KEY, title TEXT NOT NULL, "
            "abstract TEXT, year INTEGER, conference VARCHAR)"
        )
        connection.commit()
        connection.close()

        set_test_db(db_path)
        script = _load_script()
        db = DatabaseManager()
        try:
            db.create_tables()
            added = script.ensure_generated_keywords_column(db)
            assert added is True
            columns = [row["name"] for row in raw_query(db, "PRAGMA table_info(papers)")]
            assert "generated_keywords" in columns
            assert script.ensure_generated_keywords_column(db) is False
        finally:
            db.close()


class TestBackfillMain:
    def test_dry_run_does_not_write(self, connected_db, tmp_path):
        _seed(connected_db)
        script = _load_script()

        result = script.main(["--dry-run"])

        assert result == 0
        rows = raw_query(connected_db, "SELECT generated_keywords FROM papers")
        assert all(row["generated_keywords"] is None for row in rows)

    def test_backfills_missing_keywords(self, connected_db):
        _seed(connected_db)
        script = _load_script()

        result = script.main([])

        assert result == 0
        rows = raw_query(connected_db, "SELECT generated_keywords FROM papers")
        assert all(row["generated_keywords"] for row in rows)

    def test_respects_conference_and_year_filters(self, connected_db):
        connected_db.add_papers(
            [
                _paper("p1", "Gradient Descent", "Neural network optimization.", year=2025),
                _paper("p2", "Gradient Descent", "Neural network optimization.", year=2024),
            ]
        )
        script = _load_script()

        result = script.main(["--conference", "NeurIPS", "--year", "2025"])

        assert result == 0
        rows = raw_query(connected_db, "SELECT year, generated_keywords FROM papers ORDER BY year")
        assert rows[0]["generated_keywords"] is None
        assert rows[1]["generated_keywords"]

    def test_force_regenerates(self, connected_db):
        _seed(connected_db)
        script = _load_script()
        script.main([])
        before = raw_query(connected_db, "SELECT uid, generated_keywords FROM papers ORDER BY uid")

        result = script.main(["--force"])

        assert result == 0
        after = raw_query(connected_db, "SELECT uid, generated_keywords FROM papers ORDER BY uid")
        assert [row["uid"] for row in after] == [row["uid"] for row in before]
        assert all(row["generated_keywords"] for row in after)

    def test_json_output(self, connected_db, capsys):
        _seed(connected_db)
        script = _load_script()

        result = script.main(["--json"])

        assert result == 0
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["updated"] == 3
        assert payload["dry_run"] is False
        assert len(payload["scopes"]) == 1
        assert payload["scopes"][0]["conference"] == "NeurIPS"
        assert payload["scopes"][0]["year"] == 2025
