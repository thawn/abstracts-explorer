"""
Tests for the SQL profiling utilities.
"""

import json

import pytest
from sqlalchemy import create_engine, text

from abstracts_explorer.profiling import (
    SqlProfiler,
    explain_statement,
    normalize_sql,
    suppress_profiling,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with a small table."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, year INTEGER)"))
        connection.execute(text("INSERT INTO t (name, year) VALUES ('alpha', 2025), ('beta', 2024)"))
    yield engine
    engine.dispose()


class TestNormalizeSql:
    """Tests for normalize_sql()."""

    def test_replaces_string_and_numeric_literals(self):
        """String and numeric literals become parameter placeholders."""
        result = normalize_sql("SELECT * FROM papers WHERE title LIKE '%x%' AND year = 2025")
        assert result == "SELECT * FROM papers WHERE title LIKE ? AND year = ?"

    def test_collapses_in_lists(self):
        """IN-list length does not affect the fingerprint."""
        assert normalize_sql("SELECT * FROM t WHERE id IN (?, ?, ?)") == "SELECT * FROM t WHERE id IN (?)"
        assert normalize_sql("SELECT * FROM t WHERE id IN (?)") == "SELECT * FROM t WHERE id IN (?)"

    def test_replaces_named_and_pyformat_params(self):
        """Named and pyformat bind parameters become placeholders."""
        assert normalize_sql("SELECT * FROM t WHERE a = :param_1") == "SELECT * FROM t WHERE a = ?"
        assert normalize_sql("SELECT * FROM t WHERE a = %(param_1)s") == "SELECT * FROM t WHERE a = ?"

    def test_removes_comments_and_whitespace(self):
        """Comments and repeated whitespace are removed."""
        result = normalize_sql("SELECT  *\nFROM t -- comment\nWHERE a = 1 /* block */")
        assert result == "SELECT * FROM t WHERE a = ?"

    def test_does_not_confuse_cast_syntax(self):
        """PostgreSQL casts are preserved."""
        result = normalize_sql("SELECT a::text FROM t WHERE b = 1")
        assert result == "SELECT a::text FROM t WHERE b = ?"


class TestSqlProfiler:
    """Tests for SqlProfiler recording and aggregation."""

    def test_records_statements_in_a_case(self, engine):
        """Statements executed inside a case are attributed and timed."""
        with SqlProfiler() as profiler:
            with profiler.case("select.one"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM t WHERE name = :name"), {"name": "alpha"}).all()

        cases = profiler.results()
        assert len(cases) == 1
        stats = cases[0]
        assert stats.label == "select.one"
        assert stats.statement_count == 1
        assert stats.sql_ms >= 0
        assert any("SELECT * FROM t WHERE name = ?" == sql for sql in stats.fingerprint_totals)

    def test_nested_cases_attribute_innermost_label(self, engine):
        """Nested cases expose the inner call while outer totals include it."""
        with SqlProfiler() as profiler:
            with profiler.case("outer"):
                with profiler.case("inner"):
                    with engine.connect() as connection:
                        connection.execute(text("SELECT * FROM t")).all()

        aggregated = {case.label: case for case in profiler.aggregated()}
        assert "outer" in aggregated
        assert "outer > inner" in aggregated
        assert aggregated["outer"].statement_count_mean == 1
        assert aggregated["outer > inner"].statement_count_mean == 1

    def test_detach_stops_recording(self, engine):
        """No statements are recorded after the profiler is detached."""
        with SqlProfiler() as profiler:
            with profiler.case("before"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM t")).all()
        recorded = len(profiler.results())

        with engine.connect() as connection:
            connection.execute(text("SELECT * FROM t")).all()

        assert len(profiler.results()) == recorded

    def test_measure_repeats_and_aggregates(self, engine):
        """measure() runs the callable the requested number of times."""

        def query() -> None:
            with engine.connect() as connection:
                connection.execute(text("SELECT * FROM t")).all()

        with SqlProfiler() as profiler:
            aggregated = profiler.measure("measure.query", query, runs=3, warmup=1)

        assert aggregated.label == "measure.query"
        assert aggregated.runs == 3
        assert aggregated.statement_count_mean == 1
        assert aggregated.wall_ms_min <= aggregated.wall_ms_median <= aggregated.wall_ms_p95

    def test_measure_excludes_nested_cases(self, engine):
        """Nested cases do not inflate the measured run count."""
        with SqlProfiler() as profiler:

            def query() -> None:
                with profiler.case("inner"):
                    with engine.connect() as connection:
                        connection.execute(text("SELECT * FROM t")).all()

            aggregated = profiler.measure("outer", query, runs=2)

        assert aggregated.label == "outer"
        assert aggregated.runs == 2
        assert aggregated.statement_count_mean == 1

    def test_suppress_profiling_skips_sql(self, engine):
        """Statements inside suppress_profiling are not recorded."""
        with SqlProfiler() as profiler:
            with profiler.case("visible"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM t")).all()
            with suppress_profiling():
                with profiler.case("hidden"):
                    with engine.connect() as connection:
                        connection.execute(text("SELECT * FROM t")).all()

        cases = {case.label: case for case in profiler.results()}
        assert "visible" in cases
        assert cases["hidden"].statement_count == 0

    def test_to_dict_is_json_serializable(self, engine):
        """The report can be serialized to JSON."""
        with SqlProfiler() as profiler:
            with profiler.case("case"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM t")).all()

        document = json.loads(json.dumps(profiler.to_dict()))
        assert document["aggregated"][0]["label"] == "case"
        assert document["aggregated"][0]["statement_count_mean"] == 1

    def test_attach_and_detach_are_idempotent(self):
        """Attaching and detaching repeatedly does not raise."""
        profiler = SqlProfiler()
        profiler.attach()
        profiler.attach()
        profiler.detach()
        profiler.detach()

    def test_recorded_sql_parameters_when_enabled(self, engine):
        """Raw SQL and parameters are retained when capture_params is enabled."""
        with SqlProfiler(capture_params=True) as profiler:
            with profiler.case("capture"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM t WHERE name = :name"), {"name": "alpha"}).all()

        records = profiler.records()
        assert len(records) == 1
        assert records[0].sql is not None
        assert records[0].parameters is not None
        assert "alpha" in str(records[0].parameters)


class TestExplainStatement:
    """Tests for explain_statement()."""

    def test_rejects_non_postgresql(self, engine):
        """EXPLAIN is only supported for PostgreSQL."""
        with pytest.raises(ValueError):
            explain_statement(engine, "SELECT 1")
