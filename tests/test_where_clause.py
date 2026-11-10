"""Tests for the shared ChromaDB where-clause builder."""

from typing import Any, cast

import pytest

from abstracts_explorer.embeddings import build_chromadb_where_clause


class TestBuildChromaDBWhereClause:
    """Tests for build_chromadb_where_clause."""

    def test_no_filters(self):
        assert build_chromadb_where_clause() is None

    def test_empty_filters(self):
        assert build_chromadb_where_clause(conferences=[], years=[], sessions=[]) is None

    def test_only_conference(self):
        result = build_chromadb_where_clause(conferences=["NeurIPS"])
        assert result == {"conference": {"$in": ["NeurIPS"]}}

    def test_only_years(self):
        result = build_chromadb_where_clause(years=[2024, 2025])
        assert result == {"year": {"$in": ["2024", "2025"]}}

    def test_only_sessions(self):
        result = build_chromadb_where_clause(sessions=["Oral", "Spotlight"])
        assert result == {"session": {"$in": ["Oral", "Spotlight"]}}

    def test_conference_and_years(self):
        result = build_chromadb_where_clause(conferences=["NeurIPS"], years=[2024])
        assert result == {
            "$and": [
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024"]}},
            ]
        }

    def test_all_filters(self):
        result = build_chromadb_where_clause(
            conferences=["NeurIPS"],
            years=[2024, 2025],
            sessions=["Oral"],
        )
        assert result == {
            "$and": [
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024", "2025"]}},
                {"session": {"$in": ["Oral"]}},
            ]
        }

    def test_tuple_inputs(self):
        result = build_chromadb_where_clause(conferences=("NeurIPS",), years=(2024,))
        assert result == {
            "$and": [
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024"]}},
            ]
        }

    def test_base_where_only_is_deep_copied(self):
        base = {"$or": [{"session": "Oral"}, {"award": {"$ne": None}}]}
        result = build_chromadb_where_clause(base_where=base)
        assert result == base
        assert result is not None
        assert result is not base
        assert result["$or"] is not base["$or"]

    def test_empty_base_where(self):
        result = build_chromadb_where_clause(conferences=["NeurIPS"], base_where={})
        assert result == {"conference": {"$in": ["NeurIPS"]}}

    def test_base_where_merged_with_filters(self):
        result = build_chromadb_where_clause(
            conferences=["NeurIPS"],
            years=[2024],
            base_where={"session": "Oral"},
        )
        assert result == {
            "$and": [
                {"session": "Oral"},
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024"]}},
            ]
        }

    def test_base_where_with_existing_and_is_flattened(self):
        base = {"$and": [{"year": {"$gte": 2024}}, {"session": "Oral"}]}
        result = build_chromadb_where_clause(conferences=["NeurIPS"], base_where=base)
        assert result == {
            "$and": [
                {"year": {"$gte": 2024}},
                {"session": "Oral"},
                {"conference": {"$in": ["NeurIPS"]}},
            ]
        }

    def test_existing_conference_is_not_duplicated(self):
        base = {"conference": "ICML"}
        result = build_chromadb_where_clause(conferences=["NeurIPS"], years=[2024], base_where=base)
        assert result == {
            "$and": [
                {"conference": "ICML"},
                {"year": {"$in": ["2024"]}},
            ]
        }

    def test_existing_conference_in_nested_and_is_not_duplicated(self):
        base = {"$and": [{"conference": "ICML"}, {"year": 2024}]}
        result = build_chromadb_where_clause(conferences=["NeurIPS"], base_where=base)
        assert result == {"$and": [{"conference": "ICML"}, {"year": 2024}]}

    def test_existing_conference_in_deeply_nested_or_is_not_duplicated(self):
        base = {"$or": [{"$and": [{"conference": "ICML"}, {"year": 2024}]}, {"session": "Oral"}]}
        result = build_chromadb_where_clause(conferences=["NeurIPS"], base_where=base)
        assert result == base

    def test_existing_year_is_not_duplicated(self):
        base = {"year": {"$gte": 2024}}
        result = build_chromadb_where_clause(years=[2024], conferences=["NeurIPS"], base_where=base)
        assert result == {
            "$and": [
                {"year": {"$gte": 2024}},
                {"conference": {"$in": ["NeurIPS"]}},
            ]
        }

    def test_existing_session_is_not_duplicated(self):
        base = {"session": {"$in": ["Oral"]}}
        result = build_chromadb_where_clause(sessions=["Spotlight"], years=[2024], base_where=base)
        assert result == {
            "$and": [
                {"session": {"$in": ["Oral"]}},
                {"year": {"$in": ["2024"]}},
            ]
        }

    def test_all_filters_already_present(self):
        base = {
            "$and": [
                {"conference": "ICML"},
                {"year": {"$in": ["2024"]}},
                {"session": "Oral"},
            ]
        }
        result = build_chromadb_where_clause(
            conferences=["NeurIPS"],
            years=[2024],
            sessions=["Spotlight"],
            base_where=base,
        )
        assert result == base

    def test_does_not_mutate_base_or_input_lists(self):
        base = {"$and": [{"session": "Oral"}]}
        conferences = ["NeurIPS"]
        years = [2024]

        result = build_chromadb_where_clause(
            conferences=conferences,
            years=years,
            base_where=base,
        )

        assert result is not None
        assert base == {"$and": [{"session": "Oral"}]}
        assert conferences == ["NeurIPS"]
        assert years == [2024]
        assert len(result["$and"]) == 3

        result["$and"].append({"modified": True})
        assert len(base["$and"]) == 1

    def test_invalid_base_where_type(self):
        with pytest.raises(ValueError, match="must be a dict"):
            build_chromadb_where_clause(base_where=cast(Any, "invalid"))

    def test_invalid_base_where_list(self):
        with pytest.raises(ValueError, match="must be a dict"):
            build_chromadb_where_clause(base_where=cast(Any, [{"year": 2024}]))

    def test_invalid_and_value(self):
        with pytest.raises(ValueError, match=r"'\$and' value must be a list"):
            build_chromadb_where_clause(
                conferences=["NeurIPS"],
                base_where={"$and": {"year": 2024}},
            )

    def test_combine_with_or(self):
        result = build_chromadb_where_clause(
            conferences=["NeurIPS"],
            years=[2024],
            sessions=["Oral"],
            combine_with="$or",
        )
        assert result == {
            "$or": [
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024"]}},
                {"session": {"$in": ["Oral"]}},
            ]
        }

    def test_combine_with_or_merges_existing_or(self):
        base = {"$or": [{"session": "Oral"}, {"session": "Spotlight"}]}
        result = build_chromadb_where_clause(
            conferences=["NeurIPS"],
            base_where=base,
            combine_with="$or",
        )
        assert result == {
            "$or": [
                {"session": "Oral"},
                {"session": "Spotlight"},
                {"conference": {"$in": ["NeurIPS"]}},
            ]
        }

    def test_invalid_combine_with(self):
        with pytest.raises(ValueError, match="combine_with must be"):
            build_chromadb_where_clause(conferences=["NeurIPS"], combine_with="$xor")

    def test_invalid_existing_or_value(self):
        with pytest.raises(ValueError, match=r"'\$or' value must be a list"):
            build_chromadb_where_clause(
                conferences=["NeurIPS"],
                base_where={"$or": {"session": "Oral"}},
                combine_with="$or",
            )

    def test_scalar_inputs_are_wrapped(self):
        result = build_chromadb_where_clause(
            conferences="NeurIPS",
            years=2024,
            sessions="Oral",
        )
        assert result == {
            "$and": [
                {"conference": {"$in": ["NeurIPS"]}},
                {"year": {"$in": ["2024"]}},
                {"session": {"$in": ["Oral"]}},
            ]
        }
