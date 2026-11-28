"""
Cached-fixture end-to-end tests for ORCID search-term suggestions.

These tests exercise the public ORCID service and MCP tool using permanent
recorded fixtures.  They never contact the live ORCID API or the configured
chat model.  The live staging check lives in ``tests/test_staging_e2e.py``.
"""

import json

import pytest

from abstracts_explorer import orcid
from abstracts_explorer.mcp_server import suggest_search_terms_from_orcid as mcp_suggest_search_terms
from abstracts_explorer.orcid import fetch_orcid_record, suggest_search_terms_from_orcid
from tests.helpers import (
    ORCID_FIXTURE_ID,
    build_orcid_fixture_record,
    install_orcid_fixture_cache,
    load_orcid_fixture,
    patch_orcid_get_json,
)

ORCID_ID = ORCID_FIXTURE_ID

pytestmark = pytest.mark.integration


def test_fixture_orcid_record_parsing(monkeypatch):
    """The recorded ORCID payloads parse into a sanitized public record."""
    orcid._ORCID_CACHE.clear()
    patch_orcid_get_json(monkeypatch)

    record = fetch_orcid_record(ORCID_ID)

    assert record["orcid"] == ORCID_ID
    assert record["name"]
    assert "Korten" in record["name"]
    assert len(record["works"]) >= 5
    assert record["keywords"] == []

    titles = [work["title"] for work in record["works"]]
    assert any("biocomputation" in title.lower() for title in titles)
    assert all("<" not in title and ">" not in title for title in titles)
    assert all("&amp;" not in title for title in titles)


def test_fixture_orcid_search_terms_via_mcp(orcid_fixture_cache):
    """The MCP tool returns the recorded terms without calling live services."""
    recorded_terms = load_orcid_fixture("suggested_terms")["search_terms"]
    result = json.loads(mcp_suggest_search_terms(ORCID_ID))

    assert result["orcid"] == ORCID_ID
    assert result["source"] == "llm"
    assert result["warning"] is None
    assert result["keywords"] == []

    terms = [item["term"] for item in result["suggested_terms"]]
    assert terms == [item["term"] for item in recorded_terms]
    assert all(item["source"] == "llm" for item in result["suggested_terms"])
    assert len({term.casefold() for term in terms}) == len(terms)
    assert all(1 <= len(term.split()) <= 4 for term in terms)
    assert all("<" not in term and ">" not in term for term in terms)

    assert result["recent_titles"]
    assert all("<" not in title and ">" not in title for title in result["recent_titles"])


def test_fixture_orcid_search_terms_respects_max_terms():
    """The cached recommendation honours the requested term limit."""
    _, expected_terms = install_orcid_fixture_cache(max_terms=3)

    result = suggest_search_terms_from_orcid(ORCID_ID, max_terms=3)

    assert [item["term"] for item in result["suggested_terms"]] == [item["term"] for item in expected_terms]
    assert len(result["suggested_terms"]) == 3


def test_fixture_orcid_falls_back_when_llm_fails(monkeypatch):
    """A failing LLM still yields terms derived from the recorded works."""
    record = build_orcid_fixture_record()
    orcid._ORCID_CACHE.set(f"orcid:{ORCID_ID}", record, 86400)

    monkeypatch.setattr(orcid, "chat_completion_text", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    result = suggest_search_terms_from_orcid(ORCID_ID)

    assert result["source"] == "fallback"
    assert result["suggested_terms"]
    assert any("biocomputation" in item["term"].lower() for item in result["suggested_terms"])
