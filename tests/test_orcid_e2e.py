"""
Live end-to-end tests for ORCID search-term suggestions.

These tests call the real public ORCID API and the configured chat model.  They
are marked slow and are excluded from the default test run.

Run the live tests with::

    uv run pytest tests/test_orcid_e2e.py -m slow
"""

import json

import pytest

from abstracts_explorer.mcp_server import suggest_search_terms_from_orcid as mcp_suggest_search_terms
from abstracts_explorer.orcid import OrcidError, fetch_orcid_record
from tests.helpers import is_transient_backend_message, requires_lm_studio

ORCID_ID = "0000-0002-2315-9247"

pytestmark = pytest.mark.integration


def _fetch_live_record():
    """Fetch the live ORCID record, skipping when the public API is unavailable."""
    try:
        return fetch_orcid_record(ORCID_ID, timeout=30)
    except OrcidError as exc:
        pytest.skip(f"ORCID public API unavailable: {exc}")


@pytest.mark.slow
def test_live_orcid_record_parsing():
    """Fetch the real ORCID record and verify parsing and text sanitization."""
    record = _fetch_live_record()

    assert record["orcid"] == ORCID_ID
    assert record["name"]
    assert "Korten" in record["name"]
    assert len(record["works"]) >= 5

    titles = [work["title"] for work in record["works"]]
    assert any("biocomputation" in title.lower() for title in titles)
    assert all("<" not in title and ">" not in title for title in titles)
    assert all("&amp;" not in title for title in titles)


@requires_lm_studio
def test_live_orcid_search_terms_with_llm():
    """Generate search terms from the real ORCID record through the MCP tool."""
    record = _fetch_live_record()
    assert record["works"]

    result = json.loads(mcp_suggest_search_terms(ORCID_ID, max_terms=8))

    assert result["orcid"] == ORCID_ID

    warning = result.get("warning") or ""
    if result["source"] != "llm":
        if is_transient_backend_message(warning):
            pytest.skip(f"LLM backend unavailable during test: {warning}")
        pytest.fail(f"LLM term generation unexpectedly fell back: {warning}")

    assert result["warning"] is None

    terms = [item["term"] for item in result["suggested_terms"]]
    assert 1 <= len(terms) <= 8
    assert len({term.casefold() for term in terms}) == len(terms)
    assert all(1 <= len(term.split()) <= 4 for term in terms)
    assert all("<" not in term and ">" not in term for term in terms)

    assert result["recent_titles"]
    assert all("<" not in title and ">" not in title for title in result["recent_titles"])
