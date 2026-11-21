"""
Tests for the ORCID MCP tool.
"""

import json
from unittest.mock import patch

import pytest

from abstracts_explorer.mcp.errors import ToolError
from abstracts_explorer.mcp_server import suggest_search_terms_from_orcid
from abstracts_explorer.orcid import OrcidError


class TestSuggestSearchTermsFromOrcidTool:
    """Tests for the MCP wrapper around the ORCID service."""

    @patch("abstracts_explorer.mcp.tools.orcid._suggest_search_terms_from_orcid")
    def test_returns_json_result(self, mock_suggest):
        mock_suggest.return_value = {
            "orcid": "0000-0002-1825-0097",
            "name": "Jane Doe",
            "works_found": 2,
            "works_used": 2,
            "keywords": ["deep learning"],
            "suggested_terms": [{"term": "deep learning", "reason": "Author keyword", "source": "llm"}],
            "recent_titles": ["Deep Learning for Vision"],
            "source": "llm",
            "warning": None,
        }

        result = json.loads(
            suggest_search_terms_from_orcid(
                "https://orcid.org/0000-0002-1825-0097",
                max_terms=5,
                min_year=2020,
                max_works=20,
            )
        )

        assert result["orcid"] == "0000-0002-1825-0097"
        assert result["suggested_terms"][0]["term"] == "deep learning"
        mock_suggest.assert_called_once_with(
            "https://orcid.org/0000-0002-1825-0097",
            max_terms=5,
            min_year=2020,
            max_works=20,
        )

    @patch("abstracts_explorer.mcp.tools.orcid._suggest_search_terms_from_orcid")
    def test_invalid_orcid_raises_tool_error(self, mock_suggest):
        mock_suggest.side_effect = OrcidError("Invalid ORCID iD")

        with pytest.raises(ToolError, match="Invalid ORCID iD"):
            suggest_search_terms_from_orcid("not-an-orcid")

    @patch("abstracts_explorer.mcp.tools.orcid._suggest_search_terms_from_orcid")
    def test_unexpected_error_raises_tool_error(self, mock_suggest):
        mock_suggest.side_effect = RuntimeError("boom")

        with pytest.raises(ToolError, match="ORCID search-term suggestion failed"):
            suggest_search_terms_from_orcid("0000-0002-1825-0097")
