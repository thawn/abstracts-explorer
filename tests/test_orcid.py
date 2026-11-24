"""
Tests for the ORCID service module.
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from abstracts_explorer import orcid
from abstracts_explorer.orcid import (
    OrcidError,
    fetch_orcid_record,
    normalize_orcid_id,
    suggest_search_terms_from_orcid,
)

WORKS_PAYLOAD = {
    "group": [
        {
            "work-summary": [
                {
                    "title": {"title": {"value": "Deep Learning for Vision"}},
                    "publication-date": {"year": {"value": "2024"}},
                    "type": "conference-paper",
                    "external-ids": {
                        "external-id": [
                            {
                                "external-id-type": "doi",
                                "external-id-value": "10.1234/vision",
                            }
                        ]
                    },
                },
                {
                    "title": {"title": {"value": "Graph Neural Networks"}},
                    "publication-date": {"year": {"value": "2022"}},
                    "type": "journal-article",
                },
            ]
        },
        {
            "work-summary": [
                {
                    "title": {"title": {"value": "Deep Learning for Vision"}},
                    "publication-date": {"year": {"value": "2024"}},
                    "type": "conference-paper",
                },
                {
                    "title": {"title": {"value": "Reinforcement Learning at Scale"}},
                    "publication-date": {"year": {"value": "2019"}},
                    "type": "conference-paper",
                },
            ]
        },
    ]
}

KEYWORDS_PAYLOAD = {
    "keyword": [
        {"content": "deep learning"},
        {"content": "reinforcement learning"},
    ]
}

PERSON_PAYLOAD = {
    "name": {
        "given-names": {"value": "Jane"},
        "family-name": {"value": "Doe"},
        "credit-name": None,
    }
}


def _response(payload, status_code=200):
    """Build a fake requests response."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _mock_orcid_get(*, works=WORKS_PAYLOAD, keywords=KEYWORDS_PAYLOAD, person=PERSON_PAYLOAD):
    """Return a requests.get side effect that serves ORCID payloads."""

    def _get(url, **kwargs):
        if url.endswith("/works"):
            return _response(works)
        if url.endswith("/keywords"):
            return _response(keywords)
        if url.endswith("/person"):
            return _response(person)
        raise AssertionError(f"Unexpected URL: {url}")

    return _get


@pytest.fixture(autouse=True)
def _clear_orcid_cache():
    """Clear the module-level ORCID cache between tests."""
    orcid._ORCID_CACHE.clear()
    yield
    orcid._ORCID_CACHE.clear()


class TestNormaliseOrcidId:
    """Tests for ORCID iD normalisation."""

    @pytest.mark.parametrize(
        "value",
        [
            "0000-0002-2315-9247",
            "https://orcid.org/0000-0002-2315-9247",
            "http://orcid.org/0000-0002-2315-9247/",
            "orcid.org/0000-0002-2315-9247",
        ],
    )
    def test_accepts_valid_forms(self, value):
        assert normalize_orcid_id(value) == "0000-0002-2315-9247"

    def test_uppercases_check_digit(self):
        assert normalize_orcid_id("0000-0002-2315-924x") == "0000-0002-2315-924X"

    @pytest.mark.parametrize("value", ["", "not-an-orcid", "0000-0002-1825", None, 123])
    def test_rejects_invalid_forms(self, value):
        with pytest.raises(OrcidError):
            normalize_orcid_id(value)


class TestFetchOrcidRecord:
    """Tests for fetching and parsing public ORCID records."""

    @patch("abstracts_explorer.orcid.requests.get")
    def test_parses_works_keywords_and_name(self, mock_get):
        mock_get.side_effect = _mock_orcid_get()

        record = fetch_orcid_record("https://orcid.org/0000-0002-2315-9247")

        assert record["orcid"] == "0000-0002-2315-9247"
        assert record["name"] == "Jane Doe"
        assert record["keywords"] == ["deep learning", "reinforcement learning"]
        assert len(record["works"]) == 3
        assert record["works"][0]["title"] == "Deep Learning for Vision"
        assert record["works"][0]["year"] == 2024
        assert record["works"][0]["doi"] == "10.1234/vision"
        assert record["works"][-1]["year"] == 2019
        assert mock_get.call_count == 3
        assert mock_get.call_args.kwargs["headers"]["Accept"] == "application/json"

    @patch("abstracts_explorer.orcid.requests.get")
    def test_uses_cache_for_repeated_fetches(self, mock_get):
        mock_get.side_effect = _mock_orcid_get()

        fetch_orcid_record("0000-0002-2315-9247")
        fetch_orcid_record("https://orcid.org/0000-0002-2315-9247")

        assert mock_get.call_count == 3

    @patch("abstracts_explorer.orcid.requests.get")
    def test_raises_not_found_for_404(self, mock_get):
        mock_get.return_value = _response({}, status_code=404)

        with pytest.raises(OrcidError, match="not found"):
            fetch_orcid_record("0000-0002-2315-9247")

    @patch("abstracts_explorer.orcid.requests.get")
    def test_raises_on_server_error(self, mock_get):
        mock_get.return_value = _response({}, status_code=503)

        with pytest.raises(OrcidError, match="503"):
            fetch_orcid_record("0000-0002-2315-9247")

    @patch("abstracts_explorer.orcid.requests.get")
    def test_raises_on_network_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("boom")

        with pytest.raises(OrcidError, match="Could not reach"):
            fetch_orcid_record("0000-0002-2315-9247")

    @patch("abstracts_explorer.orcid.requests.get")
    def test_raises_on_malformed_json(self, mock_get):
        response = _response({})
        response.json.side_effect = ValueError("bad json")
        mock_get.return_value = response

        with pytest.raises(OrcidError, match="malformed"):
            fetch_orcid_record("0000-0002-2315-9247")

    @patch("abstracts_explorer.orcid.requests.get")
    def test_handles_empty_record(self, mock_get):
        mock_get.side_effect = _mock_orcid_get(works={}, keywords={}, person={})

        record = fetch_orcid_record("0000-0002-2315-9247")

        assert record["works"] == []
        assert record["keywords"] == []
        assert record["name"] is None

    @patch("abstracts_explorer.orcid.requests.get")
    def test_strips_html_and_entities_from_record_text(self, mock_get):
        works = {
            "group": [
                {
                    "work-summary": [
                        {
                            "title": {
                                "title": {
                                    "value": "<b>Data Management</b> Makes &amp; Machine Learning Easier",
                                }
                            },
                            "publication-date": {"year": {"value": "2024"}},
                            "journal-title": "<i>Journal</i> &amp; Letters",
                        }
                    ]
                }
            ]
        }
        keywords = {"keyword": [{"content": "<i>bioinformatics</i>"}]}
        person = {
            "name": {
                "given-names": {"value": "<b>Till</b>"},
                "family-name": {"value": "Korten &amp; Co"},
            }
        }
        mock_get.side_effect = _mock_orcid_get(works=works, keywords=keywords, person=person)

        record = fetch_orcid_record("0000-0002-2315-9247")

        assert record["works"][0]["title"] == "Data Management Makes & Machine Learning Easier"
        assert record["works"][0]["journal"] == "Journal & Letters"
        assert record["keywords"] == ["bioinformatics"]
        assert record["name"] == "Till Korten & Co"


class TestSuggestSearchTerms:
    """Tests for LLM-generated ORCID search terms."""

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_returns_llm_terms(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": ["deep learning"],
            "works": [
                {"title": "Deep Learning for Vision", "year": 2024, "type": "paper", "doi": None, "journal": None}
            ],
        }
        mock_chat.return_value = json.dumps(
            {
                "search_terms": [
                    {"term": "deep learning", "reason": "Author keyword"},
                    {"term": "computer vision", "reason": "Work topic"},
                ]
            }
        )

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", client=Mock())

        assert result["orcid"] == "0000-0002-2315-9247"
        assert result["source"] == "llm"
        assert result["warning"] is None
        assert [item["term"] for item in result["suggested_terms"]] == [
            "deep learning",
            "computer vision",
        ]
        assert result["suggested_terms"][0]["source"] == "llm"
        assert result["recent_titles"] == ["Deep Learning for Vision"]
        assert mock_chat.call_count == 1

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_parses_json_inside_code_fences(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": [],
            "works": [
                {"title": "Deep Learning for Vision", "year": 2024, "type": "paper", "doi": None, "journal": None}
            ],
        }
        mock_chat.return_value = 'Here you go:\n```json\n{"search_terms": ["deep learning"]}\n```'

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", client=Mock())

        assert result["suggested_terms"][0]["term"] == "deep learning"

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_falls_back_to_keywords_and_titles(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": ["deep learning"],
            "works": [
                {"title": "Deep Learning for Vision", "year": 2024, "type": "paper", "doi": None, "journal": None},
                {"title": "Deep Learning at Scale", "year": 2023, "type": "paper", "doi": None, "journal": None},
            ],
        }
        mock_chat.side_effect = RuntimeError("backend down")

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", client=Mock())

        assert result["source"] == "fallback"
        assert result["warning"] and "backend down" in result["warning"]
        terms = [item["term"] for item in result["suggested_terms"]]
        assert "deep learning" in terms
        assert all(item["source"] == "fallback" for item in result["suggested_terms"])

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_falls_back_on_invalid_llm_output(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": ["deep learning"],
            "works": [
                {"title": "Deep Learning for Vision", "year": 2024, "type": "paper", "doi": None, "journal": None}
            ],
        }
        mock_chat.return_value = "no json here"

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", client=Mock())

        assert result["source"] == "fallback"
        assert result["suggested_terms"]

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_filters_works_by_min_year(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": [],
            "works": [
                {"title": "Old Work", "year": 2010, "type": "paper", "doi": None, "journal": None},
                {"title": "New Work", "year": 2024, "type": "paper", "doi": None, "journal": None},
            ],
        }
        mock_chat.return_value = json.dumps({"search_terms": ["new work"]})

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", min_year=2020, client=Mock())

        user_prompt = mock_chat.call_args.kwargs["user_prompt"]
        assert "New Work" in user_prompt
        assert "Old Work" not in user_prompt
        assert result["works_used"] == 1

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_limits_terms(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": [],
            "works": [
                {"title": "Deep Learning for Vision", "year": 2024, "type": "paper", "doi": None, "journal": None}
            ],
        }
        mock_chat.return_value = json.dumps({"search_terms": [f"term {i}" for i in range(10)]})

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", max_terms=3, client=Mock())

        assert [item["term"] for item in result["suggested_terms"]] == ["term 0", "term 1", "term 2"]

    @patch("abstracts_explorer.orcid.chat_completion_text")
    @patch("abstracts_explorer.orcid.fetch_orcid_record")
    def test_returns_empty_for_empty_record(self, mock_fetch, mock_chat):
        mock_fetch.return_value = {
            "orcid": "0000-0002-2315-9247",
            "name": "Jane Doe",
            "keywords": [],
            "works": [],
        }

        result = suggest_search_terms_from_orcid("0000-0002-2315-9247", client=Mock())

        assert result["suggested_terms"] == []
        assert result["source"] == "none"
        mock_chat.assert_not_called()


def test_provider_client_uses_central_retry_and_cooldown_transport():
    """The ORCID LLM client uses the same retry budget and transports as other clients."""
    from abstracts_explorer.config import get_config
    from abstracts_explorer.rate_limiter import (
        CooldownAwareTransport,
        RateLimitedTransport,
        get_global_cooldown,
    )

    config = get_config()
    provider = orcid._OrcidLlmProvider()
    client = provider.openai_client
    try:
        assert client.max_retries == config.llm_max_attempts - 1
        transport = client._client._transport
        assert isinstance(transport, CooldownAwareTransport)
        assert transport._cooldown is get_global_cooldown()
        if config.requests_per_minute > 0:
            assert isinstance(transport._transport, RateLimitedTransport)
    finally:
        client.close()
