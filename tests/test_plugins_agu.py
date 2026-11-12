"""Tests for the AGU m-anage downloader plugin."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from abstracts_explorer.plugins import AGUDownloaderPlugin, get_plugin, list_plugin_names


@pytest.fixture
def agu_plugin():
    plugin = AGUDownloaderPlugin()
    plugin.supported_years = [2026]
    return plugin


def _response(data):
    response = Mock()
    response.json.return_value = data
    response.raise_for_status = Mock()
    return response


def _api_response(url, params):
    if url.endswith("/ModuleProgramBook/0/ChildList_Program"):
        return _response(
            {
                "0": {
                    "ChildList_Session": ["Session/1"],
                    "ChildList_SessionReadOnly": ["Session/2"],
                    "ChildList_UnscheduledSession": [],
                }
            }
        )

    refs = params["ids"].split(",")
    models = {
        "Session/1": {"ChildList_PaperSlot": ["SlotData/Paper10_Slot1"]},
        "Session/2": {"ChildList_PaperSlot": ["SlotData/Paper10_Slot2"]},
        "SlotData/Paper10_Slot1": {
            "ChildList_Entry": ["Paper/10"],
            "SessionTitle": "Atmospheric Science",
            "Entry_FinalNumber": "A1",
        },
        "SlotData/Paper10_Slot2": {
            "ChildList_Entry": ["Paper/10"],
            "SessionTitle": "Duplicate session",
        },
        "Paper/10": {
            "id": "10",
            "Title": "A sample AGU abstract",
            "Abstract": "<p>First sentence.</p><p>Second &amp; final sentence.</p>",
            "ChildList_Person": ["Person/20"],
            "SessionTitle": "Atmospheric Science",
            "FinalNumber": "A1",
        },
        "Person/20": {"FirstName": "Ada", "MiddleName": "Byron", "LastName": "Lovelace"},
    }
    return _response({"models": {ref: models[ref] for ref in refs if ref in models}})


class TestAGUPluginProperties:
    def test_metadata_and_url(self, agu_plugin):
        assert agu_plugin.plugin_name == "agu"
        assert agu_plugin.conference_name == "AGU"
        assert agu_plugin.get_url(2026) == "https://studio.m-anage.com/agu/agu26/meetingapp.cgi"
        assert agu_plugin.get_metadata()["name"] == "agu"

    @pytest.mark.parametrize(
        ("year", "slug"),
        [
            (2014, "fm14"),
            (2020, "fm20"),
            (2021, "fm21"),
            (2023, "fm23"),
            (2024, "agu24"),
            (2025, "agu25"),
            (2026, "agu26"),
        ],
    )
    def test_get_url_uses_year_specific_slug(self, agu_plugin, year, slug):
        assert agu_plugin.get_url(year) == f"https://studio.m-anage.com/agu/{slug}/meetingapp.cgi"

    def test_supported_years_start_at_2014(self):
        plugin = AGUDownloaderPlugin()
        with patch.object(plugin, "_check_current_year_available", return_value=False):
            years = plugin.supported_years

        assert min(years) == 2014
        assert 2013 not in years

    def test_registered(self):
        assert isinstance(get_plugin("agu"), AGUDownloaderPlugin)
        assert "agu" in list_plugin_names()


class TestAGUDownload:
    def test_batch_reads_respect_api_limit(self, agu_plugin):
        refs = [f"Session/{index}" for index in range(21)]

        def mock_get(url, params=None, **kwargs):
            batch_refs = (params or {})["ids"].split(",")
            return _response({"models": {ref: {"_url": ref} for ref in batch_refs}})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            records = agu_plugin._fetch_batch_records(
                "https://studio.m-anage.com/agu/agu26/meetingapi.cgi",
                refs,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
            )

        assert len(records) == 21
        assert [len(call.kwargs["params"]["ids"].split(",")) for call in mock_request.call_args_list] == [20, 1]

    def test_downloads_and_deduplicates_abstracts(self, agu_plugin):
        def mock_get(url, params=None, **kwargs):
            return _api_response(url, params or {})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            papers = agu_plugin.download(year=2026, request_delay=0)

        assert len(papers) == 1
        assert papers[0].title == "A sample AGU abstract"
        assert papers[0].abstract == "First sentence. Second & final sentence."
        assert papers[0].authors == ["Ada Byron Lovelace"]
        assert papers[0].session == "Atmospheric Science"
        assert papers[0].poster_position == "A1"
        assert papers[0].original_id == 10
        assert papers[0].conference == "AGU"
        assert mock_request.call_count == 5

    @pytest.mark.parametrize(("year", "slug"), [(2022, "fm22"), (2024, "agu24")])
    def test_download_uses_year_specific_api_url(self, agu_plugin, year, slug):
        agu_plugin.supported_years = list(range(2014, 2027))

        def mock_get(url, params=None, **kwargs):
            return _api_response(url, params or {})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            agu_plugin.download(year=year, request_delay=0)

        assert mock_request.call_count == 5
        for call in mock_request.call_args_list:
            assert f"/agu/{slug}/meetingapi.cgi" in call.args[0]

    def test_download_saves_and_reuses_cache(self, agu_plugin, tmp_path):
        output_path = Path(tmp_path) / "agu.json"

        def mock_get(url, params=None, **kwargs):
            return _api_response(url, params or {})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get):
            agu_plugin.download(year=2026, output_path=str(output_path), request_delay=0)

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved[0]["conference"] == "AGU"

        with patch.object(agu_plugin._session, "get") as mock_request:
            papers = agu_plugin.download(year=2026, output_path=str(output_path))

        mock_request.assert_not_called()
        assert papers[0].title == "A sample AGU abstract"

    def test_unsupported_year(self, agu_plugin):
        with pytest.raises(ValueError, match="not supported"):
            agu_plugin.download(year=2013)

    def test_empty_api_results_raise(self, agu_plugin):
        with patch.object(
            agu_plugin._session,
            "get",
            return_value=_response({"0": {"ChildList_Session": []}}),
        ):
            with pytest.raises(RuntimeError, match="No abstracts"):
                agu_plugin.download(year=2026)
