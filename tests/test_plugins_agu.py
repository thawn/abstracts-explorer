"""Tests for the AGU m-anage downloader plugin."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from abstracts_explorer.plugins import AGUDownloaderPlugin, get_plugin, list_plugin_names


@pytest.fixture
def agu_plugin():
    plugin = AGUDownloaderPlugin()
    plugin.supported_years = [2026]
    plugin._PEOPLE_INDEX_LETTERS = ("A",)
    return plugin


def _response(data):
    response = Mock()
    response.json.return_value = data
    response.raise_for_status = Mock()
    return response


def _api_response(url, params):
    if url.endswith("/ModuleProgramBook/0/ChildList_Program"):
        return _response(
            [
                {
                    "_url": "Program/1",
                    "ChildList_Session": ["Session/1"],
                    "ChildList_SessionReadOnly": ["Session/2"],
                    "ChildList_UnscheduledSession": [],
                }
            ]
        )

    if url.endswith("/Program/1/ChildList_Session"):
        return _response(
            [{"_url": "Session/1", "ChildList_Paper": ["Paper/10"], "SessionTitle": "Atmospheric Science"}]
        )

    if url.endswith("/Program/1/ChildList_SessionReadOnly"):
        return _response(
            [{"_url": "Session/2", "ChildList_Paper": ["Paper/10"], "SessionTitle": "Duplicate session"}]
        )

    paper = {
        "id": "10",
        "Title": "A sample AGU abstract",
        "Abstract": "<p>First sentence.</p><p>Second &amp; final sentence.</p>",
        "ChildList_Person": ["Person/20"],
        "SessionTitle": "Atmospheric Science",
        "FinalNumber": "A1",
    }
    if url.endswith("/Session/1/ChildList_Paper"):
        return _response([paper])
    if url.endswith("/Session/2/ChildList_Paper"):
        return _response([paper])
    if url.endswith("/PeopleIndex/A/ChildList_Person"):
        return _response([{"_url": "Person/20", "FirstName": "Ada", "MiddleName": "Byron", "LastName": "Lovelace"}])

    refs = params["ids"].split(",")
    models = {"Paper/10": paper}
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

    def test_batch_reads_fetch_missing_records_individually(self, agu_plugin):
        refs = ["Session/0", "Session/1", "Session/2", "Session/3"]

        def mock_get(url, params=None, **kwargs):
            if url.endswith("/BatchRead/0"):
                return _response(
                    {"models": {"Session/0": {}, "Session/1": {}}, "unanswered": ["Session/2", "Session/3"]}
                )
            ref = "/".join(url.rstrip("/").split("/")[-2:])
            return _response({"_url": ref})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            records = agu_plugin._fetch_batch_records(
                "https://studio.m-anage.com/agu/fm14/meetingapi.cgi",
                refs,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
            )

        assert sorted(records) == refs
        assert mock_request.call_count == 3
        assert [call.args[0].rsplit("/meetingapi.cgi/", 1)[-1] for call in mock_request.call_args_list] == [
            "BatchRead/0",
            "Session/2",
            "Session/3",
        ]

    def test_batch_reads_skip_refused_records(self, agu_plugin):
        refs = ["Session/0", "Session/1"]

        def mock_get(url, params=None, **kwargs):
            return _response({"models": {"Session/0": {}}, "refused": {"Session/1": 404}})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            records = agu_plugin._fetch_batch_records(
                "https://studio.m-anage.com/agu/fm14/meetingapi.cgi",
                refs,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
            )

        assert sorted(records) == ["Session/0"]
        assert mock_request.call_count == 1

    def test_people_index_falls_back_to_batch_for_missing_people(self, agu_plugin):
        refs = ["Person/1", "Person/2"]

        def mock_get(url, params=None, **kwargs):
            if url.endswith("/PeopleIndex/A/ChildList_Person"):
                return _response([{"_url": "Person/1", "FirstName": "Ada"}])
            if url.endswith("/BatchRead/0"):
                return _response({"models": {"Person/2": {"FirstName": "Grace"}}})
            raise AssertionError(url)

        with patch.object(agu_plugin._session, "get", side_effect=mock_get):
            people = agu_plugin._fetch_people(
                "https://studio.m-anage.com/agu/fm14/meetingapi.cgi",
                refs,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
            )

        assert sorted(people) == refs

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
        assert mock_request.call_count == 6

    def test_fetch_papers_displays_progress(self, agu_plugin):
        sessions = {
            "Session/1": {"_url": "Session/1", "ChildList_Paper": ["Paper/10"]},
            "Session/2": {"_url": "Session/2", "ChildList_Paper": ["Paper/11"]},
        }

        def mock_get(url, params=None, **kwargs):
            if url.endswith("/Session/1/ChildList_Paper"):
                return _response([{"_url": "Paper/10"}])
            if url.endswith("/Session/2/ChildList_Paper"):
                return _response([{"_url": "Paper/11"}])
            raise AssertionError(url)

        with (
            patch.object(agu_plugin._session, "get", side_effect=mock_get),
            patch("abstracts_explorer.plugins.agu_downloader.tqdm") as mock_tqdm,
        ):
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
            papers = agu_plugin._fetch_papers(
                "https://studio.m-anage.com/agu/agu26/meetingapi.cgi",
                sessions,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
                show_progress=True,
            )

        assert sorted(papers) == ["Paper/10", "Paper/11"]
        mock_tqdm.assert_called_once()
        assert mock_tqdm.call_args.kwargs["total"] == 2
        assert mock_tqdm.call_args.kwargs["disable"] is False

    def test_fetch_papers_progress_auto_disables_by_default(self, agu_plugin):
        sessions = {"Session/1": {"_url": "Session/1", "ChildList_Paper": ["Paper/10"]}}

        def mock_get(url, params=None, **kwargs):
            return _response([{"_url": "Paper/10"}])

        with (
            patch.object(agu_plugin._session, "get", side_effect=mock_get),
            patch("abstracts_explorer.plugins.agu_downloader.tqdm") as mock_tqdm,
        ):
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
            agu_plugin._fetch_papers(
                "https://studio.m-anage.com/agu/agu26/meetingapi.cgi",
                sessions,
                timeout=30,
                verify_ssl=True,
                request_delay=0,
            )

        assert mock_tqdm.call_args.kwargs["disable"] is None

    def test_download_passes_progress_flag_to_fetch(self, agu_plugin):
        def mock_get(url, params=None, **kwargs):
            return _api_response(url, params or {})

        with (
            patch.object(agu_plugin._session, "get", side_effect=mock_get),
            patch("abstracts_explorer.plugins.agu_downloader.tqdm") as mock_tqdm,
        ):
            mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
            agu_plugin.download(year=2026, request_delay=0, progress=False)

        assert mock_tqdm.call_args.kwargs["disable"] is True

    @pytest.mark.parametrize(("year", "slug"), [(2022, "fm22"), (2024, "agu24")])
    def test_download_uses_year_specific_api_url(self, agu_plugin, year, slug):
        agu_plugin.supported_years = list(range(2014, 2027))

        def mock_get(url, params=None, **kwargs):
            return _api_response(url, params or {})

        with patch.object(agu_plugin._session, "get", side_effect=mock_get) as mock_request:
            agu_plugin.download(year=year, request_delay=0)

        assert mock_request.call_count == 6
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


class TestAGURetryHandling:
    @staticmethod
    def _http_error(status, headers=None):
        response = Mock(status_code=status, headers=headers or {})
        return requests.HTTPError(response=response)

    def test_session_uses_browser_user_agent(self, agu_plugin):
        assert "Mozilla" in agu_plugin._session.headers["User-Agent"]

    def test_fetch_retries_429_and_then_succeeds(self, agu_plugin):
        rate_limited = _response({})
        rate_limited.raise_for_status.side_effect = self._http_error(429, {"Retry-After": "7"})
        success = _response({"ok": True})

        with (
            patch.object(agu_plugin._session, "get", side_effect=[rate_limited, success]) as mock_get,
            patch("abstracts_explorer.plugin.time.sleep") as mock_sleep,
        ):
            result = agu_plugin._fetch_json("https://example.com/api", 30, True)

        assert result == {"ok": True}
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1
        assert 0 < mock_sleep.call_args.args[0] <= 7

    def test_fetch_raises_after_max_attempts(self, agu_plugin):
        agu_plugin._MAX_FETCH_ATTEMPTS = 3
        rate_limited = _response({})
        rate_limited.raise_for_status.side_effect = self._http_error(429)

        with (
            patch.object(agu_plugin._session, "get", return_value=rate_limited) as mock_get,
            patch("abstracts_explorer.plugin.time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="429"):
                agu_plugin._fetch_json("https://example.com/api", 30, True)

        assert mock_get.call_count == 3
