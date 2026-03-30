"""
Tests for CHI Downloader Plugin
=================================

Test suite for the CHI (ACM Conference on Human Factors in Computing Systems)
conference data downloader plugin.
"""

import copy
import json
import pytest

from abstracts_explorer.plugins.chi_downloader import CHIDownloaderPlugin
from abstracts_explorer.plugin import LightweightPaper
from abstracts_explorer.database import DatabaseManager
from tests.conftest import set_test_db

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CHI_JSON_TEMPLATE = {
    "conferenceAddons": {},
    "conference": {
        "id": 10107,
        "shortName": "CHI",
        "displayShortName": "",
        "year": 2024,
        "startDate": 1715385600000,
        "endDate": 1715817600000,
        "fullName": "ACM CHI Conference on Human Factors in Computing Systems",
        "url": "https://chi2024.acm.org/",
        "location": "Honolulu, USA",
        "timeZoneOffset": -600,
        "timeZoneName": "Pacific/Honolulu",
        "name": "CHI 2024",
    },
    "publicationInfo": {
        "publicationStatus": "PUBLISHED",
        "isProgramEnabled": True,
        "isDraft": False,
        "version": 111,
    },
    "contentTypes": [
        {"id": 13269, "name": "Paper", "color": "#aabbcc", "duration": 20},
        {"id": 13268, "name": "Late-Breaking Work", "color": "#ddeeff", "duration": 40},
        {"id": 13273, "name": "Break", "color": "#999999", "duration": 15},
    ],
    "sessions": [
        {
            "id": 156056,
            "name": "Social Activism B",
            "isParallelPresentation": False,
            "startDate": 1715598000000,
            "endDate": 1715602800000,
            "roomId": 11470,
            "sessionType": "SESSION",
            "contentIds": [146610, 146611],
            "typeId": 13269,
        },
        {
            "id": 156057,
            "name": "Poster Session 1",
            "isParallelPresentation": True,
            "startDate": 1715620000000,
            "endDate": 1715624000000,
            "roomId": 11471,
            "sessionType": "SESSION",
            "contentIds": [146612],
            "typeId": 13268,
        },
    ],
    "people": [
        {"id": 144808, "firstName": "Alice", "lastName": "Smith", "middleInitial": "", "institutions": []},
        {"id": 143163, "firstName": "Bob", "lastName": "Jones", "middleInitial": "", "institutions": []},
        {"id": 145767, "firstName": "Carol", "lastName": "Williams", "middleInitial": "", "institutions": []},
    ],
    "recognitions": [
        {"id": 10084, "name": "Artefact Available", "iconName": "achievement-big-badge"},
    ],
    "contents": [
        {
            "id": 146610,
            "typeId": 13269,
            "title": "Test Paper One",
            "abstract": "Abstract for test paper one about human-computer interaction.",
            "award": None,
            "recognitionIds": [],
            "isBreak": False,
            "sessionIds": [156056],
            "authors": [
                {"personId": 144808, "institutions": ["MIT"]},
                {"personId": 143163, "institutions": ["Stanford"]},
            ],
            "addons": {
                "doi": {
                    "hideBeforeConference": True,
                    "type": "doiLink",
                    "url": "doi.org/10.1145/3613904.3642337",
                }
            },
        },
        {
            "id": 146611,
            "typeId": 13269,
            "title": "Test Best Paper",
            "abstract": "Abstract for the best paper on accessibility research.",
            "award": "BEST_PAPER",
            "recognitionIds": [],
            "isBreak": False,
            "sessionIds": [156056],
            "authors": [
                {"personId": 145767, "institutions": ["CMU"]},
            ],
            "addons": {
                "doi": {
                    "hideBeforeConference": True,
                    "type": "doiLink",
                    "url": "doi.org/10.1145/3613904.9999999",
                }
            },
        },
        {
            "id": 146612,
            "typeId": 13268,
            "title": "Late-Breaking Work Demo",
            "abstract": "Abstract for a late-breaking work submission.",
            "award": "HONORABLE_MENTION",
            "recognitionIds": [],
            "isBreak": False,
            "sessionIds": [156057],
            "authors": [
                {"personId": 144808, "institutions": ["MIT"]},
            ],
            "addons": {},
        },
        {
            "id": 146613,
            "typeId": 13273,
            "title": "Break",
            "abstract": "",  # No abstract – should be skipped
            "award": None,
            "recognitionIds": [],
            "isBreak": True,
            "sessionIds": [],
            "authors": [],
            "addons": {},
        },
    ],
    "sponsors": [],
    "sponsorLevels": [],
    "floors": [],
    "rooms": [],
    "team": {"contacts": [], "teamMembers": []},
}


@pytest.fixture
def chi_plugin():
    """Return a fresh CHIDownloaderPlugin instance."""
    return CHIDownloaderPlugin()


@pytest.fixture
def chi_json_file(tmp_path):
    """Write a minimal CHI program JSON to a temp file and return its path."""
    path = tmp_path / "chi_2024_program.json"
    path.write_text(json.dumps(CHI_JSON_TEMPLATE), encoding="utf-8")
    return str(path)


@pytest.fixture
def chi_json_2023_file(tmp_path):
    """CHI 2023 variant of the JSON (year field differs)."""
    data = copy.deepcopy(CHI_JSON_TEMPLATE)  # deep copy
    data["conference"]["year"] = 2023
    path = tmp_path / "chi_2023_program.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def chi_json_2025_file(tmp_path):
    """CHI 2025 variant of the JSON."""
    data = copy.deepcopy(CHI_JSON_TEMPLATE)
    data["conference"]["year"] = 2025
    path = tmp_path / "chi_2025_program.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Plugin properties
# ---------------------------------------------------------------------------


class TestCHIPluginProperties:
    """Tests for CHI plugin metadata and properties."""

    def test_plugin_name(self, chi_plugin):
        """Plugin name must be 'chi'."""
        assert chi_plugin.plugin_name == "chi"

    def test_plugin_description(self, chi_plugin):
        """Plugin must have a non-empty description."""
        assert chi_plugin.plugin_description
        assert "CHI" in chi_plugin.plugin_description

    def test_supported_years(self, chi_plugin):
        """Plugin must support 2023, 2024, and 2025."""
        assert 2023 in chi_plugin.supported_years
        assert 2024 in chi_plugin.supported_years
        assert 2025 in chi_plugin.supported_years

    def test_conference_name(self, chi_plugin):
        """Conference name must be CHI."""
        assert chi_plugin.conference_name == "CHI"

    def test_get_metadata(self, chi_plugin):
        """get_metadata() must return the expected structure."""
        meta = chi_plugin.get_metadata()
        assert meta["name"] == "chi"
        assert meta["conference_name"] == "CHI"
        assert 2024 in meta["supported_years"]
        assert "input_path" in meta["parameters"]
        assert "output_path" in meta["parameters"]
        assert "year" in meta["parameters"]

    def test_inherits_lightweight_downloader(self, chi_plugin):
        """CHIDownloaderPlugin must be a LightweightDownloaderPlugin."""
        from abstracts_explorer.plugin import LightweightDownloaderPlugin

        assert isinstance(chi_plugin, LightweightDownloaderPlugin)

    def test_award_names_mapping(self, chi_plugin):
        """AWARD_NAMES must contain the expected keys."""
        assert "BEST_PAPER" in chi_plugin.AWARD_NAMES
        assert "HONORABLE_MENTION" in chi_plugin.AWARD_NAMES
        assert chi_plugin.AWARD_NAMES["BEST_PAPER"] == "Best Paper Award"
        assert chi_plugin.AWARD_NAMES["HONORABLE_MENTION"] == "Honorable Mention"


# ---------------------------------------------------------------------------
# Year validation
# ---------------------------------------------------------------------------


class TestCHIPluginValidation:
    """Tests for year validation."""

    def test_validate_supported_years(self, chi_plugin):
        """validate_year should not raise for supported years."""
        for year in [2023, 2024, 2025]:
            chi_plugin.validate_year(year)  # must not raise

    def test_validate_unsupported_year(self, chi_plugin):
        """validate_year must raise ValueError for unsupported year."""
        with pytest.raises(ValueError, match="1999"):
            chi_plugin.validate_year(1999)

    def test_validate_year_none_is_ok(self, chi_plugin):
        """validate_year(None) must not raise."""
        chi_plugin.validate_year(None)  # must not raise

    def test_download_raises_for_unsupported_year(self, chi_plugin, chi_json_file):
        """download() raises ValueError when year is not in supported_years."""
        with pytest.raises(ValueError):
            chi_plugin.download(year=1999, input_path=chi_json_file)


# ---------------------------------------------------------------------------
# Successful download / parsing
# ---------------------------------------------------------------------------


class TestCHIPluginDownload:
    """Tests for the core download / parse workflow."""

    def test_download_returns_lightweight_papers(self, chi_plugin, chi_json_file):
        """download() returns a list of LightweightPaper objects."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        assert isinstance(papers, list)
        assert all(isinstance(p, LightweightPaper) for p in papers)

    def test_download_skips_items_without_abstract(self, chi_plugin, chi_json_file):
        """Items with empty/missing abstract must be excluded."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        for p in papers:
            assert p.abstract.strip()

    def test_download_excludes_breaks(self, chi_plugin, chi_json_file):
        """Break items (isBreak=True) with no abstract must not appear."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        titles = [p.title for p in papers]
        assert "Break" not in titles

    def test_download_paper_fields(self, chi_plugin, chi_json_file):
        """Verify field mapping for the first test paper."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Test Paper One")

        assert paper.year == 2024
        assert paper.conference == "CHI"
        assert paper.abstract.startswith("Abstract for test paper one")
        assert "Alice Smith" in paper.authors
        assert "Bob Jones" in paper.authors
        assert paper.session == "Social Activism B"
        assert paper.poster_position == "146610"

    def test_download_doi_url_prefixed(self, chi_plugin, chi_json_file):
        """doi.org URLs must be expanded to https://doi.org/…"""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Test Paper One")
        assert paper.url is not None
        assert paper.url.startswith("https://")
        assert "10.1145" in paper.url

    def test_download_best_paper_award(self, chi_plugin, chi_json_file):
        """BEST_PAPER award must be mapped to the human-readable string."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Test Best Paper")
        assert paper.award == "Best Paper Award"

    def test_download_honorable_mention_award(self, chi_plugin, chi_json_file):
        """HONORABLE_MENTION award must be mapped correctly."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Late-Breaking Work Demo")
        assert paper.award == "Honorable Mention"

    def test_download_no_award(self, chi_plugin, chi_json_file):
        """Papers without an award must have award=None."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Test Paper One")
        assert paper.award is None

    def test_download_content_type_as_keyword(self, chi_plugin, chi_json_file):
        """Content-type name should appear in keywords."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Test Paper One")
        assert paper.keywords is not None
        assert "Paper" in paper.keywords

    def test_download_lbw_session(self, chi_plugin, chi_json_file):
        """Late-Breaking Work items must resolve to the correct session."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        paper = next(p for p in papers if p.title == "Late-Breaking Work Demo")
        assert paper.session == "Poster Session 1"

    def test_download_count(self, chi_plugin, chi_json_file):
        """Exactly 3 content items have non-empty abstracts in the fixture."""
        papers = chi_plugin.download(year=2024, input_path=chi_json_file)
        assert len(papers) == 3

    def test_download_2023(self, chi_plugin, chi_json_2023_file):
        """Plugin must parse CHI 2023 JSON and set year=2023."""
        papers = chi_plugin.download(year=2023, input_path=chi_json_2023_file)
        assert all(p.year == 2023 for p in papers)

    def test_download_2025(self, chi_plugin, chi_json_2025_file):
        """Plugin must parse CHI 2025 JSON and set year=2025."""
        papers = chi_plugin.download(year=2025, input_path=chi_json_2025_file)
        assert all(p.year == 2025 for p in papers)

    def test_download_year_inferred_from_json(self, chi_plugin, chi_json_file):
        """When year=None, the year must be read from the JSON conference field."""
        papers = chi_plugin.download(year=None, input_path=chi_json_file)
        assert all(p.year == 2024 for p in papers)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCHIPluginErrors:
    """Tests for error-handling behaviour."""

    @pytest.mark.skipif(
        CHIDownloaderPlugin()._get_default_input_path(2024).exists(),
        reason="Auto-detect file exists, cannot test missing input_path error.",
    )
    def test_download_raises_without_input_path(self, chi_plugin):
        """ValueError must be raised when no input_path is given and no auto-detect file exists."""
        with pytest.raises(ValueError, match="programs.sigchi.org"):
            chi_plugin.download(year=2024)

    def test_download_raises_file_not_found(self, chi_plugin):
        """FileNotFoundError must be raised when input_path does not exist."""
        with pytest.raises(FileNotFoundError, match="nonexistent.json"):
            chi_plugin.download(year=2024, input_path="/nonexistent/path/nonexistent.json")

    def test_download_raises_on_year_mismatch(self, chi_plugin, chi_json_file):
        """ValueError when requested year differs from year in JSON."""
        with pytest.raises(ValueError, match="2025"):
            # chi_json_file contains year=2024; requesting year=2025 should fail
            chi_plugin.download(year=2025, input_path=chi_json_file)

    def test_download_raises_on_no_papers(self, chi_plugin, tmp_path):
        """RuntimeError when the JSON has no content items with abstracts."""
        empty_data = {
            "conference": {"year": 2024, "shortName": "CHI"},
            "contentTypes": [],
            "sessions": [],
            "people": [],
            "recognitions": [],
            "contents": [],
        }
        json_file = tmp_path / "empty.json"
        json_file.write_text(json.dumps(empty_data), encoding="utf-8")

        with pytest.raises(RuntimeError, match="No papers"):
            chi_plugin.download(year=2024, input_path=str(json_file))

    def test_download_invalid_json_file(self, chi_plugin, tmp_path):
        """Raises an exception when the JSON file is malformed."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")

        with pytest.raises(Exception):
            chi_plugin.download(year=2024, input_path=str(bad_file))


# ---------------------------------------------------------------------------
# Caching (output_path)
# ---------------------------------------------------------------------------


class TestCHIPluginCaching:
    """Tests for the output_path caching behaviour."""

    def test_save_to_output_path(self, chi_plugin, chi_json_file, tmp_path):
        """Parsed papers must be saved as a JSON list when output_path is given."""
        out = tmp_path / "chi_2024.json"
        chi_plugin.download(year=2024, input_path=chi_json_file, output_path=str(out))

        assert out.exists()
        with open(out, encoding="utf-8") as fh:
            saved = json.load(fh)
        assert isinstance(saved, list)
        assert len(saved) == 3
        assert saved[0]["conference"] == "CHI"

    def test_load_from_existing_output_path(self, chi_plugin, chi_json_file, tmp_path):
        """When output_path exists, the plugin loads from it (no re-parsing)."""
        out = tmp_path / "chi_2024.json"
        # First run: parse and save
        chi_plugin.download(year=2024, input_path=chi_json_file, output_path=str(out))

        # Second run: should load from cache – remove input_path to prove no re-parsing
        papers = chi_plugin.download(year=2024, output_path=str(out))
        assert isinstance(papers, list)
        assert len(papers) == 3

    def test_force_download_skips_cache(self, chi_plugin, chi_json_file, tmp_path):
        """With force_download=True, cached output_path is ignored."""
        out = tmp_path / "chi_2024.json"
        # Create a stale cache with wrong data
        stale_data = [
            {
                "title": "Old Cached Paper",
                "abstract": "Old",
                "authors": ["X"],
                "session": "S",
                "poster_position": "1",
                "year": 2024,
                "conference": "CHI",
            }
        ]
        out.write_text(json.dumps(stale_data), encoding="utf-8")

        # Force re-parse
        papers = chi_plugin.download(year=2024, input_path=chi_json_file, output_path=str(out), force_download=True)
        titles = [p.title for p in papers]
        assert "Old Cached Paper" not in titles
        assert "Test Paper One" in titles


# ---------------------------------------------------------------------------
# Auto-detection of data/CHI_{year}_program.json
# ---------------------------------------------------------------------------


class TestCHIPluginAutoDetect:
    """Tests for automatic detection of CHI JSON files in the data/ directory."""

    def test_auto_detect_chi_json(self, chi_plugin, tmp_path, monkeypatch):
        """download() auto-detects data/CHI_{year}_program.json when no input_path is given."""
        # Create the expected auto-detect path
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        chi_json = data_dir / "CHI_2024_program.json"
        chi_json.write_text(json.dumps(CHI_JSON_TEMPLATE), encoding="utf-8")

        # Run from tmp_path so data/ is found
        monkeypatch.chdir(tmp_path)

        papers = chi_plugin.download(year=2024)
        assert isinstance(papers, list)
        assert len(papers) == 3
        assert all(p.year == 2024 for p in papers)
        assert all(p.conference == "CHI" for p in papers)
        titles = {p.title for p in papers}
        assert "Test Paper One" in titles
        assert "Test Best Paper" in titles

    def test_auto_detect_falls_back_to_error(self, chi_plugin, tmp_path, monkeypatch):
        """download() raises ValueError when auto-detect file does not exist."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="CHI_2024_program.json"):
            chi_plugin.download(year=2024)


# ---------------------------------------------------------------------------
# DOI URL normalisation
# ---------------------------------------------------------------------------


class TestCHIDoiUrlNormalisation:
    """Tests for various DOI URL format inputs."""

    def _paper_with_doi(self, doi_url: str) -> LightweightPaper:
        """Return a LightweightPaper for a content item whose DOI addon has doi_url."""
        plugin = CHIDownloaderPlugin()
        item = {
            "id": 1,
            "typeId": 13269,
            "title": "T",
            "abstract": "A",
            "award": None,
            "sessionIds": [],
            "authors": [{"personId": 1}],
            "addons": {"doi": {"url": doi_url, "type": "doiLink"}},
        }
        people = {1: {"firstName": "A", "lastName": "B"}}
        result = plugin._convert_to_lightweight(item, 2024, people, {}, {13269: "Paper"})
        assert result is not None
        return result

    def test_doi_org_prefix(self):
        """doi.org/… must become https://doi.org/…"""
        paper = self._paper_with_doi("doi.org/10.1145/1234567.1234568")
        assert paper.url == "https://doi.org/10.1145/1234567.1234568"

    def test_http_doi_org(self):
        """http://doi.org/… must become https://doi.org/…"""
        paper = self._paper_with_doi("http://doi.org/10.1145/1234567.1234568")
        assert paper.url == "https://doi.org/10.1145/1234567.1234568"

    def test_https_doi_org(self):
        """https://doi.org/… must remain unchanged."""
        paper = self._paper_with_doi("https://doi.org/10.1145/1234567.1234568")
        assert paper.url == "https://doi.org/10.1145/1234567.1234568"

    def test_bare_doi(self):
        """A bare DOI (e.g. 10.1145/…) must be prefixed with https://doi.org/"""
        paper = self._paper_with_doi("10.1145/1234567.1234568")
        assert paper.url == "https://doi.org/10.1145/1234567.1234568"

    def test_empty_doi(self):
        """An empty DOI addon must result in url=None."""
        plugin = CHIDownloaderPlugin()
        item = {
            "id": 2,
            "typeId": 13269,
            "title": "T2",
            "abstract": "A2",
            "award": None,
            "sessionIds": [],
            "authors": [{"personId": 1}],
            "addons": {},
        }
        paper = plugin._convert_to_lightweight(
            item, 2024, {1: {"firstName": "A", "lastName": "B"}}, {}, {13269: "Paper"}
        )
        assert paper is not None
        assert paper.url is None


# ---------------------------------------------------------------------------
# Session fallback
# ---------------------------------------------------------------------------


class TestCHISessionFallback:
    """Tests for session-name resolution edge cases."""

    def test_fallback_to_content_type_when_no_session(self, chi_plugin, tmp_path):
        """When sessionIds is empty, the content-type name is used as session."""
        data = copy.deepcopy(CHI_JSON_TEMPLATE)
        # Clear sessionIds for the first content item
        data["contents"][0]["sessionIds"] = []
        json_file = tmp_path / "chi_nosession.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        papers = chi_plugin.download(year=2024, input_path=str(json_file))
        paper = next(p for p in papers if p.title == "Test Paper One")
        assert paper.session == "Paper"

    def test_fallback_to_chi_when_unknown_type(self, chi_plugin, tmp_path):
        """When typeId has no mapping, session falls back to 'CHI'."""
        data = copy.deepcopy(CHI_JSON_TEMPLATE)
        data["contents"][0]["sessionIds"] = []
        data["contents"][0]["typeId"] = 99999  # unknown type
        json_file = tmp_path / "chi_unknowntype.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        papers = chi_plugin.download(year=2024, input_path=str(json_file))
        paper = next(p for p in papers if p.title == "Test Paper One")
        assert paper.session == "CHI"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestCHIPluginRegistration:
    """Tests for plugin auto-registration."""

    def test_plugin_auto_registers(self):
        """CHI plugin must be available via get_plugin('chi') after import."""
        from abstracts_explorer.plugins import get_plugin

        plugin = get_plugin("chi")
        assert plugin is not None
        assert isinstance(plugin, CHIDownloaderPlugin)

    def test_plugin_in_plugin_list(self):
        """'chi' must appear in list_plugin_names()."""
        from abstracts_explorer.plugins import list_plugin_names

        assert "chi" in list_plugin_names()


# ---------------------------------------------------------------------------
# Database integration
# ---------------------------------------------------------------------------


class TestCHIPluginDatabaseIntegration:
    """Tests for end-to-end DB insertion."""

    def test_chi_papers_inserted_into_database(self, chi_plugin, chi_json_file, tmp_path):
        """Parsed CHI papers must be insertable into a DatabaseManager."""
        db_path = tmp_path / "chi_test.db"
        set_test_db(db_path)

        papers = chi_plugin.download(year=2024, input_path=chi_json_file)

        with DatabaseManager() as db:
            db.create_tables()
            count = db.add_papers(papers)
            assert count == len(papers)

            rows = db.query("SELECT title, year, conference, authors FROM papers")
            titles = [r["title"] for r in rows]
            assert "Test Paper One" in titles
            assert "Test Best Paper" in titles

            # Verify year and conference
            for row in rows:
                assert row["year"] == 2024
                assert row["conference"] == "CHI"

    def test_chi_authors_stored_in_database(self, chi_plugin, chi_json_file, tmp_path):
        """Author names must be persisted in the database."""
        db_path = tmp_path / "chi_authors.db"
        set_test_db(db_path)

        papers = chi_plugin.download(year=2024, input_path=chi_json_file)

        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(papers)

            rows = db.query("SELECT authors FROM papers WHERE title = 'Test Paper One'")
            assert rows
            authors_str = rows[0]["authors"]
            assert "Alice Smith" in authors_str
            assert "Bob Jones" in authors_str

    def test_chi_award_stored_in_database(self, chi_plugin, chi_json_file, tmp_path):
        """Award field must be persisted in the database."""
        db_path = tmp_path / "chi_award.db"
        set_test_db(db_path)

        papers = chi_plugin.download(year=2024, input_path=chi_json_file)

        with DatabaseManager() as db:
            db.create_tables()
            db.add_papers(papers)

            rows = db.query("SELECT award FROM papers WHERE title = 'Test Best Paper'")
            assert rows
            assert rows[0]["award"] == "Best Paper Award"
