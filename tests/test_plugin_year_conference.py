"""
Tests for year and conference fields in plugins.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from neurips_abstracts.plugins.neurips_downloader import NeurIPSDownloaderPlugin
from neurips_abstracts.plugins.icml_downloader import ICMLDownloaderPlugin
from neurips_abstracts.plugins.ml4ps_downloader import ML4PSDownloaderPlugin
from neurips_abstracts.database import DatabaseManager


class TestNeurIPSPluginYearConference:
    """Test that NeurIPS plugin sets year and conference fields."""

    @patch("neurips_abstracts.plugins.json_conference_downloader.requests.get")
    def test_neurips_plugin_adds_year_and_conference(self, mock_get):
        """Test that NeurIPS plugin adds year and conference to each paper."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        papers = plugin.download(year=2024)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 2

        # Verify year and conference were set
        for paper in papers:
            assert paper.year == 2024
            assert paper.conference == "NeurIPS"

    @patch("neurips_abstracts.plugins.json_conference_downloader.requests.get")
    def test_neurips_plugin_preserves_existing_fields(self, mock_get):
        """Test that NeurIPS plugin preserves existing paper fields."""
        # Mock the requests.get to return test data with existing fields
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "Test Abstract",
                    "authors": [{"fullname": "John Doe"}],
                    "keywords": ["ML", "AI"],
                    "session": "Session A",
                    "poster_position": "A1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 1

        # Verify existing fields are preserved in LightweightPaper
        paper = papers[0]
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test Abstract"
        assert "John Doe" in paper.authors
        assert paper.keywords == ["ML", "AI"]

        # Verify new fields were added
        assert paper.year == 2025
        assert paper.conference == "NeurIPS"


class TestICMLPluginYearConference:
    """Test that ICML plugin sets year and conference fields."""

    @patch("neurips_abstracts.plugins.json_conference_downloader.requests.get")
    def test_icml_plugin_adds_year_and_conference(self, mock_get):
        """Test that ICML plugin adds year and conference to each paper."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = ICMLDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 2

        # Verify year and conference were set
        for paper in papers:
            assert paper.year == 2025
            assert paper.conference == "ICML"

    @patch("neurips_abstracts.plugins.json_conference_downloader.requests.get")
    def test_icml_plugin_preserves_existing_fields(self, mock_get):
        """Test that ICML plugin preserves existing paper fields."""
        # Mock the requests.get to return test data with existing fields
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper",
                    "abstract": "Test Abstract",
                    "authors": [{"fullname": "John Doe"}],
                    "keywords": ["ML", "AI"],
                    "session": "Session A",
                    "poster_position": "A1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = ICMLDownloaderPlugin()
        papers = plugin.download(year=2025)

        # Verify we got a list of LightweightPaper objects
        assert isinstance(papers, list)
        assert len(papers) == 1

        # Verify existing fields are preserved in LightweightPaper
        paper = papers[0]
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test Abstract"
        assert "John Doe" in paper.authors
        assert paper.keywords == ["ML", "AI"]

        # Verify new fields were added
        assert paper.year == 2025
        assert paper.conference == "ICML"


class TestML4PSPluginYearConference:
    """Test that ML4PS plugin sets year and conference fields."""

    def test_ml4ps_lightweight_format_includes_year_and_conference(self):
        """Test that ML4PS plugin includes year and conference in lightweight format."""
        plugin = ML4PSDownloaderPlugin()

        # Create sample papers data
        papers = [
            {
                "id": 1,
                "title": "Test Paper",
                "authors_str": "John Doe, Jane Smith",
                "abstract": "Test abstract",
                "eventtype": "Poster",
                "awards": [],
            }
        ]

        # Convert to lightweight format
        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        # Verify year and conference are set
        assert len(lightweight_papers) == 1
        paper = lightweight_papers[0]
        assert paper["year"] == 2025
        assert paper["conference"] == "ML4PS@Neurips"

    def test_ml4ps_lightweight_format_preserves_fields(self):
        """Test that ML4PS plugin preserves all required fields."""
        plugin = ML4PSDownloaderPlugin()

        papers = [
            {
                "id": 42,
                "title": "Amazing Paper",
                "authors_str": "Alice, Bob, Charlie",
                "abstract": "This is an amazing abstract",
                "eventtype": "Spotlight",
                "awards": ["Best Paper", "Outstanding Poster"],
                "paper_url": "https://example.com/paper.pdf",
                "openreview_url": "https://openreview.net/paper",
            }
        ]

        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        paper = lightweight_papers[0]
        assert paper["title"] == "Amazing Paper"
        assert paper["authors"] == ["Alice", "Bob", "Charlie"]
        assert paper["abstract"] == "This is an amazing abstract"
        assert paper["session"] == "ML4PhysicalSciences 2025 Workshop - Spotlight"
        assert paper["id"] == 42
        assert paper["year"] == 2025
        assert paper["conference"] == "ML4PS@Neurips"
        assert paper["award"] == "Best Paper, Outstanding Poster"


class TestDatabaseYearConferenceIntegration:
    """Test that year and conference fields are properly stored in the database."""

    @patch("neurips_abstracts.plugins.json_conference_downloader.requests.get")
    def test_neurips_year_conference_in_database(self, mock_get):
        """Test that year and conference are stored in database from NeurIPS plugin."""
        # Mock the requests.get to return test data
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": 1,
                    "name": "Test Paper 1",
                    "abstract": "Abstract 1",
                    "authors": [{"fullname": "Author 1"}],
                    "session": "Session A",
                    "eventtype": "Poster",
                    "poster_position": "A1",
                },
                {
                    "id": 2,
                    "name": "Test Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [{"fullname": "Author 2"}],
                    "session": "Session B",
                    "eventtype": "Oral",
                    "poster_position": "B1",
                },
            ],
        }
        mock_get.return_value = mock_response

        plugin = NeurIPSDownloaderPlugin()
        data = plugin.download(year=2024)

        # Create temporary database and load data (data is now List[LightweightPaper])
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with DatabaseManager(db_path) as db:
                db.create_tables()
                db.add_papers(data)

                # Query papers and verify year and conference
                papers = db.query("SELECT uid, title, year, conference FROM papers ORDER BY title")
                assert len(papers) == 2

                # Check first paper (Test Paper 1)
                assert papers[0]["title"] == "Test Paper 1"
                assert papers[0]["year"] == 2024
                assert papers[0]["conference"] == "NeurIPS"

                # Check second paper (Test Paper 2)
                assert papers[1]["title"] == "Test Paper 2"
                assert papers[1]["year"] == 2024
                assert papers[1]["conference"] == "NeurIPS"

                # Test filtering by year
                papers_2024 = db.query("SELECT * FROM papers WHERE year = 2024")
                assert len(papers_2024) == 2

                # Test filtering by conference
                neurips_papers = db.query("SELECT * FROM papers WHERE conference = 'NeurIPS'")
                assert len(neurips_papers) == 2

    def test_ml4ps_year_conference_in_database(self):
        """Test that year and conference are stored in database from ML4PS plugin."""
        plugin = ML4PSDownloaderPlugin()

        # Create sample lightweight papers
        papers = [
            {
                "id": 1,
                "title": "ML4PS Paper 1",
                "authors_str": "Alice, Bob",
                "abstract": "Abstract for paper 1",
                "eventtype": "Poster",
                "awards": [],
            },
            {
                "id": 2,
                "title": "ML4PS Paper 2",
                "authors_str": "Charlie",
                "abstract": "Abstract for paper 2",
                "eventtype": "Spotlight",
                "awards": ["Best Paper"],
            },
        ]

        # Convert to lightweight format (which adds year and conference)
        lightweight_papers = plugin._convert_to_lightweight_format(papers)

        # Convert to LightweightPaper objects for insertion
        from neurips_abstracts.plugin import LightweightPaper

        papers_to_insert = [LightweightPaper(**paper_dict) for paper_dict in lightweight_papers]

        # Create temporary database and load data
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with DatabaseManager(db_path) as db:
                db.create_tables()
                db.add_papers(papers_to_insert)

                # Query papers and verify year and conference
                papers = db.query("SELECT uid, title, year, conference FROM papers ORDER BY uid")
                assert len(papers) == 2

                # Check first paper
                assert papers[0]["title"] == "ML4PS Paper 1"
                assert papers[0]["year"] == 2025
                assert papers[0]["conference"] == "ML4PS@Neurips"

                # Check second paper
                assert papers[1]["title"] == "ML4PS Paper 2"
                assert papers[1]["year"] == 2025
                assert papers[1]["conference"] == "ML4PS@Neurips"

                # Test filtering by year
                papers_2025 = db.query("SELECT * FROM papers WHERE year = 2025")
                assert len(papers_2025) == 2

                # Test filtering by conference
                ml4ps_papers = db.query("SELECT * FROM papers WHERE conference = 'ML4PS@Neurips'")
                assert len(ml4ps_papers) == 2
