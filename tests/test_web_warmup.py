"""Tests for the web UI server-start warm-up."""

from unittest.mock import Mock, patch

from abstracts_explorer.web_ui.warmup import warm_up


def test_warm_up_primes_database():
    """Warm-up connects, verifies the schema, and runs metadata queries."""
    mock_config = Mock()
    mock_config.default_conference = "NeurIPS"
    mock_config.default_year = 2025

    mock_db = Mock()

    with (
        patch("abstracts_explorer.config.get_config", return_value=mock_config),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
    ):
        warm_up()

    mock_db.connect.assert_called_once()
    mock_db.create_tables.assert_called_once()
    mock_db.get_conference_years_from_db.assert_called_once()
    mock_db.get_conference_year_locations.assert_called_once()
    mock_db.get_conferences.assert_called_once()
    mock_db.confirm_default_conference_year_exists.assert_called_once_with("NeurIPS", 2025)
    mock_db.close.assert_called_once()


def test_warm_up_handles_missing_defaults():
    """Empty defaults are passed as empty string and None."""
    mock_config = Mock()
    mock_config.default_conference = ""
    mock_config.default_year = 0

    mock_db = Mock()

    with (
        patch("abstracts_explorer.config.get_config", return_value=mock_config),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
    ):
        warm_up()

    mock_db.confirm_default_conference_year_exists.assert_called_once_with("", None)


def test_warm_up_swallows_errors_and_closes():
    """A database failure is logged, not raised, and the connection is closed."""
    mock_db = Mock()
    mock_db.connect.side_effect = RuntimeError("boom")

    with (
        patch("abstracts_explorer.config.get_config", return_value=Mock()),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
    ):
        warm_up()

    mock_db.close.assert_called_once()


def test_warm_up_swallows_import_errors():
    """Import failures during warm-up do not prevent server startup."""
    with patch("abstracts_explorer.database.DatabaseManager", side_effect=ImportError("missing")):
        warm_up()
