"""Tests for the web UI server-start warm-up."""

from unittest.mock import Mock, patch

from abstracts_explorer.web_ui.warmup import warm_up


def test_warm_up_primes_database():
    """Warm-up connects, verifies the schema, and runs metadata queries."""
    mock_config = Mock()
    mock_config.default_conference = "NeurIPS"
    mock_config.default_year = 2025
    mock_config.warmup_embeddings = False

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
    mock_config.warmup_embeddings = False

    mock_db = Mock()

    with (
        patch("abstracts_explorer.config.get_config", return_value=mock_config),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
    ):
        warm_up()

    mock_db.confirm_default_conference_year_exists.assert_called_once_with("", None)


def test_warm_up_swallows_errors_and_closes():
    """A database failure is logged, not raised, and the connection is closed."""
    mock_config = Mock()
    mock_config.warmup_embeddings = False
    mock_db = Mock()
    mock_db.connect.side_effect = RuntimeError("boom")

    with (
        patch("abstracts_explorer.config.get_config", return_value=mock_config),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
    ):
        warm_up()

    mock_db.close.assert_called_once()


def test_warm_up_swallows_import_errors():
    """Import failures during warm-up do not prevent server startup."""
    mock_config = Mock()
    mock_config.warmup_embeddings = False
    with (
        patch("abstracts_explorer.config.get_config", return_value=mock_config),
        patch("abstracts_explorer.database.DatabaseManager", side_effect=ImportError("missing")),
    ):
        warm_up()


def _embedding_config():
    config = Mock()
    config.default_conference = ""
    config.default_year = 0
    config.warmup_embeddings = True
    config.collection_name = "papers"
    return config


def test_warm_up_loads_embeddings_collection():
    """Warm-up counts and queries the configured Chroma collection."""
    mock_db = Mock()
    mock_em = Mock()
    collection = Mock()
    collection.count.return_value = 3
    collection.peek.return_value = {"embeddings": [[0.0, 0.0, 0.0, 0.0]]}
    mock_em.collection = collection

    with (
        patch("abstracts_explorer.config.get_config", return_value=_embedding_config()),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
        patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
    ):
        warm_up()

    collection.count.assert_called_once()
    collection.peek.assert_called_once()
    collection.query.assert_called_once()
    assert collection.query.call_args.kwargs["query_embeddings"] == [[0.0, 0.0, 0.0, 0.0]]
    mock_em.close.assert_called_once()


def test_warm_up_skips_empty_embeddings_collection():
    """An empty collection is counted but not queried."""
    mock_db = Mock()
    mock_em = Mock()
    collection = Mock()
    collection.count.return_value = 0
    mock_em.collection = collection

    with (
        patch("abstracts_explorer.config.get_config", return_value=_embedding_config()),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
        patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
    ):
        warm_up()

    collection.query.assert_not_called()
    mock_em.close.assert_called_once()


def test_warm_up_swallows_embedding_errors():
    """Embedding warm-up failures do not prevent server startup."""
    mock_db = Mock()
    mock_em = Mock()
    collection = Mock()
    collection.count.side_effect = RuntimeError("chroma down")
    mock_em.collection = collection

    with (
        patch("abstracts_explorer.config.get_config", return_value=_embedding_config()),
        patch("abstracts_explorer.database.DatabaseManager", return_value=mock_db),
        patch("abstracts_explorer.embeddings.EmbeddingsManager", return_value=mock_em),
    ):
        warm_up()

    mock_em.close.assert_called_once()


def test_warm_up_respects_embedding_opt_out():
    """WARMUP_EMBEDDINGS=0 skips the Chroma warm-up."""
    config = _embedding_config()
    config.warmup_embeddings = False

    with (
        patch("abstracts_explorer.config.get_config", return_value=config),
        patch("abstracts_explorer.database.DatabaseManager", return_value=Mock()),
        patch("abstracts_explorer.embeddings.EmbeddingsManager") as mock_em_class,
    ):
        warm_up()

    mock_em_class.assert_not_called()
