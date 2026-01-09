"""
Shared test helper utilities.

This module contains common utility functions used across multiple test files
to reduce code duplication and ensure consistency.
"""

import socket
import requests
import pytest
from neurips_abstracts.config import get_config
from neurips_abstracts.embeddings import EmbeddingsManager
from neurips_abstracts.database import DatabaseManager

# Cache for LM Studio availability check to avoid multiple API calls during test collection
_lm_studio_available_cache = None


def create_test_db_with_paper(db_path, paper_data):
    """
    Create a test database with a single paper.
    
    Parameters
    ----------
    db_path : Path or str
        Path where the database should be created
    paper_data : dict
        Dictionary containing paper data with keys: uid, title, abstract, authors,
        session, poster_position, keywords, year, conference
        
    Returns
    -------
    DatabaseManager
        Connected database manager instance. Caller is responsible for closing it.
        
    Notes
    -----
    This helper reduces duplication in integration tests that need to manually
    create databases with specific paper data. Default values are provided for
    optional fields if not specified.
    
    Examples
    --------
    >>> db = create_test_db_with_paper(
    ...     tmp_path / "test.db",
    ...     {
    ...         "uid": "1",
    ...         "title": "Test Paper",
    ...         "abstract": "Test abstract",
    ...         "authors": "Test Author",
    ...         "session": "Test Session",
    ...         "keywords": "test",
    ...         "year": 2025,
    ...         "conference": "NeurIPS"
    ...     }
    ... )
    >>> # Use db...
    >>> db.close()
    """
    db = DatabaseManager(str(db_path))
    db.connect()
    db.create_tables()
    
    # Provide defaults for optional fields
    defaults = {
        "poster_position": "",
        "keywords": "",
    }
    paper_data = {**defaults, **paper_data}
    
    # Insert paper into database (lightweight schema)
    cursor = db.connection.cursor()
    cursor.execute(
        """
        INSERT INTO papers (uid, title, abstract, authors, session, poster_position, keywords, year, conference)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_data["uid"],
            paper_data["title"],
            paper_data["abstract"],
            paper_data["authors"],
            paper_data["session"],
            paper_data["poster_position"],
            paper_data["keywords"],
            paper_data["year"],
            paper_data["conference"],
        ),
    )
    db.connection.commit()
    
    return db


def check_lm_studio_available():
    """
    Check if OpenAI-compatible API is available with the configured models.

    Returns
    -------
    bool
        True if the API is available with required models, False otherwise.

    Notes
    -----
    This function checks:
    1. If the API server is running
    2. If any models are loaded
    3. If an embedding generation request works with the configured model

    Used to skip tests that require a running OpenAI-compatible API backend
    (such as LM Studio or blablador).
    
    The result is cached to avoid multiple API calls during test collection.
    """
    global _lm_studio_available_cache

    # Return cached result if available
    if _lm_studio_available_cache is not None:
        return _lm_studio_available_cache

    try:
        config = get_config()
        em = EmbeddingsManager(
            lm_studio_url=config.llm_backend_url,
            auth_token=config.llm_backend_auth_token,
            model_name=config.embedding_model,
        )

        try:
            # Connect and perform a lightweight liveness check via the embeddings API
            if not em.test_lm_studio_connection():
                _lm_studio_available_cache = False
                return False

            _lm_studio_available_cache = True
            return True
        finally:
            em.close()

    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        _lm_studio_available_cache = False
        return False


def find_free_port():
    """
    Find a free port to use for testing.

    Returns
    -------
    int
        Available port number

    Notes
    -----
    Uses a temporary socket to find an available port on the system.
    Useful for starting test servers that need unique ports.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def requires_lm_studio(func):
    """
    Decorator that marks tests as slow and skips them if OpenAI API backend is not available.

    This decorator:
    1. Marks the test as 'slow' (so it's skipped by default with -m "not slow")
    2. Skips the test if the configured OpenAI-compatible API is not running or no model is loaded

    Usage
    -----
    @requires_lm_studio
    def test_something_with_api():
        ...
    """
    # Apply both slow marker and skipif condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not check_lm_studio_available(),
        reason="OpenAI-compatible API not available. Check configuration and ensure the API backend (LM Studio or blablador) is accessible with the configured models.",
    )(func)
    return func
