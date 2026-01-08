"""
Shared test helper utilities.

This module contains common utility functions used across multiple test files
to reduce code duplication and ensure consistency.
"""
from warnings import warn

import socket
import requests
import pytest
from neurips_abstracts.config import get_config
from neurips_abstracts.embeddings import EmbeddingsManager


def check_lm_studio_available():
    """
    Check if LM Studio is running and available with the configured chat model.

    Returns
    -------
    bool
        True if LM Studio is available with a chat model, False otherwise.

    Notes
    -----
    This function checks:
    1. If the LM Studio server is running
    2. If any models are loaded
    3. If a chat completion request works with the configured model

    Used to skip tests that require a running LM Studio instance.
    """
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
                return False

            # Try generating a tiny embedding to ensure the embedding endpoint works
            try:
                embedding = em.generate_embedding("test")
                if not isinstance(embedding, list) or len(embedding) == 0:
                    return False
            except Exception as e:
                warn("Embedding generation failed", UserWarning)
                return False

            return True
        finally:
            em.close()

    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
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
    Decorator that marks tests as slow and skips them if LM Studio is not available.

    This decorator:
    1. Marks the test as 'slow' (so it's skipped by default with -m "not slow")
    2. Skips the test if LM Studio is not running or no chat model is loaded

    Usage
    -----
    @requires_lm_studio
    def test_something_with_lm_studio():
        ...
    """
    # Apply both slow marker and skipif condition
    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(
        not check_lm_studio_available(),
        reason="LM Studio not running or no chat model loaded. Check configuration and ensure LM Studio is started with the configured chat model.",
    )(func)
    return func
