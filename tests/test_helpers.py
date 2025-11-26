"""
Shared test helper utilities.

This module contains common utility functions used across multiple test files
to reduce code duplication and ensure consistency.
"""

import socket
import requests
import pytest
from neurips_abstracts.config import get_config


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
        url = config.llm_backend_url
        model = config.chat_model

        # Check if server is running
        response = requests.get(f"{url}/v1/models", timeout=2)
        if response.status_code != 200:
            return False

        # Check if there are any models loaded
        data = response.json()
        if not data.get("data"):
            return False

        # Try a simple chat completion with the configured model
        test_response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
            timeout=5,
        )
        return test_response.status_code == 200

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


# Skip marker for tests requiring LM Studio - can be imported directly
requires_lm_studio = pytest.mark.skipif(
    not check_lm_studio_available(),
    reason="LM Studio not running or no chat model loaded. Check configuration and ensure LM Studio is started with the configured chat model.",
)
