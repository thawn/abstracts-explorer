"""Tests for shared test helper utilities."""

import httpx2
import pytest
from openai import APIConnectionError, APITimeoutError

from abstracts_explorer.embeddings import EmbeddingsError
from tests.helpers import (
    is_transient_backend_error,
    is_transient_backend_message,
    skip_on_backend_unavailable,
)


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str = "server error") -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_is_transient_backend_error_status_code(status_code):
    assert is_transient_backend_error(_StatusError(status_code))


def test_is_transient_backend_error_wrapped_embeddings_error():
    try:
        try:
            raise _StatusError(502)
        except Exception as cause:
            raise EmbeddingsError("Failed to generate embedding") from cause
    except EmbeddingsError as exc:
        assert is_transient_backend_error(exc)


def test_is_transient_backend_error_timeout_and_connection():
    request = httpx2.Request("POST", "http://test/v1/embeddings")
    assert is_transient_backend_error(APITimeoutError(request=request))
    assert is_transient_backend_error(APIConnectionError(request=request))


def test_is_transient_backend_error_ignores_non_transient():
    assert not is_transient_backend_error(AssertionError("bad result 502"))
    assert not is_transient_backend_error(ValueError("invalid input"))
    assert not is_transient_backend_error(_StatusError(400))
    assert not is_transient_backend_error(_StatusError(404))


@pytest.mark.parametrize(
    "message",
    [
        "LLM term generation failed (Error code: 502); used keywords instead.",
        "Request timed out.",
        "HTTP/1.1 503 Service Unavailable",
        "Connection reset by peer",
    ],
)
def test_is_transient_backend_message_matches(message):
    assert is_transient_backend_message(message)


@pytest.mark.parametrize(
    "message",
    ["", "Could not parse LLM response", "No suggested terms returned"],
)
def test_is_transient_backend_message_does_not_match(message):
    assert not is_transient_backend_message(message)


def test_skip_on_backend_unavailable_skips_transient():
    @skip_on_backend_unavailable
    def failing():
        raise _StatusError(502)

    with pytest.raises(pytest.skip.Exception):
        failing()


def test_skip_on_backend_unavailable_reraises_real_failures():
    @skip_on_backend_unavailable
    def failing():
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        failing()
