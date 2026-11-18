"""Tests for utils/http_utils.py."""

import httpx2

from abstracts_explorer.utils.http_utils import (
    build_llm_timeout,
    is_llm_connection_error,
    is_llm_timeout,
)


class TestIsLlmTimeout:
    """Tests for LLM backend timeout detection."""

    def test_httpx_timeout(self):
        assert is_llm_timeout(httpx2.ReadTimeout("timed out"))

    def test_builtin_timeout(self):
        assert is_llm_timeout(TimeoutError("timed out"))

    def test_wrapped_timeout(self):
        try:
            try:
                raise httpx2.ConnectTimeout("connection timed out")
            except Exception as exc:
                raise RuntimeError("outer failure") from exc
        except RuntimeError as exc:
            assert is_llm_timeout(exc)

    def test_timeout_class_name(self):
        class APITimeoutError(Exception):
            pass

        assert is_llm_timeout(APITimeoutError("request failed"))

    def test_pydantic_ai_timeout_class_names(self):
        class ModelRequestTimeout(Exception):
            pass

        class ModelTimeoutError(Exception):
            pass

        assert is_llm_timeout(ModelRequestTimeout("request deadline exceeded"))
        assert is_llm_timeout(ModelTimeoutError("attempt timed out"))

    def test_non_timeout(self):
        assert not is_llm_timeout(RuntimeError("kaboom"))


class TestIsLlmConnectionError:
    """Tests for LLM backend connection failure detection."""

    def test_httpx_connection_error(self):
        assert is_llm_connection_error(httpx2.ConnectError("connection refused"))

    def test_connection_class_name(self):
        class APIConnectionError(Exception):
            pass

        assert is_llm_connection_error(APIConnectionError("cannot connect"))

    def test_timeout_is_not_connection_error(self):
        assert not is_llm_connection_error(httpx2.ReadTimeout("timed out"))

    def test_non_connection_error(self):
        assert not is_llm_connection_error(RuntimeError("kaboom"))


class TestBuildLlmTimeout:
    """Tests for structured timeout construction."""

    def test_builds_expected_timeout(self):
        timeout = build_llm_timeout(60, connect_timeout=10)
        assert timeout.connect == 10
        assert timeout.read == 60
        assert timeout.write == 60
        assert timeout.pool == 10
