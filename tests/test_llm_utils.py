"""
Tests for shared LLM chat-completion helpers.
"""

from unittest.mock import Mock

import pytest

from abstracts_explorer.llm_utils import chat_completion_text


def _client_with_response(response):
    client = Mock()
    client.chat.completions.create.return_value = response
    return client


class TestChatCompletionText:
    """Tests for :func:`chat_completion_text`."""

    def test_returns_stripped_content(self):
        """The response content is returned without surrounding whitespace."""
        response = Mock()
        response.choices = [Mock(message=Mock(content="  hello \n"))]

        result = chat_completion_text(_client_with_response(response), "model", "system", "user")

        assert result == "hello"

    def test_none_content_returns_empty_string(self):
        """A missing content field yields an empty string."""
        response = Mock()
        response.choices = [Mock(message=Mock(content=None))]

        assert chat_completion_text(_client_with_response(response), "model", "system", "user") == ""

    def test_empty_choices_raises_clear_error(self):
        """A response without choices raises an actionable ValueError."""
        response = Mock()
        response.choices = []

        with pytest.raises(ValueError, match="contained no choices"):
            chat_completion_text(_client_with_response(response), "model", "system", "user")

    def test_missing_choices_attribute_raises_clear_error(self):
        """A malformed response without choices also raises a clear ValueError."""
        response = Mock(spec=["id"])

        with pytest.raises(ValueError, match="contained no choices"):
            chat_completion_text(_client_with_response(response), "model", "system", "user")

    def test_omits_optional_request_options_when_not_set(self):
        """Optional timeout/reasoning hints are omitted by default."""
        response = Mock()
        response.choices = [Mock(message=Mock(content="ok"))]
        client = _client_with_response(response)

        chat_completion_text(client, "model", "system", "user")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "timeout" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 1000

    def test_forwards_optional_request_options(self):
        """Timeout and reasoning hints are forwarded when provided."""
        response = Mock()
        response.choices = [Mock(message=Mock(content="ok"))]
        client = _client_with_response(response)

        chat_completion_text(
            client,
            "model",
            "system",
            "user",
            temperature=0.7,
            max_tokens=42,
            timeout=5,
            reasoning_effort="low",
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 42
        assert kwargs["timeout"] == 5
        assert kwargs["reasoning_effort"] == "low"
