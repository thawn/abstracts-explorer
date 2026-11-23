"""
Integration-style tests for Pydantic AI retries with our rate-limit transports.

Network calls are mocked, but Pydantic AI's provider, the OpenAI SDK retry
loop, the cooldown transport, and the RAG agent wiring are real.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import abstracts_explorer.rag.agent as agent_module
from abstracts_explorer.config import get_config
from abstracts_explorer.rag import RAGChat
from abstracts_explorer.rate_limiter import CooldownAwareTransport, set_global_cooldown


def _ok(request, content: str = "ok") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        request=request,
    )


def _error(request, status: int = 429, headers=None) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers=headers or {},
        json={"error": {"message": "test error", "type": "error"}},
        request=request,
    )


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch):
    """Skip the SDK's own backoff sleep without touching the global time module."""
    from openai._base_client import AsyncAPIClient

    async def _async_no_sleep(self, **kwargs):
        return None

    monkeypatch.setattr(AsyncAPIClient, "_sleep_for_retry", _async_no_sleep)


@pytest.fixture
def mock_database():
    """Minimal mock database for RAGChat."""
    return Mock()


class TestPydanticAIRetryIntegration:
    """The Pydantic AI provider with our cooldown transport."""

    def test_agent_retries_429_and_defers_cooldown(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return _error(request, headers={"retry-after": "0"})
            return _ok(request)

        cooldown = Mock()
        cooldown.async_wait = AsyncMock()
        transport = CooldownAwareTransport(
            httpx2.MockTransport(handler),
            rate_limit_delay=0.0,
            cooldown=cooldown,
        )
        client = httpx2.AsyncClient(transport=transport)
        model = OpenAIChatModel(
            "test-model",
            provider=OpenAIProvider(
                base_url="http://test/v1",
                api_key="test-key",
                http_client=client,
            ),
        )
        agent = Agent(model, model_settings={"max_retries": 1})
        try:
            result = agent.run_sync("hello")
        finally:
            asyncio.run(client.aclose())

        assert result.output == "ok"
        assert calls["n"] == 2
        assert cooldown.async_wait.await_count == 2
        cooldown.defer.assert_called_once_with(0.0)

    def test_rag_chat_retries_429_through_production_wiring(
        self, mock_embeddings_manager, mock_database, monkeypatch
    ):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return _error(request, headers={"retry-after": "0"})
            return _ok(request)

        config = get_config()
        monkeypatch.setattr(config, "requests_per_minute", 0)
        monkeypatch.setattr(
            agent_module.httpx2,
            "AsyncHTTPTransport",
            lambda *args, **kwargs: httpx2.MockTransport(handler),
        )

        cooldown = Mock()
        cooldown.async_wait = AsyncMock()
        set_global_cooldown(cooldown)

        chat = RAGChat(
            mock_embeddings_manager,
            mock_database,
            backend_url="http://test",
            model="test-model",
        )
        try:
            result = chat.query("hello", limit=1)
        finally:
            chat.close()

        assert result["response"] == "ok"
        assert calls["n"] == 2
        cooldown.defer.assert_called_once()
