"""
Integration-style tests for SDK retries plus our rate-limit transports.

The network is mocked with ``httpx2.MockTransport``, but the real OpenAI SDK
retry loop, our cooldown transport, and the token bucket are exercised together.
"""

import threading
import time
from typing import Any
from unittest.mock import Mock

import httpx2
import pytest
from openai import AuthenticationError, OpenAI, RateLimitError

from abstracts_explorer.rate_limiter import (
    CooldownAwareTransport,
    RateLimitCooldown,
    RateLimitedTransport,
    TokenBucketRateLimiter,
    set_global_rate_limiter,
)

_MESSAGES: list[Any] = [{"role": "user", "content": "hello"}]


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


def _build_client(handler, *, max_attempts: int = 3, rpm: int = 0, cooldown=None) -> OpenAI:
    inner = httpx2.MockTransport(handler)
    if rpm > 0:
        inner = RateLimitedTransport(inner, requests_per_minute=rpm)
    transport = CooldownAwareTransport(inner, rate_limit_delay=0.0, cooldown=cooldown)
    return OpenAI(
        base_url="http://test/v1",
        api_key="test-key",
        http_client=httpx2.Client(transport=transport),
        max_retries=max_attempts - 1,
    )


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch):
    """Skip the SDK's own backoff sleep without touching the global time module."""
    from openai._base_client import AsyncAPIClient, SyncAPIClient

    monkeypatch.setattr(SyncAPIClient, "_sleep_for_retry", lambda self, **kwargs: None)

    async def _async_no_sleep(self, **kwargs):
        return None

    monkeypatch.setattr(AsyncAPIClient, "_sleep_for_retry", _async_no_sleep)


class TestSDKRetryIntegration:
    """The real SDK retry loop and our transports working together."""

    def test_persistent_429_uses_full_attempt_budget(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return _error(request)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=3, cooldown=cooldown)
        try:
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert calls["n"] == 3
        assert cooldown.wait.call_count == 3
        assert cooldown.defer.call_count == 3

    def test_retry_after_header_drives_cooldown(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return _error(request, headers={"retry-after": "12"})
            return _ok(request)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=2, cooldown=cooldown)
        try:
            response = client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert response.choices[0].message.content == "ok"
        assert calls["n"] == 2
        cooldown.defer.assert_called_once_with(12.0)

    def test_retry_after_ms_drives_cooldown(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return _error(request, headers={"retry-after-ms": "1500"})
            return _ok(request)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=2, cooldown=cooldown)
        try:
            client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        cooldown.defer.assert_called_once_with(1.5)

    def test_503_then_success(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return _error(request, status=503)
            return _ok(request)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=3, cooldown=cooldown)
        try:
            response = client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert response.choices[0].message.content == "ok"
        assert calls["n"] == 2
        cooldown.defer.assert_not_called()

    def test_401_is_not_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return _error(request, status=401)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=3, cooldown=cooldown)
        try:
            with pytest.raises(AuthenticationError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert calls["n"] == 1
        cooldown.defer.assert_not_called()

    def test_x_should_retry_false_is_not_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return _error(request, headers={"x-should-retry": "false"})

        cooldown = Mock()
        client = _build_client(handler, max_attempts=3, cooldown=cooldown)
        try:
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert calls["n"] == 1
        cooldown.defer.assert_called_once()

    def test_retry_after_above_sdk_cap_is_not_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return _error(request, headers={"retry-after": "9999"})

        cooldown = Mock()
        client = _build_client(handler, max_attempts=3, cooldown=cooldown)
        try:
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert calls["n"] == 1
        cooldown.defer.assert_called_once_with(9999.0)

    def test_retries_consume_rate_limit_tokens(self, monkeypatch):
        set_global_rate_limiter(None)
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        set_global_rate_limiter(limiter)
        acquired = []
        monkeypatch.setattr(limiter, "acquire", lambda tokens=1.0: acquired.append(tokens))

        def handler(request):
            return _error(request)

        client = _build_client(handler, max_attempts=3, rpm=60, cooldown=Mock())
        try:
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert acquired == [1.0, 1.0, 1.0]

    def test_cooldown_still_applies_without_rate_limiting(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return _error(request)

        cooldown = Mock()
        client = _build_client(handler, max_attempts=2, rpm=0, cooldown=cooldown)
        try:
            with pytest.raises(RateLimitError):
                client.chat.completions.create(model="test-model", messages=_MESSAGES)
        finally:
            client.close()

        assert calls["n"] == 2
        assert cooldown.wait.call_count == 2
        assert cooldown.defer.call_count == 2


class _EventCooldown(RateLimitCooldown):
    """Cooldown that signals when a deferral has been recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.deferred = threading.Event()

    def defer(self, delay: float) -> None:
        super().defer(delay)
        self.deferred.set()


class TestSharedCooldownAcrossClients:
    """A 429 on one client pauses other clients sharing the cooldown."""

    def test_second_client_waits_for_shared_cooldown(self):
        cooldown = _EventCooldown()
        first_calls = {"n": 0}

        def first_handler(request):
            first_calls["n"] += 1
            if first_calls["n"] == 1:
                return _error(request, headers={"retry-after": "0.2"})
            return _ok(request)

        def second_handler(request):
            return _ok(request)

        first_client = _build_client(first_handler, max_attempts=2, cooldown=cooldown)
        second_client = _build_client(second_handler, max_attempts=1, cooldown=cooldown)

        def run_first():
            try:
                first_client.chat.completions.create(model="test-model", messages=_MESSAGES)
            finally:
                first_client.close()

        thread = threading.Thread(target=run_first)
        try:
            thread.start()
            assert cooldown.deferred.wait(timeout=5)

            start = time.monotonic()
            second_client.chat.completions.create(model="test-model", messages=_MESSAGES)
            elapsed = time.monotonic() - start
        finally:
            thread.join(timeout=5)
            second_client.close()

        assert not thread.is_alive()
        assert elapsed >= 0.1


class _WindowRateLimitBackend:
    """Mock backend that allows a fixed number of requests per one-second window."""

    def __init__(self, clock: Any, limit_per_window: int = 1) -> None:
        self._clock = clock
        self._limit = limit_per_window
        self._lock = threading.Lock()
        self._windows: dict[int, int] = {}
        self.accepted = 0
        self.rejected = 0
        self.arrivals: list[float] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        now = self._clock.monotonic()
        window = int(now)
        with self._lock:
            count = self._windows.get(window, 0) + 1
            self._windows[window] = count
            self.arrivals.append(now)
            if count > self._limit:
                self.rejected += 1
                return _error(request, headers={"retry-after": "1"})
            self.accepted += 1
        return _ok(request, content="Label: Parallel Cluster\nDescription: A parallel cluster.")


class TestClusterLabelRateLimitE2E:
    """End-to-end cluster labeling against a mocked rate-limited backend."""

    def test_parallel_labeling_respects_60_requests_per_minute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import abstracts_explorer.embeddings.models as models_module
        from abstracts_explorer.clustering.representation import ProjectLLMRepresentation
        from abstracts_explorer.embeddings import EmbeddingsManager
        from tests.helpers import FakeTime

        clock = FakeTime()
        monkeypatch.setattr("abstracts_explorer.rate_limiter.time", clock)

        backend = _WindowRateLimitBackend(clock, limit_per_window=1)
        monkeypatch.setattr(
            models_module.httpx2,
            "HTTPTransport",
            lambda *args, **kwargs: httpx2.MockTransport(backend.handle),
        )

        embeddings_manager = EmbeddingsManager(backend_url="http://test", requests_per_minute=60)
        client = embeddings_manager.openai_client
        try:
            representation = ProjectLLMRepresentation(embeddings_manager, max_workers=8)
            topic_model = Mock()
            topic_model._extract_representative_docs.return_value = (
                {topic: [f"doc {topic}"] for topic in range(25)},
                None,
                None,
                None,
            )
            topics = {topic: [(f"keyword{topic}", 1.0)] for topic in range(25)}

            updated = representation.extract_topics(topic_model, ["d"], None, topics)
        finally:
            client.close()

        assert len(updated) == 25
        assert all(updated[topic][0][0].startswith("Parallel Cluster") for topic in range(25))
        assert backend.accepted == 25
        assert backend.rejected == 0
        assert max(backend._windows.values()) <= 1
        assert backend.arrivals[-1] - backend.arrivals[0] >= 24.0
