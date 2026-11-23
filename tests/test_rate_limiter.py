"""Tests for thread-safe, async-compatible TokenBucketRateLimiter."""

import asyncio
import threading
import time

import httpx2
import pytest

from abstracts_explorer.rate_limiter import RateLimitedTransport
from abstracts_explorer.rate_limiter import (
    TokenBucketRateLimiter,
    get_global_rate_limiter,
    set_global_rate_limiter,
)
from tests.helpers import FakeTime


@pytest.fixture
def fake_time(monkeypatch):
    """Replace the rate limiter's clock and sleepers with a virtual clock."""
    clock = FakeTime()

    async def fake_async_sleep(seconds: float) -> None:
        clock.sleep(seconds)

    monkeypatch.setattr("abstracts_explorer.rate_limiter.time", clock)
    monkeypatch.setattr("abstracts_explorer.rate_limiter.asyncio.sleep", fake_async_sleep)
    return clock


class TestTokenBucketInitialization:
    def test_initial_state_full_capacity(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.requests_per_minute == 60
        assert limiter._tokens == pytest.approx(60.0)

    def test_zero_requests_per_minute(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=0)
        assert limiter.requests_per_minute == 0
        assert limiter._tokens == 0.0

    def test_negative_requests_per_minute_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            TokenBucketRateLimiter(requests_per_minute=-1)

    def test_high_requests_per_minute(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=1000)
        assert limiter.requests_per_minute == 1000
        assert limiter._tokens == pytest.approx(1000.0)

    def test_burst_limits_initial_capacity(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=1)
        assert limiter.burst == 1
        assert limiter._tokens == pytest.approx(1.0)

    def test_zero_burst_is_clamped_to_one(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=0)
        assert limiter.burst == 1
        assert limiter._tokens == pytest.approx(1.0)

    def test_negative_burst_raises(self):
        with pytest.raises(ValueError, match="burst must be non-negative"):
            TokenBucketRateLimiter(requests_per_minute=60, burst=-1)


class TestTokenBucketAcquire:
    def test_acquire_single_token(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter.acquire(tokens=1.0)
        assert limiter._tokens == pytest.approx(59.0, abs=0.1)

    def test_acquire_multiple_tokens(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter.acquire(tokens=10.0)
        assert limiter._tokens == pytest.approx(110.0, abs=0.1)

    def test_acquire_all_tokens(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=30)
        limiter.acquire(tokens=30.0)
        assert limiter._tokens == pytest.approx(0.0, abs=0.1)


class TestTokenBucketTryAcquire:
    def test_try_acquire_success(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=2)
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    def test_time_until_available_zero_when_ready(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        assert limiter.time_until_available() == 0.0

    def test_time_until_available_reports_positive_wait(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        wait = limiter.time_until_available()
        assert 0.0 < wait <= 1.0

    def test_time_until_available_zero_rate_is_infinite(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=0)
        assert limiter.time_until_available() == float("inf")


class TestTokenBucketRefill:
    def test_refill_rate(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 0.0
        limiter._last_update = time.monotonic() - 1.0
        limiter._refill()
        assert limiter._tokens == pytest.approx(2.0, abs=0.1)

    def test_refill_capped_at_max(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._last_update = time.monotonic() - 100.0
        limiter._refill()
        assert limiter._tokens == pytest.approx(60.0)

    def test_refill_no_elapsed_time(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 30.0
        before = limiter._tokens
        limiter._refill()
        assert limiter._tokens == pytest.approx(before, abs=0.01)


class TestTokenBucketBlocking:
    def test_blocks_until_token_available(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        limiter._last_update = fake_time.monotonic()

        start = fake_time.monotonic()
        limiter.acquire(tokens=1.0)
        elapsed = fake_time.monotonic() - start

        assert elapsed == pytest.approx(1.0, abs=0.01)
        assert sum(fake_time.sleeps) == pytest.approx(1.0, abs=0.01)

    def test_does_not_block_when_token_available(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        start = fake_time.monotonic()
        limiter.acquire(tokens=1.0)
        elapsed = fake_time.monotonic() - start

        assert elapsed == 0.0
        assert fake_time.sleeps == []


class TestTokenBucketReset:
    def test_reset_restores_capacity(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter.acquire(tokens=30.0)
        limiter.reset()
        assert limiter._tokens == pytest.approx(60.0, abs=0.1)


class TestGlobalRateLimiter:
    def test_get_returns_none_initially(self):
        set_global_rate_limiter(None)
        assert get_global_rate_limiter() is None

    def test_set_and_get(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=30)
        set_global_rate_limiter(limiter)
        assert get_global_rate_limiter() is limiter

    def test_clear_global(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=30)
        set_global_rate_limiter(limiter)
        set_global_rate_limiter(None)
        assert get_global_rate_limiter() is None


class TestMultithreadedRateLimiting:
    def test_concurrent_acquire_total_requests(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 2.0
        limiter._last_update = fake_time.monotonic()

        acquired = []
        lock = threading.Lock()

        def worker():
            for _ in range(2):
                limiter.acquire(tokens=1.0)
                with lock:
                    acquired.append(fake_time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]

        start = fake_time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = fake_time.monotonic() - start

        assert len(acquired) == 10
        assert elapsed == pytest.approx(4.0, abs=0.2)

    def test_concurrent_acquire_spread(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 1.0
        limiter._last_update = fake_time.monotonic()

        timestamps = []
        lock = threading.Lock()

        def worker():
            limiter.acquire(tokens=1.0)
            with lock:
                timestamps.append(fake_time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(10)]

        start = fake_time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = fake_time.monotonic() - start

        assert len(timestamps) == 10
        assert elapsed == pytest.approx(9.0, abs=0.2)
        assert len(fake_time.sleeps) >= 9

    def test_no_deadlock_many_threads(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=600)
        limiter._tokens = 50.0
        limiter._last_update = fake_time.monotonic()

        completed = []
        lock = threading.Lock()

        def worker():
            for _ in range(3):
                limiter.acquire(tokens=1.0)
            with lock:
                completed.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]

        start = fake_time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(completed) == 20
        assert fake_time.monotonic() - start == pytest.approx(1.0, abs=0.2)


class TestWaitressScenario:
    def test_six_threads_simultaneous_burst(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 3.0
        limiter._last_update = fake_time.monotonic()

        timestamps = []
        lock = threading.Lock()

        def worker(pid: int):
            limiter.acquire(tokens=1.0)
            with lock:
                timestamps.append(pid)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]

        start = fake_time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = fake_time.monotonic() - start

        assert len(timestamps) == 6
        assert elapsed == pytest.approx(3.0, abs=0.2)

    def test_waitress_threads_no_requests_lost(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 30.0
        limiter._last_update = fake_time.monotonic()

        success_count = []
        lock = threading.Lock()

        def worker():
            for _ in range(6):
                limiter.acquire(tokens=1.0)
            with lock:
                success_count.append(1)

        threads = [threading.Thread(target=worker) for _ in range(6)]

        start = fake_time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = fake_time.monotonic() - start

        assert len(success_count) == 6
        assert elapsed == pytest.approx(3.0, abs=0.2)


class TestAsyncTokenBucketAcquire:
    def test_async_acquire_single_token(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        asyncio.run(limiter.async_acquire(tokens=1.0))
        assert limiter._tokens == pytest.approx(59.0, abs=0.5)

    def test_async_acquire_multiple_tokens(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        asyncio.run(limiter.async_acquire(tokens=10.0))
        assert limiter._tokens == pytest.approx(110.0, abs=0.5)

    def test_async_blocks_until_token_available(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        limiter._last_update = fake_time.monotonic()

        async def acquire():
            await limiter.async_acquire(tokens=1.0)

        start = fake_time.monotonic()
        asyncio.run(acquire())
        elapsed = fake_time.monotonic() - start

        assert elapsed == pytest.approx(1.0, abs=0.01)
        assert sum(fake_time.sleeps) == pytest.approx(1.0, abs=0.01)

    def test_async_does_not_block_when_token_available(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        async def acquire():
            await limiter.async_acquire(tokens=1.0)

        start = fake_time.monotonic()
        asyncio.run(acquire())
        elapsed = fake_time.monotonic() - start

        assert elapsed == 0.0
        assert fake_time.sleeps == []


class TestRateLimitedTransport:
    def test_transport_init_creates_global_limiter(self):
        set_global_rate_limiter(None)
        _ = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=60)
        gl = get_global_rate_limiter()
        assert gl is not None
        assert gl.requests_per_minute == 60
        assert gl.burst == 1
        assert gl._tokens == pytest.approx(1.0)

    def test_transport_accepts_custom_burst(self):
        set_global_rate_limiter(None)
        _ = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=60, burst=5)
        gl = get_global_rate_limiter()
        assert gl is not None
        assert gl.burst == 5
        assert gl._tokens == pytest.approx(5.0)

    def test_async_transport_forwards_request(self):
        set_global_rate_limiter(None)

        def handler(request):
            return httpx2.Response(200, json={"ok": True})

        transport = RateLimitedTransport(httpx2.MockTransport(handler), requests_per_minute=60)

        async def send():
            request = httpx2.Request("GET", "http://example.com")
            response = await transport.handle_async_request(request)
            return response

        response = asyncio.run(send())
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_async_transport_acquires_token(self):
        set_global_rate_limiter(None)
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        set_global_rate_limiter(limiter)
        before = limiter._tokens

        transport = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=60)

        async def send():
            request = httpx2.Request("GET", "http://example.com")
            return await transport.handle_async_request(request)

        asyncio.run(send())
        assert limiter._tokens < before


class TestSyncAsyncSharedLimiter:
    def test_sync_and_async_share_same_limiter(self):
        set_global_rate_limiter(None)
        sync_transport = RateLimitedTransport(httpx2.HTTPTransport(), requests_per_minute=60)
        async_transport = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=60)
        assert sync_transport._rate_limiter is async_transport._rate_limiter

    def test_async_depletes_tokens_for_sync(self):
        set_global_rate_limiter(None)
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 30.0
        limiter._last_update = time.monotonic()
        set_global_rate_limiter(limiter)

        _ = RateLimitedTransport(httpx2.AsyncHTTPTransport(), requests_per_minute=60)

        async def acquire_all():
            for _ in range(25):
                await limiter.async_acquire(tokens=1.0)

        asyncio.run(acquire_all())
        limiter._tokens = limiter._tokens  # no-op to confirm access
        sync_after = limiter._tokens
        assert sync_after <= 6

    def test_sync_depletes_tokens_for_async(self):
        set_global_rate_limiter(None)
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 30.0
        limiter._last_update = time.monotonic()
        set_global_rate_limiter(limiter)

        limiter.acquire(tokens=25.0)
        assert limiter._tokens <= 6


class TestMultithreadedAsyncRateLimiting:
    def test_concurrent_async_acquire_total_requests(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 2.0
        limiter._last_update = fake_time.monotonic()

        acquired = []
        lock = threading.Lock()

        async def worker():
            for _ in range(2):
                await limiter.async_acquire(tokens=1.0)
                with lock:
                    acquired.append(fake_time.monotonic())

        async def run_all():
            await asyncio.gather(*[worker() for _ in range(5)])

        start = fake_time.monotonic()
        asyncio.run(run_all())
        elapsed = fake_time.monotonic() - start

        assert len(acquired) == 10
        assert elapsed == pytest.approx(4.0, abs=0.2)

    def test_no_deadlock_async_many_tasks(self, fake_time):
        limiter = TokenBucketRateLimiter(requests_per_minute=600)
        limiter._tokens = 50.0
        limiter._last_update = fake_time.monotonic()

        completed = []
        lock = threading.Lock()

        async def worker():
            for _ in range(3):
                await limiter.async_acquire(tokens=1.0)
            with lock:
                completed.append(1)

        async def run_all():
            await asyncio.gather(*[worker() for _ in range(20)])

        start = fake_time.monotonic()
        asyncio.run(run_all())
        elapsed = fake_time.monotonic() - start

        assert len(completed) == 20
        assert elapsed == pytest.approx(1.0, abs=0.2)

    def test_concurrent_async_tasks_spread(self, fake_time):
        """Multiple concurrent async tasks should be spread out over time, not all fire at once."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 1.0
        limiter._last_update = fake_time.monotonic()

        timestamps = []
        lock = threading.Lock()

        async def worker():
            await limiter.async_acquire(tokens=1.0)
            with lock:
                timestamps.append(fake_time.monotonic())

        async def run_all():
            await asyncio.gather(*[worker() for _ in range(10)])

        start = fake_time.monotonic()
        asyncio.run(run_all())
        elapsed = fake_time.monotonic() - start

        assert len(timestamps) == 10
        assert elapsed == pytest.approx(9.0, abs=0.2)
        assert len(fake_time.sleeps) >= 9

    def test_async_and_sync_interleaved(self, fake_time):
        """Sync and async acquires should interleave correctly, consuming the same token pool."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 5.0
        limiter._last_update = fake_time.monotonic()

        acquired = []
        lock = threading.Lock()

        def sync_worker():
            limiter.acquire(tokens=1.0)
            with lock:
                acquired.append("sync")

        async def async_worker():
            await limiter.async_acquire(tokens=1.0)
            with lock:
                acquired.append("async")

        async def run_async_tasks():
            await asyncio.gather(*[async_worker() for _ in range(3)])

        threads = [threading.Thread(target=sync_worker) for _ in range(2)]
        for t in threads:
            t.start()
        asyncio.run(run_async_tasks())
        for t in threads:
            t.join()

        assert len(acquired) == 5
        assert acquired.count("sync") == 2
        assert acquired.count("async") == 3
        assert fake_time.sleeps == []


class _SyncInner(httpx2.BaseTransport):
    """Minimal sync transport for cooldown tests."""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.closed = False

    def handle_request(self, request):
        self.calls += 1
        return self.response

    def close(self):
        self.closed = True


class _AsyncInner(httpx2.AsyncBaseTransport):
    """Minimal async transport for cooldown tests."""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.closed = False

    async def handle_async_request(self, request):
        self.calls += 1
        return self.response

    async def aclose(self):
        self.closed = True


def _response(status, headers=None):
    return httpx2.Response(status, headers=headers or {})


class TestParseRetryAfterSeconds:
    """Tests for the shared Retry-After parser."""

    def test_retry_after_ms_takes_precedence(self):
        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        headers = httpx2.Headers({"retry-after": "30", "retry-after-ms": "1500"})
        assert parse_retry_after_seconds(headers) == pytest.approx(1.5)

    def test_retry_after_seconds(self):
        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        assert parse_retry_after_seconds(httpx2.Headers({"retry-after": "12"})) == pytest.approx(12.0)

    def test_retry_after_http_date(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        headers = httpx2.Headers({"retry-after": format_datetime(retry_at, usegmt=True)})
        delay = parse_retry_after_seconds(headers)
        assert delay is not None
        assert 25 <= delay <= 35

    def test_invalid_value_returns_none(self):
        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        assert parse_retry_after_seconds(httpx2.Headers({"retry-after": "not-a-date"})) is None

    def test_missing_headers_returns_none(self):
        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        assert parse_retry_after_seconds(None) is None

    def test_caps_at_max_wait(self):
        from abstracts_explorer.rate_limiter import parse_retry_after_seconds

        headers = httpx2.Headers({"retry-after": "9999"})
        assert parse_retry_after_seconds(headers, max_wait=120) == 120.0


class TestCooldownAwareTransport:
    """Tests for the shared-cooldown LLM transport."""

    def test_sync_429_defers_cooldown(self):
        from unittest.mock import Mock

        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        inner = _SyncInner(_response(429, {"retry-after": "12"}))
        cooldown = Mock()
        transport = CooldownAwareTransport(inner, cooldown=cooldown)

        transport.handle_request(httpx2.Request("POST", "http://test"))

        cooldown.wait.assert_called_once_with()
        cooldown.defer.assert_called_once_with(12.0)
        assert inner.calls == 1

    def test_sync_429_uses_fallback_delay(self):
        from unittest.mock import Mock

        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        inner = _SyncInner(_response(429))
        cooldown = Mock()
        transport = CooldownAwareTransport(inner, rate_limit_delay=45.0, cooldown=cooldown)

        transport.handle_request(httpx2.Request("POST", "http://test"))

        cooldown.defer.assert_called_once_with(45.0)

    def test_sync_non_429_does_not_defer(self):
        from unittest.mock import Mock

        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        inner = _SyncInner(_response(200))
        cooldown = Mock()
        transport = CooldownAwareTransport(inner, cooldown=cooldown)

        response = transport.handle_request(httpx2.Request("POST", "http://test"))

        assert response.status_code == 200
        cooldown.wait.assert_called_once_with()
        cooldown.defer.assert_not_called()

    def test_sync_close_delegates(self):
        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        inner = _SyncInner(_response(200))
        transport = CooldownAwareTransport(inner)
        transport.close()
        assert inner.closed is True

    def test_async_429_defers_cooldown(self):
        from unittest.mock import AsyncMock, Mock

        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        async def run():
            inner = _AsyncInner(_response(429, {"retry-after-ms": "2500"}))
            cooldown = Mock()
            cooldown.async_wait = AsyncMock()
            transport = CooldownAwareTransport(inner, cooldown=cooldown)

            await transport.handle_async_request(httpx2.Request("POST", "http://test"))
            await transport.aclose()

            return inner, cooldown

        inner, cooldown = asyncio.run(run())
        cooldown.async_wait.assert_awaited_once_with()
        cooldown.defer.assert_called_once_with(2.5)
        assert inner.calls == 1
        assert inner.closed is True

    def test_sync_transport_rejects_async_inner(self):
        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        transport = CooldownAwareTransport(_AsyncInner(_response(200)))
        with pytest.raises(TypeError):
            transport.handle_request(httpx2.Request("GET", "http://test"))

    def test_async_transport_rejects_sync_inner(self):
        from abstracts_explorer.rate_limiter import CooldownAwareTransport

        async def run():
            transport = CooldownAwareTransport(_SyncInner(_response(200)))
            with pytest.raises(TypeError):
                await transport.handle_async_request(httpx2.Request("GET", "http://test"))

        asyncio.run(run())


def test_reuses_global_limiter_and_warns(caplog):
    """A second transport with a different RPM reuses the global limiter and warns."""
    set_global_rate_limiter(None)
    RateLimitedTransport(httpx2.HTTPTransport(), requests_per_minute=60)

    with caplog.at_level("WARNING", logger="abstracts_explorer.rate_limiter"):
        RateLimitedTransport(httpx2.HTTPTransport(), requests_per_minute=120)

    limiter = get_global_rate_limiter()
    assert limiter is not None
    assert limiter.requests_per_minute == 60
    assert "Reusing existing global rate limiter" in caplog.text
