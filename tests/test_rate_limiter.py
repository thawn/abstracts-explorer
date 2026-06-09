"""Tests for thread-safe TokenBucketRateLimiter."""

import threading
import time

import pytest

from abstracts_explorer.rate_limiter import (
    TokenBucketRateLimiter,
    get_global_rate_limiter,
    set_global_rate_limiter,
)


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
    def test_blocks_until_token_available(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 0.0
        limiter._last_update = time.monotonic()

        start = time.monotonic()
        limiter.acquire(tokens=1.0)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.8

    def test_does_not_block_when_token_available(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        start = time.monotonic()
        limiter.acquire(tokens=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1


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
    def test_concurrent_acquire_total_requests(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 2.0
        limiter._last_update = time.monotonic()

        acquired = []
        lock = threading.Lock()

        def worker():
            for _ in range(2):
                limiter.acquire(tokens=1.0)
                with lock:
                    acquired.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]

        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        assert len(acquired) == 10
        assert elapsed >= 2.0

    def test_concurrent_acquire_spread(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 1.0
        limiter._last_update = time.monotonic()

        timestamps = []
        lock = threading.Lock()

        def worker():
            limiter.acquire(tokens=1.0)
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(timestamps) == 10
        sorted_ts = sorted(timestamps)
        for i in range(1, len(sorted_ts)):
            gap = sorted_ts[i] - sorted_ts[i - 1]
            assert gap >= 0.5

    def test_no_deadlock_many_threads(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=600)
        limiter._tokens = 50.0
        limiter._last_update = time.monotonic()

        completed = []
        lock = threading.Lock()

        def worker():
            for _ in range(3):
                limiter.acquire(tokens=1.0)
            with lock:
                completed.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(completed) == 20


class TestWaitressScenario:
    def test_six_threads_simultaneous_burst(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60)
        limiter._tokens = 3.0
        limiter._last_update = time.monotonic()

        timestamps = []
        lock = threading.Lock()

        def worker(pid: int):
            limiter.acquire(tokens=1.0)
            with lock:
                timestamps.append(pid)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]

        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        assert len(timestamps) == 6
        assert elapsed >= 2.0
        assert elapsed < 10.0

    def test_waitress_threads_no_requests_lost(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=120)
        limiter._tokens = 30.0
        limiter._last_update = time.monotonic()

        success_count = []
        lock = threading.Lock()

        def worker():
            for _ in range(6):
                limiter.acquire(tokens=1.0)
            with lock:
                success_count.append(1)

        threads = [threading.Thread(target=worker) for _ in range(6)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(success_count) == 6
        assert len(success_count) == 6
