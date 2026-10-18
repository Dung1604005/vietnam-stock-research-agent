from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import TypeVar
from stock_multi_agent.services.request_telemetry import current_metrics


T = TypeVar("T")


@dataclass(frozen=True)
class ResiliencePolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.4
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: int = 60
    min_request_interval_seconds: float = 0.25


@dataclass(frozen=True)
class ProviderHealth:
    attempts_total: int
    successful_operations: int
    failed_operations: int
    retries_total: int
    circuit_state: str
    consecutive_failures: int
    rate_limit_wait_seconds: float


class CircuitOpenError(RuntimeError):
    pass


def is_rate_limit_error(error: BaseException) -> bool:
    """Recognize upstream quota responses without coupling to one provider's exception type."""
    text = str(error).casefold()
    return any(marker in text for marker in (
        "rate limit", "rate-limit", "too many requests", "http 429", "status 429",
        "hết hạn mức", "het han muc", "quota exceeded",
    ))


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.opened_at: datetime | None = None

    def before_call(self) -> None:
        if self.opened_at is None:
            return
        if datetime.now(timezone.utc) - self.opened_at >= timedelta(seconds=self.cooldown_seconds):
            self.failures = 0
            self.opened_at = None
            return
        raise CircuitOpenError("Market-data circuit is temporarily open after repeated provider failures")

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = datetime.now(timezone.utc)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if datetime.now(timezone.utc) - self.opened_at >= timedelta(seconds=self.cooldown_seconds):
            return "half_open"
        return "open"


class RateLimiter:
    """In-process request pacing for a single upstream provider.

    It deliberately paces *attempts*, including retries, so a temporary outage
    cannot turn into a burst of calls. It is injectable for deterministic tests.
    """

    def __init__(
        self,
        min_interval_seconds: float,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.sleeper = sleeper
        self.clock = clock
        self._next_allowed = 0.0
        self._lock = Lock()
        self.total_wait_seconds = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self.clock()
            wait_seconds = max(0.0, self._next_allowed - now)
            if wait_seconds:
                self.total_wait_seconds += wait_seconds
                self.sleeper(wait_seconds)
                now = self.clock()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval_seconds


class ResilientExecutor:
    def __init__(
        self,
        policy: ResiliencePolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.policy = policy or ResiliencePolicy()
        self.sleeper = sleeper
        self.breaker = CircuitBreaker(
            self.policy.circuit_failure_threshold,
            self.policy.circuit_cooldown_seconds,
        )
        self.rate_limiter = rate_limiter or RateLimiter(self.policy.min_request_interval_seconds, sleeper=sleeper)
        self.attempts_total = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.retries_total = 0

    def run(self, operation: Callable[[], T]) -> T:
        self.breaker.before_call()
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.policy.max_attempts) + 1):
            try:
                self.rate_limiter.acquire()
                self.attempts_total += 1
                metrics = current_metrics()
                if metrics:
                    metrics.api_calls += 1
                    metrics.retries += int(attempt > 1)
                result = operation()
                self.breaker.success()
                self.successful_operations += 1
                return result
            except Exception as exc:
                last_error = exc
                # Repeating an explicit quota response burns more of the same
                # quota and makes the user's wait longer. Fail once instead.
                if is_rate_limit_error(exc):
                    break
                if attempt < self.policy.max_attempts:
                    self.retries_total += 1
                    self.sleeper(self.policy.base_delay_seconds * (2 ** (attempt - 1)))
        self.breaker.failure()
        self.failed_operations += 1
        assert last_error is not None
        raise last_error

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            attempts_total=self.attempts_total,
            successful_operations=self.successful_operations,
            failed_operations=self.failed_operations,
            retries_total=self.retries_total,
            circuit_state=self.breaker.state,
            consecutive_failures=self.breaker.failures,
            rate_limit_wait_seconds=round(self.rate_limiter.total_wait_seconds, 6),
        )
