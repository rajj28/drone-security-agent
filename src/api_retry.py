"""
api_retry.py — Rate limiting and retries for Gemini / Hugging Face / OpenAI calls.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_quota_exhausted = False
_rate_limit_hits = 0


def quota_exhausted() -> bool:
    return _quota_exhausted


def rate_limit_hits() -> int:
    return _rate_limit_hits


def mark_quota_exhausted() -> None:
    global _quota_exhausted
    _quota_exhausted = True


def reset_api_run_state() -> None:
    """Call at the start of each pipeline run."""
    global _quota_exhausted, _limiter, _rate_limit_hits
    _quota_exhausted = False
    _rate_limit_hits = 0
    _limiter = None


def is_rate_or_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return True
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return True
    if "resource_exhausted" in text:
        return True
    if "409" in text:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429, 409, 503):
        return True
    return False


def is_quota_exhausted_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "retry in" in text or "please retry" in text or "429" in text:
        return False
    return "insufficient_quota" in text or "exceeded your current quota" in text


def _retry_delay(exc: BaseException, attempt: int, base_delay: float) -> float:
    """Prefer API-suggested delay when present."""
    retry_after = getattr(exc, "retry_after_sec", None)
    if retry_after is not None and retry_after > 0:
        return min(retry_after + 0.5, 60.0)
    return min(base_delay * (2**attempt), 45.0)


class ApiRateLimiter:
    """Minimum spacing between outbound API calls (process-wide)."""

    def __init__(self, min_interval_sec: float):
        self.min_interval = max(0.0, min_interval_sec)
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


_limiter: Optional[ApiRateLimiter] = None
# When set (e.g. scaled by number of API keys), overrides settings.API_MIN_INTERVAL_SEC.
_effective_interval: Optional[float] = None


def configure_min_interval(per_key_interval: float, num_keys: int = 1) -> None:
    """Configure global call spacing, scaled down by the number of API keys.

    With N keys rotating round-robin, the global throughput can be N× higher while each
    individual key still respects its own per-key minimum interval.
    """
    global _effective_interval, _limiter
    _effective_interval = max(0.0, float(per_key_interval) / max(1, int(num_keys)))
    _limiter = ApiRateLimiter(_effective_interval)


def get_rate_limiter() -> ApiRateLimiter:
    global _limiter
    if _limiter is None:
        if _effective_interval is not None:
            _limiter = ApiRateLimiter(_effective_interval)
        else:
            try:
                from src.config import settings

                interval = float(settings.API_MIN_INTERVAL_SEC)
            except Exception:
                interval = float(os.environ.get("API_MIN_INTERVAL_SEC", "12"))
            _limiter = ApiRateLimiter(interval)
    return _limiter


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    label: str = "api",
) -> T:
    """Retry with backoff on 429/503; honor Retry-After when parsed; fail fast on quota."""
    global _rate_limit_hits
    if quota_exhausted():
        raise RuntimeError(
            "API quota exhausted for this run — use --offline-vision or add billing"
        )

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        get_rate_limiter().wait()
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if is_quota_exhausted_error(exc):
                mark_quota_exhausted()
                raise
            if not is_rate_or_quota_error(exc) or attempt >= max_retries:
                raise
            _rate_limit_hits += 1
            delay = _retry_delay(exc, attempt, base_delay)
            print(
                f"[API] {label} rate limited — retry {attempt + 1}/{max_retries} in {delay:.1f}s"
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
