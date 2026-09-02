"""Retry helpers for the embedding provider's rate limits.

The Gemini free tier limits embedding calls per minute as well as per day, and
the per-minute ceiling is the one that bites: a burst of 50 chunks trips it
immediately. Retrying with backoff turns that into a pause instead of a crash
halfway through a long ingest.

The daily cap is not retryable on any useful timescale, so it is detected
separately and raised straight away rather than burning the retry budget.
"""

import random
import re
import time

# Substrings that identify a retryable rate-limit error across the providers in
# use. Matching on text is crude, but the SDKs wrap the underlying 429 in
# several different exception types.
RATE_LIMIT_MARKERS = (
    "resource_exhausted",
    "429",
    "rate limit",
    "quota exceeded",
    "too many requests",
)

# A per-day quota is not worth waiting out inside a run.
DAILY_QUOTA_MARKERS = (
    "per_day",
    "perday",
    "requests_per_day",
    "daily limit",
)

# Providers often say how long to wait ("Please retry in 42.7s").
RETRY_DELAY_PATTERNS = (
    re.compile(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.I),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I),
)

DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_BASE_DELAY = 15.0
MAX_DELAY = 120.0


class DailyQuotaExhausted(RuntimeError):
    """The provider's per-day allowance is gone; retrying will not help today."""


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def is_daily_quota(exc: Exception) -> bool:
    text = str(exc).lower().replace("-", "_")
    return any(marker in text for marker in DAILY_QUOTA_MARKERS)


def suggested_delay(exc: Exception) -> float | None:
    """The provider's own retry hint, if it gave one."""
    text = str(exc)
    for pattern in RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def call_with_backoff(
    fn,
    *args,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    on_retry=None,
    **kwargs,
):
    """Call fn, retrying rate limits with exponential backoff and jitter.

    Honours the provider's own retry hint when present. Anything that is not a
    rate limit -- and a per-day exhaustion, which no backoff can fix -- is
    raised immediately rather than retried.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if is_daily_quota(e):
                raise DailyQuotaExhausted(str(e)[:300]) from e
            if not is_rate_limit(e) or attempt == max_attempts:
                raise

            hinted = suggested_delay(e)
            backoff = min(base_delay * (2 ** (attempt - 1)), MAX_DELAY)
            # Jitter keeps parallel runs from retrying in lockstep.
            delay = max(hinted or 0.0, backoff) + random.uniform(0, 3)

            if on_retry:
                on_retry(attempt, delay, e)
            time.sleep(delay)
