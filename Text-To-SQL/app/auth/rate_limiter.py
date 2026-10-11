"""In-process rate limiter for brute-force protection on auth endpoints.

A fixed-window counter keyed by caller identity (client IP + attempted
username, so one abusive client can't lock out a legitimate user's IP, and a
targeted attacker rotating usernames from one IP still gets throttled per
username). Deliberately in-memory rather than Redis-backed: this project runs
with no external cache in local dev or in the test suite,
and a single-process deployment doesn't need distributed state for this — a
horizontally-scaled deployment would need a shared store instead, but that's a
different problem from "the demo has zero throttling today".

The clock is injectable (`time_fn`, defaulting to `time.time`) so tests can
advance time deterministically instead of sleeping for real — the same
dependency-injection idiom used for the LLM/DB/pipeline callables elsewhere in
this codebase (see the `*_fn` parameters in app/text2sql/pipeline.py and
app/evaluation/evaluator.py).
"""
import time
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_s: float = 0.0


class InMemoryRateLimiter:
    """Fixed-window request counter, keyed by an arbitrary string.

    Not thread-safe across real OS threads, but FastAPI's async routes run
    this on a single event loop thread, so a plain dict is sufficient here.
    """

    def __init__(self, max_attempts: int, window_s: float, time_fn=time.time):
        self._max_attempts = max_attempts
        self._window_s = window_s
        self._time_fn = time_fn
        self._attempts: dict[str, list[float]] = {}

    def check(self, key: str) -> RateLimitResult:
        """Records one attempt for `key` and reports whether it's allowed.

        Uses a sliding window (attempts older than `window_s` are dropped
        before counting) rather than a naive fixed-clock-tick bucket, so a
        client can't get a fresh allowance by straddling a bucket boundary.
        """
        now = self._time_fn()
        window_start = now - self._window_s
        attempts = [t for t in self._attempts.get(key, []) if t > window_start]
        if len(attempts) >= self._max_attempts:
            self._attempts[key] = attempts
            retry_after = self._window_s - (now - attempts[0])
            return RateLimitResult(allowed=False, retry_after_s=max(retry_after, 0.0))
        attempts.append(now)
        self._attempts[key] = attempts
        return RateLimitResult(allowed=True)

    def reset(self) -> None:
        """Clears all tracked attempts — used by tests between cases."""
        self._attempts.clear()
