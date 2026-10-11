from __future__ import annotations

from ledger_copilot.middleware.rate_limit import TokenBucket


def test_allows_up_to_capacity_then_trips() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    bucket = TokenBucket(capacity=3, refill_per_second=1.0, clock=clock)

    assert bucket.try_consume() is True
    assert bucket.try_consume() is True
    assert bucket.try_consume() is True
    assert bucket.try_consume() is False


def test_recovers_after_refill_window() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    bucket = TokenBucket(capacity=1, refill_per_second=1.0, clock=clock)

    assert bucket.try_consume() is True
    assert bucket.try_consume() is False

    clock_value = 1.0
    assert bucket.try_consume() is True


def test_refill_never_exceeds_capacity() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    bucket = TokenBucket(capacity=2, refill_per_second=10.0, clock=clock)
    bucket.try_consume()

    clock_value = 100.0
    assert bucket.try_consume() is True
    assert bucket.try_consume() is True
    assert bucket.try_consume() is False
