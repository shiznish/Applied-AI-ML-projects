from __future__ import annotations

from ledger_copilot.domain.idempotency import IdempotencyStore


def test_first_call_creates() -> None:
    store: IdempotencyStore[str] = IdempotencyStore(window_seconds=60)

    result, created = store.get_or_create("key-1", lambda: "result-1")

    assert result == "result-1"
    assert created is True


def test_duplicate_key_returns_cached() -> None:
    store: IdempotencyStore[str] = IdempotencyStore(window_seconds=60)
    calls = 0

    def factory() -> str:
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    first_result, first_created = store.get_or_create("key-1", factory)
    second_result, second_created = store.get_or_create("key-1", factory)

    assert first_result == "result-1"
    assert second_result == "result-1"
    assert first_created is True
    assert second_created is False
    assert calls == 1


def test_different_key_creates_again() -> None:
    store: IdempotencyStore[str] = IdempotencyStore(window_seconds=60)

    first_result, first_created = store.get_or_create("key-1", lambda: "result-1")
    second_result, second_created = store.get_or_create("key-2", lambda: "result-2")

    assert first_result == "result-1"
    assert second_result == "result-2"
    assert first_created is True
    assert second_created is True


def test_key_outside_window_creates_again() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    store: IdempotencyStore[str] = IdempotencyStore(window_seconds=10, clock=clock)
    calls = 0

    def factory() -> str:
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    store.get_or_create("key-1", factory)
    clock_value = 11.0
    result, created = store.get_or_create("key-1", factory)

    assert result == "result-2"
    assert created is True
    assert calls == 2
