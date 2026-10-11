from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    result: T
    created_at: float


class IdempotencyStore(Generic[T]):
    def __init__(self, window_seconds: float, clock: Callable[[], float] | None = None) -> None:
        self._window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._entries: dict[str, _Entry[T]] = {}

    def get_or_create(self, key: str, factory: Callable[[], T]) -> tuple[T, bool]:
        entry = self._entries.get(key)
        if entry is not None and self._clock() - entry.created_at <= self._window_seconds:
            return entry.result, False

        result = factory()
        self._entries[key] = _Entry(result=result, created_at=self._clock())
        return result, True
