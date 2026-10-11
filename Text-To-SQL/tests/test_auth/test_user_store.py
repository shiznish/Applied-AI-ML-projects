"""UserStore tests. PostgresUserStore needs a live DB and can't be exercised
here — this file just checks it imports/constructs cleanly and exercises the
UserStore Protocol contract via an in-memory fake, which is the pattern the
rest of the app (routes, other tests) should use for DB-free testing.
"""
from typing import Optional

import pytest

from app.auth.jwt import hash_password
from app.auth.models import User
from app.auth.user_store import PostgresUserStore, UserStore
from app.db.connection import ConnectionPool


class InMemoryUserStore:
    """A minimal fake satisfying the UserStore protocol, for use in tests that
    need a login flow without a live Postgres instance."""

    def __init__(self, users: list[User]):
        self._by_username = {u.username: u for u in users}

    async def get_user_by_username(self, username: str) -> Optional[User]:
        return self._by_username.get(username)


def _demo_store() -> InMemoryUserStore:
    return InMemoryUserStore([
        User(username="alice", hashed_password=hash_password("password123"), role="sales_us", region="us"),
        User(username="raj", hashed_password=hash_password("password123"), role="sales_india", region="india"),
        User(username="admin", hashed_password=hash_password("password123"), role="admin", region="*"),
    ])


@pytest.mark.asyncio
async def test_in_memory_store_finds_seeded_user():
    store: UserStore = _demo_store()
    user = await store.get_user_by_username("alice")
    assert user is not None
    assert user.role == "sales_us"
    assert user.region == "us"
    assert user.permissions == frozenset({"query:execute"})


@pytest.mark.asyncio
async def test_in_memory_store_returns_none_for_unknown_user():
    store: UserStore = _demo_store()
    assert await store.get_user_by_username("nobody") is None


def test_postgres_user_store_constructs_cleanly_without_a_live_db():
    """Constructing PostgresUserStore must not touch the network — only
    calling get_user_by_username() does, and that requires a live DB this
    sandbox doesn't have."""
    store = PostgresUserStore(ConnectionPool())
    assert isinstance(store, PostgresUserStore)
