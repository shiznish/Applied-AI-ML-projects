"""Pluggable user lookup.

`UserStore` is a Protocol so the login route (and any future consumer) can
depend on "something that can look up a user by username" without caring
whether that's Postgres or an in-memory fake — the same injection pattern
used throughout app/text2sql/ (e.g. schema_retriever taking a resolver, not
hardcoding layer.yaml's path). `PostgresUserStore` is the real, default
implementation; tests supply their own in-memory fake (see
tests/test_auth/test_user_store.py) implementing the same Protocol.
"""
from typing import Optional, Protocol

from ..db.connection import ConnectionPool
from .models import User


class UserStore(Protocol):
    async def get_user_by_username(self, username: str) -> Optional[User]:
        ...


class PostgresUserStore:
    """Looks up users from the `users`/`roles` tables created by
    app/auth/schema.sql, via the shared asyncpg pool (app.db.connection.pool),
    the same way app/db/executor.py and app/schema_extract.py use it.
    """

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    async def get_user_by_username(self, username: str) -> Optional[User]:
        async with self._pool.scoped_connection() as conn:
            row = await conn.fetchrow(
                "SELECT username, hashed_password, role, region FROM users WHERE username = $1",
                username,
            )
        if row is None:
            return None
        return User(
            username=row["username"],
            hashed_password=row["hashed_password"],
            role=row["role"],
            region=row["region"],
        )
