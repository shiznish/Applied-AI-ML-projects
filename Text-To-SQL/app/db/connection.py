"""Async PostgreSQL connection pool + per-request security-context wiring.

Two layers of defense, same shape as the original DuckDB executor:
  1. Guardrail middleware (app/guardrails/sql_validator.py) blocks destructive SQL
     before it gets here.
  2. Every connection handed out for query execution is put into a request's role
     and region via `SET LOCAL` inside a transaction, so PostgreSQL's own
     row-level-security policies and column grants apply regardless of what SQL
     text was generated — the LLM never has to know tenancy exists, and a
     guardrail bug still can't cross a region boundary. See app/db/policies.sql
     for the policy definitions this depends on.
"""
import asyncpg

from ..config import settings


class ConnectionPool:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None

    async def open(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    def scoped_connection(self, security_context: "SecurityContext | None" = None):
        """Async context manager yielding a connection inside a transaction whose
        session has been scoped to the caller's Postgres role and region (if a
        SecurityContext is given). Statement timeout is also set here as a blunt,
        always-on backstop independent of the cost-guard heuristic."""
        return _ScopedConnection(self._pool, security_context)


class _ScopedConnection:
    def __init__(self, pool: asyncpg.Pool, security_context: "SecurityContext | None"):
        self._pool = pool
        self._ctx = security_context
        self._conn: asyncpg.Connection | None = None
        self._tx = None

    async def __aenter__(self) -> asyncpg.Connection:
        self._conn = await self._pool.acquire()
        self._tx = self._conn.transaction()
        await self._tx.start()
        await self._conn.execute(
            f"SET LOCAL statement_timeout = '{int(settings.guardrails.query_timeout_s * 1000)}'"
        )
        if self._ctx is not None:
            # SET LOCAL doesn't accept bind parameters for the value in all
            # positions; region is validated against a fixed enum server-side by
            # the RLS policy itself, but we still quote defensively here since
            # this string is interpolated into SQL text.
            await self._conn.execute(
                "SELECT set_config('app.current_region', $1, true)", self._ctx.region
            )
            await self._conn.execute(f'SET LOCAL ROLE "{self._ctx.postgres_role}"')
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                await self._tx.commit()
            else:
                await self._tx.rollback()
        finally:
            await self._pool.release(self._conn)


class SecurityContext:
    """Resolved identity for one request — produced by app/middleware/auth.py from
    the decoded JWT, consumed here to scope the DB session and later by
    app/audit/audit_log.py to attribute the query."""

    def __init__(self, username: str, role: str, region: str, permissions: frozenset[str]):
        self.username = username
        self.role = role
        self.region = region
        self.permissions = permissions

    @property
    def postgres_role(self) -> str:
        return f"app_role_{self.role}"

    def has_permission(self, perm: str) -> bool:
        if "admin:*" in self.permissions:
            return True
        return perm in self.permissions


pool = ConnectionPool()
