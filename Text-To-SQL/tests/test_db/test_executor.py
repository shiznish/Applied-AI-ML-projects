"""Live-Postgres integration tests. Skipped automatically when no database is
reachable (e.g. this repo's CI-less sandbox authoring environment) — run them
locally after `docker compose up -d && python data/seed.py`.
"""
import asyncio

import asyncpg
import pytest
import pytest_asyncio

from app.config import settings
from app.db import executor as db_executor
from app.db.connection import ConnectionPool


def _postgres_reachable() -> bool:
    async def _ping():
        conn = await asyncpg.connect(settings.database_url, timeout=2)
        await conn.close()

    try:
        asyncio.run(_ping())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason="no reachable PostgreSQL instance for live executor tests"
)


@pytest_asyncio.fixture()
async def pool():
    p = ConnectionPool()
    await p.open()
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_executor_runs_select(pool):
    result = await db_executor.execute(pool, "SELECT COUNT(*) AS n FROM customers")
    assert result.ok
    assert result.columns == ["n"]
    assert result.explain_plan


@pytest.mark.asyncio
async def test_executor_reports_postgres_errors_with_sqlstate(pool):
    """sql_repair.py keys its retry decision off error_code, so a schema error
    must come back structured, not just as a string."""
    result = await db_executor.execute(pool, "SELECT customer_nmae FROM customers")
    assert not result.ok
    assert result.error_code  # e.g. 42703 undefined_column


# Note: a role-scoped write-block test (the Layer-2 defense equivalent of the
# original DuckDB read_only=True test) lives in tests/test_security/ instead,
# since it depends on the per-role Postgres roles app/db/policies.sql
# creates — without those, every connection uses the pool's single
# admin-ish role and a write block here would be testing nothing real.
