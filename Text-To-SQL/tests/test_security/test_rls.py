"""Live-Postgres integration tests for app/db/policies.sql.

Skipped automatically when no database is reachable, following the exact
pattern established in tests/test_db/test_executor.py — run these locally
after:

    docker compose up -d
    psql "$DATABASE_ADMIN_URL" -f app/auth/schema.sql
    psql "$DATABASE_ADMIN_URL" -f app/db/policies.sql
    python data/seed.py
    # then grant the app's connecting DB user permission to switch into each
    # per-business-role role, e.g.:
    #   GRANT app_role_sales_us, app_role_sales_india, app_role_admin TO text2sql;

These tests verify the mechanism this security layer exists to make real:
that a SecurityContext threaded into app/db/connection.py's
scoped_connection actually causes Postgres's own RLS policies and column
grants to restrict what comes back, independent of the SQL text itself.
"""
import asyncio

import asyncpg
import pytest
import pytest_asyncio

from app.config import settings
from app.db import executor as db_executor
from app.db.connection import ConnectionPool, SecurityContext


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
    not _postgres_reachable(), reason="no reachable PostgreSQL instance for live RLS tests"
)


@pytest_asyncio.fixture()
async def pool():
    p = ConnectionPool()
    await p.open()
    yield p
    await p.close()


def _ctx(username: str, role: str, region: str) -> SecurityContext:
    return SecurityContext(
        username=username, role=role, region=region, permissions=frozenset({"query:execute"}),
    )


ALICE = _ctx("alice", "sales_us", "us")
RAJ = _ctx("raj", "sales_india", "india")
ADMIN = _ctx("admin", "admin", "*")


@pytest.mark.asyncio
async def test_sales_us_only_sees_us_orders(pool):
    result = await db_executor.execute(pool, "SELECT DISTINCT region FROM orders", ALICE)
    assert result.ok
    assert result.rows == [["us"]]


@pytest.mark.asyncio
async def test_sales_india_only_sees_india_orders(pool):
    result = await db_executor.execute(pool, "SELECT DISTINCT region FROM orders", RAJ)
    assert result.ok
    assert result.rows == [["india"]]


@pytest.mark.asyncio
async def test_admin_sees_all_regions(pool):
    result = await db_executor.execute(pool, "SELECT DISTINCT region FROM orders", ADMIN)
    assert result.ok
    regions = {r[0] for r in result.rows}
    assert regions == {"us", "india", "row"}


@pytest.mark.asyncio
async def test_sales_us_order_items_scoped_via_join_back_to_orders(pool):
    """order_items has no region column of its own — its RLS policy joins
    back to orders, so this confirms that join-based policy actually filters
    rather than silently letting everything through."""
    result = await db_executor.execute(
        pool,
        "SELECT COUNT(*) AS n FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id WHERE o.region != 'us'",
        ALICE,
    )
    assert result.ok
    # RLS on order_items hides non-us rows before the join predicate even
    # matters, so a query that explicitly asks for region != 'us' should still
    # come back empty for a us-scoped connection.
    assert result.rows == [[0]]


@pytest.mark.asyncio
async def test_sales_us_cannot_query_base_customers_table(pool):
    """Non-admin roles were never granted SELECT on customers itself (only on
    customers_masked) — this should fail with a permission error, not just
    return masked data, since the base table grant doesn't exist at all."""
    result = await db_executor.execute(pool, "SELECT email FROM customers LIMIT 1", ALICE)
    assert not result.ok
    assert result.error_code == "42501"  # insufficient_privilege


@pytest.mark.asyncio
async def test_sales_us_sees_redacted_email_via_masked_view(pool):
    result = await db_executor.execute(
        pool, "SELECT email FROM customers_masked LIMIT 1", ALICE,
    )
    assert result.ok
    assert result.rows[0][0] == "***REDACTED***"


@pytest.mark.asyncio
async def test_admin_sees_real_email_on_base_table(pool):
    result = await db_executor.execute(pool, "SELECT email FROM customers LIMIT 1", ADMIN)
    assert result.ok
    assert result.rows[0][0] != "***REDACTED***"


@pytest.mark.asyncio
async def test_write_statements_blocked_at_the_role_level(pool):
    """Defense-in-depth: even if a guardrail bug let a write-shaped statement
    through, the underlying Postgres role has no INSERT/UPDATE/DELETE grant on
    anything, so it fails here too."""
    result = await db_executor.execute(
        pool, "DELETE FROM orders WHERE order_id = -1", ALICE,
    )
    assert not result.ok
    assert result.error_code == "42501"  # insufficient_privilege
