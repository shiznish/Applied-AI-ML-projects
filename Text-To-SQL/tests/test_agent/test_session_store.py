"""Live-Postgres integration tests. Skipped automatically when no database is
reachable (e.g. this repo's CI-less sandbox authoring environment) — run them
locally after `docker compose up -d && python data/seed.py && psql ... -f
app/db/agent_schema.sql`.

Uses `@pytest_asyncio.fixture` rather than tests/test_db/test_executor.py's
plain `@pytest.fixture` on the equivalent `pool` fixture — the installed
pytest/pytest-asyncio versions in this repo's current sandbox only recognize
an async generator fixture as async in strict mode when it's explicitly
decorated this way; the plain-decorator version raises
`PytestRemovedIn9Warning`/errors at fixture setup instead of either running
or skipping cleanly, independent of whether Postgres is reachable. Not fixed
in test_db/test_executor.py or test_security/test_rls.py here since they
predate this plan and touching them is out of scope — see this plan's
Status log for the finding.
"""
import asyncio
import uuid

import asyncpg
import pytest
import pytest_asyncio

from app.agent.session_store import load_session, save_session
from app.agent.state import new_session
from app.config import settings
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
    not _postgres_reachable(), reason="no reachable PostgreSQL instance for live session-store tests"
)


@pytest_asyncio.fixture()
async def pool():
    p = ConnectionPool()
    await p.open()
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(pool):
    session_id = uuid.uuid4().hex
    state = new_session(session_id)
    state["domain"] = "sales"
    state["metrics"] = ["revenue"]

    await save_session(pool, session_id, "user_a", state)
    loaded = await load_session(pool, session_id, "user_a")

    assert loaded is not None
    assert loaded["domain"] == "sales"
    assert loaded["metrics"] == ["revenue"]


@pytest.mark.asyncio
async def test_load_session_is_isolated_by_user(pool):
    session_id = uuid.uuid4().hex
    state = new_session(session_id)
    state["domain"] = "sales"

    await save_session(pool, session_id, "user_a", state)

    assert await load_session(pool, session_id, "user_b") is None
    assert (await load_session(pool, session_id, "user_a"))["domain"] == "sales"


@pytest.mark.asyncio
async def test_save_session_does_not_overwrite_another_users_session(pool):
    session_id = uuid.uuid4().hex
    original = new_session(session_id)
    original["domain"] = "sales"
    await save_session(pool, session_id, "user_a", original)

    hijack_attempt = new_session(session_id)
    hijack_attempt["domain"] = "marketing"
    await save_session(pool, session_id, "user_b", hijack_attempt)

    untouched = await load_session(pool, session_id, "user_a")
    assert untouched["domain"] == "sales"
    assert await load_session(pool, session_id, "user_b") is None
