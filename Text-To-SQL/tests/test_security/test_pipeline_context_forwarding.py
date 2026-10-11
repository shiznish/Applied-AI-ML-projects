"""Offline test (no DB, no LLM) that app/text2sql/pipeline.run() actually
forwards a given SecurityContext into its default executor call, rather than
that being an untested assumption.

The whole point of this security design is that a SecurityContext resolved
from a JWT actually reaches query execution — app/db/executor.py and
app/db/connection.py already have the machinery to *use* one, and the
live-Postgres RLS behavior itself is covered by
tests/test_security/test_rls.py (skipped without a real database). This test
covers the one link in the chain that doesn't need a
database at all: that pipeline.run()'s default `exec_fn` is built with the
`security_context` argument closed over, so it's actually passed down to
app.db.executor.execute rather than silently defaulting to None.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.db.connection import SecurityContext
from app.db.executor import ExecutionResult
from app.schema_extract import Table
from app.semantic import resolver
from app.text2sql import pipeline

TABLE_NAMES = ["orders", "customers"]


def _live_schema():
    return {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
            for n in TABLE_NAMES}


def _gen(sql):
    return llm.GeneratedSQL(
        needs_clarification=False, interpretations=[], sql=sql,
        explanation="test", confidence=0.9, tables_used=["orders"], columns_used=[],
    )


def test_run_forwards_security_context_into_default_exec_fn(monkeypatch):
    """Don't inject executor_fn (that would bypass the exact code path we're
    trying to verify) — instead monkeypatch app.db.executor.execute, the
    function pipeline.py's default exec_fn lambda calls, and assert the
    security_context it receives is the one passed into run()."""
    expected_ctx = SecurityContext(
        username="alice", role="sales_us", region="us", permissions=frozenset({"query:execute"}),
    )
    seen = {}

    async def fake_db_execute(pool, sql, security_context=None, max_rows=1000):
        seen["security_context"] = security_context
        return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)

    monkeypatch.setattr(pipeline, "db_execute", fake_db_execute)

    def fake_generate(question, linked, fewshots):
        return _gen("SELECT COUNT(*) AS n FROM orders")

    result = asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=resolver.get(), generate_fn=fake_generate,
        security_context=expected_ctx,
    ))

    assert result.status == "ok"
    assert seen["security_context"] is expected_ctx


def test_run_defaults_security_context_to_none(monkeypatch):
    """No security_context passed in -> the default exec_fn must not
    silently invent one; it should forward None, same as before this phase
    for callers that don't (yet) authenticate."""
    seen = {}

    async def fake_db_execute(pool, sql, security_context=None, max_rows=1000):
        seen["security_context"] = security_context
        return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)

    monkeypatch.setattr(pipeline, "db_execute", fake_db_execute)

    def fake_generate(question, linked, fewshots):
        return _gen("SELECT COUNT(*) AS n FROM orders")

    result = asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=resolver.get(), generate_fn=fake_generate,
    ))

    assert result.status == "ok"
    assert seen["security_context"] is None
