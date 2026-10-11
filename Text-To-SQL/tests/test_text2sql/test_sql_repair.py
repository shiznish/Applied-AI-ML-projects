"""Offline unit tests for app/text2sql/sql_repair.py. Both the DB execution
and the LLM regeneration call are injected as fakes (executor_fn, generate_fn)
so these tests need neither a live Postgres instance nor an LLM API key.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.db.executor import ExecutionResult
from app.schema_extract import Table
from app.semantic import resolver
from app.text2sql import schema_linker, sql_repair


def _linked_schema():
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
                   for n in names}
    return schema_linker.link("orders by customer", set(names), live_schema, sl)


def _gen(sql, needs_clarification=False):
    return llm.GeneratedSQL(
        needs_clarification=needs_clarification, interpretations=[], sql=sql,
        explanation="repaired", confidence=0.8, tables_used=["orders"], columns_used=[],
    )


def _failed_execution(error="column \"customer_nmae\" does not exist", error_code="42703"):
    return ExecutionResult(ok=False, error=error, error_code=error_code)


def test_repair_succeeds_when_executor_fails_once_then_succeeds():
    calls = {"n": 0}

    async def fake_executor(sql):
        calls["n"] += 1
        if calls["n"] == 1:
            return ExecutionResult(ok=False, error="still broken", error_code="42703")
        return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)

    def fake_generate(question, linked, fewshots, failed_sql, sanitized_error):
        return _gen("SELECT COUNT(*) AS n FROM orders")

    linked = _linked_schema()
    result = asyncio.run(sql_repair.repair(
        "how many orders", linked, [], "SELECT customer_nmae FROM orders",
        _failed_execution(), fake_executor, max_attempts=2, generate_fn=fake_generate,
    ))

    assert result.succeeded is True
    assert result.attempts == 2
    # Guardrails injects a LIMIT on the way through — the regenerated query's
    # intent survives, but the exact text is transformed, same as attempt 1.
    assert result.sql.startswith("SELECT COUNT(*) AS n FROM orders")
    assert calls["n"] == 2


def test_repair_gives_up_after_max_attempts_when_executor_always_fails():
    async def always_fails(sql):
        return ExecutionResult(ok=False, error="nope", error_code="42703")

    def fake_generate(question, linked, fewshots, failed_sql, sanitized_error):
        return _gen("SELECT COUNT(*) FROM orders")

    linked = _linked_schema()
    result = asyncio.run(sql_repair.repair(
        "how many orders", linked, [], "SELECT customer_nmae FROM orders",
        _failed_execution(), always_fails, max_attempts=2, generate_fn=fake_generate,
    ))

    assert result.succeeded is False
    assert result.attempts == 2
    assert result.final_error


def test_repair_never_executes_sql_that_fails_guardrails():
    """A repair attempt must go through guardrails just like the first
    attempt — never skip that check even though the SQL came from the
    regeneration step, not the original generation step."""
    executed = []

    async def fake_executor(sql):
        executed.append(sql)
        return ExecutionResult(ok=True, columns=[], rows=[], row_count=0)

    def fake_generate(question, linked, fewshots, failed_sql, sanitized_error):
        # A regenerated query that tries to smuggle a destructive statement.
        return _gen("DELETE FROM orders")

    linked = _linked_schema()
    result = asyncio.run(sql_repair.repair(
        "how many orders", linked, [], "SELECT customer_nmae FROM orders",
        _failed_execution(), fake_executor, max_attempts=2, generate_fn=fake_generate,
    ))

    assert result.succeeded is False
    assert executed == []  # guardrail-blocked SQL must never reach the executor
    assert "Guardrail violation" in result.final_error


def test_repair_stops_when_model_flags_needs_clarification():
    def fake_generate(question, linked, fewshots, failed_sql, sanitized_error):
        return _gen("", needs_clarification=True)

    async def unused_executor(sql):
        raise AssertionError("executor should not be called")

    linked = _linked_schema()
    result = asyncio.run(sql_repair.repair(
        "how many orders", linked, [], "SELECT customer_nmae FROM orders",
        _failed_execution(), unused_executor, max_attempts=2, generate_fn=fake_generate,
    ))

    assert result.succeeded is False
    assert result.attempts == 2


def test_repair_sanitizes_error_before_it_reaches_generate_fn():
    seen_errors = []

    def fake_generate(question, linked, fewshots, failed_sql, sanitized_error):
        seen_errors.append(sanitized_error)
        return _gen("SELECT 1")

    async def fake_executor(sql):
        return ExecutionResult(ok=True, columns=[], rows=[], row_count=0)

    linked = _linked_schema()
    secret_error = "connection failed: sk-abcdefghij1234567890secret"
    asyncio.run(sql_repair.repair(
        "q", linked, [], "SELECT 1", _failed_execution(error=secret_error), fake_executor,
        max_attempts=2, generate_fn=fake_generate,
    ))

    assert "sk-abcdefghij1234567890secret" not in seen_errors[0]
    assert "REDACTED" in seen_errors[0]
