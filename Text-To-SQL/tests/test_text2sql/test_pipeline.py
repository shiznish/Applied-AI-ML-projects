"""Offline unit tests for app/text2sql/pipeline.py's cost-guard wiring.

Confirms a cost-guard rejection is a distinct pipeline outcome
("complexity_rejected") from both a guardrail block ("blocked") and a
repair-eligible execution error ("error") — the three must not collapse into
each other, since sql_repair.py should never be handed a query that was
rejected for being too expensive rather than wrong.

Everything is dependency-injected (generate_fn, executor_fn, repair_fn) so no
live Postgres instance or LLM API key is needed, matching
tests/test_text2sql/test_sql_repair.py's approach.
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
from app.text2sql.sql_repair import RepairResult

TABLE_NAMES = ["orders", "customers", "order_items", "products", "suppliers", "warehouses"]

# 5 joins — above the default max_join_count of 4 configured in
# app/config.py's GuardrailConfig.
OVER_JOIN_LIMIT_SQL = (
    "SELECT o.id FROM orders o "
    "JOIN customers c ON o.customer_id = c.id "
    "JOIN order_items oi ON oi.order_id = o.id "
    "JOIN products p ON p.id = oi.product_id "
    "JOIN suppliers s ON s.id = p.supplier_id "
    "JOIN warehouses w ON w.id = s.warehouse_id"
)

SIMPLE_SQL = "SELECT COUNT(*) AS n FROM orders"
DELETE_SQL = "DELETE FROM orders"


def _live_schema():
    return {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
            for n in TABLE_NAMES}


def _gen(sql, needs_clarification=False):
    return llm.GeneratedSQL(
        needs_clarification=needs_clarification, interpretations=[], sql=sql,
        explanation="test", confidence=0.9, tables_used=["orders"], columns_used=[],
    )


def _fake_route(question, semantic_layer):
    return llm.RoutedIntent(
        domains=["sales_performance"], metrics=["order_count"], dimensions=[], filters=[],
        needs_clarification=False, clarification_reason="", interpretations=[],
    )


def _run(sql, executor_fn=None, repair_fn=None):
    semantic_layer = resolver.get()

    def fake_generate(question, linked, fewshots):
        return _gen(sql)

    async def unused_executor(sql):
        raise AssertionError("executor should not have been called")

    async def unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        raise AssertionError("repair should not have been called")

    return asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=fake_generate,
        executor_fn=executor_fn or unused_executor,
        repair_fn=repair_fn or unused_repair,
        route_fn=_fake_route,
    ))


def test_static_cost_guard_rejects_before_execution():
    """A query that's too complex on its face (too many joins) never reaches
    the executor at all — the static half of cost_guard runs between
    guardrails and execute."""
    result = _run(OVER_JOIN_LIMIT_SQL)
    assert result.status == "complexity_rejected"
    assert result.violations
    assert any("join" in v.lower() for v in result.violations)


def test_guardrail_block_is_not_reported_as_complexity_rejected():
    """A destructive statement is blocked by sql_validator, not cost_guard —
    the two rejection paths must stay distinguishable."""
    result = _run(DELETE_SQL)
    assert result.status == "blocked"
    assert result.status != "complexity_rejected"


def test_explain_based_rejection_from_executor_surfaces_as_complexity_rejected():
    """When db/executor.py rejects post-EXPLAIN (error_code COMPLEXITY_REJECTED),
    the pipeline must surface that as complexity_rejected and must NOT hand it
    to sql_repair.py — repair is for correctness errors, not cost rejections."""
    async def executor_rejects_for_cost(sql):
        return ExecutionResult(
            ok=False, error="Query rejected by cost guard: too expensive",
            error_code="COMPLEXITY_REJECTED",
        )

    result = _run(SIMPLE_SQL, executor_fn=executor_rejects_for_cost)
    assert result.status == "complexity_rejected"
    assert "too expensive" in result.violations[0]


def test_genuine_execution_error_still_goes_through_repair():
    """A real correctness error (undefined column, sqlstate 42703) must still
    be routed to sql_repair.py, unlike a cost-guard rejection."""
    async def executor_fails_with_undefined_column(sql):
        return ExecutionResult(
            ok=False, error='column "bogus" does not exist', error_code="42703",
        )

    async def fake_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        return RepairResult(sql=failed_sql, attempts=2, succeeded=False,
                             final_error="still broken")

    result = _run(SIMPLE_SQL, executor_fn=executor_fails_with_undefined_column,
                  repair_fn=fake_repair)
    assert result.status == "error"
    assert result.status != "complexity_rejected"


def test_single_interpretation_clarification_still_asks_not_auto_accepts():
    """A lone fallback interpretation is still a real choice (silently
    executing the generator's own best-guess — the prior behavior here —
    means a dropped date filter or other gap gets presented as an ordinary
    success, with the caveat buried in small print). Always returning
    status=ambiguous, one interpretation or many, is what app/agent/
    graph.py's clarify_node now has a real resume path for: a reply that
    confirms it directly EXECUTES that already-shown SQL (see
    app/agent/interpretation.py) rather than re-deriving a question from
    scratch, which is what actually avoids the old dead-end loop — not
    silently skipping the question in the first place."""
    semantic_layer = resolver.get()

    def fake_generate(question, linked, fewshots):
        return llm.GeneratedSQL(
            needs_clarification=True,
            interpretations=[llm.Interpretation(
                description="Revenue across all available orders",
                example_sql=SIMPLE_SQL,
            )],
            sql="", explanation="No date column is available to filter by year.",
            confidence=0.9, tables_used=["orders"], columns_used=[],
        )

    async def unused_executor(sql):
        raise AssertionError("executor should not have been called")

    async def unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        raise AssertionError("repair should not have been called")

    result = asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=fake_generate,
        executor_fn=unused_executor, repair_fn=unused_repair, route_fn=_fake_route,
    ))

    assert result.status == "ambiguous"
    assert len(result.ambiguity.interpretations) == 1
    assert result.ambiguity.interpretations[0]["example_sql"] == SIMPLE_SQL
    assert "No date column" in result.ambiguity.explanation


def test_multiple_interpretations_still_asks_for_clarification():
    """A genuine multi-way choice (2+ interpretations) also asks."""
    semantic_layer = resolver.get()

    def fake_generate(question, linked, fewshots):
        return llm.GeneratedSQL(
            needs_clarification=True,
            interpretations=[
                llm.Interpretation(description="Customer purchase revenue", example_sql=""),
                llm.Interpretation(description="Marketing campaign spend", example_sql=""),
            ],
            sql="", explanation="ambiguous between revenue and spend",
            confidence=0.5, tables_used=[], columns_used=[],
        )

    async def unused_executor(sql):
        raise AssertionError("executor should not have been called")

    async def unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        raise AssertionError("repair should not have been called")

    result = asyncio.run(pipeline.run(
        "show spend", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=fake_generate,
        executor_fn=unused_executor, repair_fn=unused_repair, route_fn=_fake_route,
    ))

    assert result.status == "ambiguous"
    assert len(result.ambiguity.interpretations) == 2


def _fake_route_customer_country(question, semantic_layer):
    return llm.RoutedIntent(
        domains=["customer_analytics"], metrics=[], dimensions=["customer_country"], filters=[],
        needs_clarification=False, clarification_reason="", interpretations=[],
    )


def test_non_admin_security_context_generates_sql_against_masked_customers_view():
    """Non-admin Postgres roles (app_role_sales_us / app_role_sales_india)
    have no grant on the base `customers` table at all — only on the
    column-masked `customers_masked` view (app/db/policies.sql). The table
    identity fed to SQL generation for a non-admin caller must reflect that,
    or every customer-related question generates SQL that fails at
    execution with a permission error (this is the bug app/db/masking.py
    and its callers in schema_linker.py/value_resolver.py fix)."""
    captured = {}

    def fake_generate(question, linked, fewshots):
        table = next(iter(linked.tables))
        captured["table"] = table
        return _gen(f"SELECT country FROM {table}")

    async def fake_executor(sql):
        return ExecutionResult(ok=True, columns=["country"], rows=[("India",)], row_count=1)

    ctx = SecurityContext(username="raj", role="sales_india", region="india", permissions=frozenset())
    result = asyncio.run(pipeline.run(
        "which countries do our customers come from", _live_schema(), pool=None, fewshots=[],
        semantic_layer=resolver.get(), generate_fn=fake_generate,
        executor_fn=fake_executor,
        route_fn=_fake_route_customer_country, security_context=ctx,
    ))

    assert result.status == "ok"
    assert captured["table"] == "customers_masked"
    assert "customers_masked" in result.sql


def test_admin_security_context_generates_sql_against_real_customers_table():
    captured = {}

    def fake_generate(question, linked, fewshots):
        table = next(iter(linked.tables))
        captured["table"] = table
        return _gen(f"SELECT country FROM {table}")

    async def fake_executor(sql):
        return ExecutionResult(ok=True, columns=["country"], rows=[("India",)], row_count=1)

    ctx = SecurityContext(username="admin", role="admin", region="*", permissions=frozenset({"admin:*"}))
    result = asyncio.run(pipeline.run(
        "which countries do our customers come from", _live_schema(), pool=None, fewshots=[],
        semantic_layer=resolver.get(), generate_fn=fake_generate,
        executor_fn=fake_executor,
        route_fn=_fake_route_customer_country, security_context=ctx,
    ))

    assert result.status == "ok"
    assert captured["table"] == "customers"
    assert "customers_masked" not in result.sql


def test_repair_stopping_early_for_cost_also_surfaces_as_complexity_rejected():
    """If sql_repair.py itself stops early because a regenerated query tripped
    the EXPLAIN-based cost guard (RepairResult.error_code), the pipeline must
    still report complexity_rejected rather than a generic "error"."""
    async def executor_fails_then_would_be_asked_to_repair(sql):
        return ExecutionResult(
            ok=False, error='column "bogus" does not exist', error_code="42703",
        )

    async def fake_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        return RepairResult(
            sql=failed_sql, attempts=1, succeeded=False,
            final_error="Query rejected by cost guard: too expensive",
            error_code="COMPLEXITY_REJECTED",
        )

    result = _run(SIMPLE_SQL, executor_fn=executor_fails_then_would_be_asked_to_repair,
                  repair_fn=fake_repair)
    assert result.status == "complexity_rejected"
