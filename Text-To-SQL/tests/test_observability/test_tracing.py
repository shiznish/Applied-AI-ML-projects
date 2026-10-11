"""Offline unit tests for app/observability/tracing.py and the custom spans
wired into app/text2sql/pipeline.py.

Uses OpenTelemetry SDK's InMemorySpanExporter — the standard way to test OTel
instrumentation without a live Jaeger/collector or any network access.
configure_tracing() is called exactly once at
module scope with that in-memory exporter: OTel's global TracerProvider can
only be set once per process (a second call is a silent no-op with a logged
warning), so every test in this module shares one provider/exporter and each
test clears the exporter before running and force-flushes afterwards instead
of reconfiguring tracing per test.

Everything here is dependency-injected (generate_fn, executor_fn, repair_fn)
so no live Postgres instance or LLM API key is needed, matching
tests/test_text2sql/test_pipeline.py's approach.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import llm
from app.db.connection import SecurityContext
from app.db.executor import ExecutionResult
from app.observability import tracing
from app.schema_extract import Table
from app.semantic import resolver
from app.text2sql import pipeline

_EXPORTER = InMemorySpanExporter()
_PROVIDER = tracing.configure_tracing(app=None, span_exporter=_EXPORTER)

TABLE_NAMES = ["orders", "customers", "order_items", "products", "suppliers", "warehouses"]

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


def _spans_by_name():
    """Flushes any pending spans and returns {span_name: [span, ...]}."""
    _PROVIDER.force_flush()
    spans = _EXPORTER.get_finished_spans()
    by_name: dict[str, list] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    return by_name


def _fake_route(question, semantic_layer):
    return llm.RoutedIntent(
        domains=["sales_performance"], metrics=["order_count"], dimensions=[], filters=[],
        needs_clarification=False, clarification_reason="", interpretations=[],
    )


def _run_pipeline(sql, executor_fn=None, security_context=None):
    semantic_layer = resolver.get()

    def fake_generate(question, linked, fewshots):
        return _gen(sql)

    async def default_executor(sql):
        return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)

    async def unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        raise AssertionError("repair should not have been called")

    return asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=fake_generate,
        executor_fn=executor_fn or default_executor,
        repair_fn=unused_repair,
        route_fn=_fake_route,
        security_context=security_context,
    ))


def setup_function(_fn):
    _EXPORTER.clear()


def test_ok_run_produces_expected_span_tree_with_guardrail_not_blocked():
    result = _run_pipeline(SIMPLE_SQL)
    assert result.status == "ok"

    by_name = _spans_by_name()
    assert "text2sql.pipeline.run" in by_name
    assert "text2sql.semantic_routing" in by_name
    assert "text2sql.schema_resolution" in by_name
    assert "text2sql.guardrail_check" in by_name
    assert "text2sql.cost_guard" in by_name
    assert "text2sql.execute" in by_name

    routing_span = by_name["text2sql.semantic_routing"][0]
    assert routing_span.attributes["text2sql.routing.needs_clarification"] is False

    resolution_span = by_name["text2sql.schema_resolution"][0]
    assert resolution_span.attributes["text2sql.resolution.ambiguous"] is False

    guardrail_span = by_name["text2sql.guardrail_check"][0]
    assert guardrail_span.attributes["text2sql.guardrail.blocked"] is False
    assert guardrail_span.attributes["text2sql.guardrail.violations"] == 0

    cost_guard_span = by_name["text2sql.cost_guard"][0]
    assert cost_guard_span.attributes["text2sql.cost_guard.rejected"] is False


def test_blocked_run_marks_guardrail_span_blocked_with_violation_count():
    result = _run_pipeline(DELETE_SQL)
    assert result.status == "blocked"

    by_name = _spans_by_name()
    guardrail_span = by_name["text2sql.guardrail_check"][0]
    assert guardrail_span.attributes["text2sql.guardrail.blocked"] is True
    assert guardrail_span.attributes["text2sql.guardrail.violations"] >= 1

    # A blocked query never reaches the cost guard or executor.
    assert "text2sql.cost_guard" not in by_name
    assert "text2sql.execute" not in by_name


def test_repair_attempts_attribute_recorded_on_repair_span():
    from app.text2sql.sql_repair import RepairResult

    async def executor_fails(sql):
        return ExecutionResult(ok=False, error='column "bogus" does not exist', error_code="42703")

    async def fake_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        return RepairResult(sql=failed_sql, attempts=2, succeeded=False, final_error="still broken")

    semantic_layer = resolver.get()

    def fake_generate(question, linked, fewshots):
        return _gen(SIMPLE_SQL)

    result = asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=fake_generate,
        executor_fn=executor_fails, repair_fn=fake_repair, route_fn=_fake_route,
    ))
    assert result.status == "error"

    by_name = _spans_by_name()
    repair_span = by_name["text2sql.repair"][0]
    assert repair_span.attributes["text2sql.repair.attempts"] == 2


def test_rls_region_attribute_set_from_security_context():
    ctx = SecurityContext(username="alice", role="analyst", region="uae", permissions=frozenset())
    result = _run_pipeline(SIMPLE_SQL, security_context=ctx)
    assert result.status == "ok"

    by_name = _spans_by_name()
    pipeline_span = by_name["text2sql.pipeline.run"][0]
    assert pipeline_span.attributes["text2sql.rls.region"] == "uae"


def test_rls_region_attribute_absent_without_security_context():
    result = _run_pipeline(SIMPLE_SQL)
    assert result.status == "ok"

    by_name = _spans_by_name()
    pipeline_span = by_name["text2sql.pipeline.run"][0]
    assert "text2sql.rls.region" not in pipeline_span.attributes


def test_correlation_id_attribute_populated_on_every_span():
    token = tracing.set_correlation_id("corr-abc-123")
    try:
        result = _run_pipeline(SIMPLE_SQL)
    finally:
        tracing.reset_correlation_id(token)
    assert result.status == "ok"

    by_name = _spans_by_name()
    for name, spans in by_name.items():
        for span in spans:
            assert span.attributes.get("text2sql.correlation_id") == "corr-abc-123", (
                f"span {name} missing/incorrect correlation id"
            )


def test_correlation_id_absent_when_not_set():
    # No set_correlation_id call in this test — contextvar default is "".
    result = _run_pipeline(SIMPLE_SQL)
    assert result.status == "ok"

    by_name = _spans_by_name()
    pipeline_span = by_name["text2sql.pipeline.run"][0]
    assert "text2sql.correlation_id" not in pipeline_span.attributes


def test_generation_span_wraps_a_gen_ai_generate_content_span_for_real_generator():
    """Uses the real app/text2sql/generator.generate_sql (not a fake) with a
    fake parse_structured injected underneath it, so the gen_ai.generate_content
    span created inside generator.py is exercised end-to-end."""
    from app.text2sql import generator, schema_linker

    def fake_parse_structured(system, user, model_cls, max_tokens):
        return _gen(SIMPLE_SQL)

    semantic_layer = resolver.get()

    def real_generate_fn(question, linked, fewshots):
        return generator.generate_sql(question, linked, fewshots, parse_structured=fake_parse_structured)

    async def default_executor(sql):
        return ExecutionResult(ok=True, columns=["n"], rows=[[1]], row_count=1)

    async def unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
        raise AssertionError("repair should not have been called")

    result = asyncio.run(pipeline.run(
        "how many orders", _live_schema(), pool=None, fewshots=[],
        semantic_layer=semantic_layer, generate_fn=real_generate_fn,
        executor_fn=default_executor, repair_fn=unused_repair, route_fn=_fake_route,
    ))
    assert result.status == "ok"

    by_name = _spans_by_name()
    assert "gen_ai.generate_content" in by_name
    gen_span = by_name["gen_ai.generate_content"][0]
    assert gen_span.attributes["gen_ai.system"]
    assert gen_span.attributes["gen_ai.request.model"] == __import__("app.config", fromlist=["settings"]).settings.model
    assert gen_span.attributes["gen_ai.operation.name"] == "generate_content"
    # It's a child of text2sql.generation (nested, not a sibling).
    generation_span = by_name["text2sql.generation"][0]
    assert gen_span.parent is not None
    assert gen_span.parent.span_id == generation_span.context.span_id
