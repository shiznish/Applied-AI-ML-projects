"""Tests for app/agent/graph.py's clarification pause/resume.

The round-trip test needs no live Postgres — LangGraph's `interrupt()`
mechanism works the same with any checkpointer, and `build_graph` defaults to
an in-memory one (see graph.py's module docstring) specifically so this stays
testable offline. Only the second test, which proves a *brand-new* compiled
graph instance can resume a session it never saw in-process, needs a real
Postgres-backed checkpointer — that one is skipped automatically without a
reachable database, matching tests/test_db/test_executor.py's convention.
"""
import asyncio
import sys
import uuid

import asyncpg
import pytest
from langgraph.types import Command

from app import llm
from app.agent.graph import build_graph, postgres_checkpointer
from app.agent.interpretation import InterpretationChoice
from app.agent.merge import MergedIntent
from app.agent.state import new_session
from app.config import settings
from app.db.executor import ExecutionResult
from app.schema_extract import Table

TABLE_NAMES = ["orders", "customers", "order_items", "products", "suppliers", "warehouses"]

AMBIGUOUS_SPEND = MergedIntent(
    domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
    needs_clarification=True,
    clarification_reason="ambiguous between revenue and campaign_spend",
    interpretations=[
        llm.Interpretation(description="Customer purchase revenue", example_sql=""),
        llm.Interpretation(description="Marketing campaign budget", example_sql=""),
    ],
)

RESOLVED_TO_REVENUE = MergedIntent(
    domains=["sales_performance"], metrics=["revenue"], dimensions=[],
    filters=[llm.FilterSpec(dimension="customer_country", operator="=", value="UAE")],
    time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
)


def _live_schema():
    return {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
            for n in TABLE_NAMES}


def _fake_generate(question, linked, fewshots):
    # SQL depends on the synthesized question text, which in turn depends on
    # the resolved metric — this is what actually proves the resumed graph
    # carried "revenue" through rather than "campaign_spend" or nothing.
    sql = ("SELECT SUM(amount) AS revenue FROM orders"
           if "revenue" in question.lower()
           else "SELECT SUM(amount) AS campaign_spend FROM campaign_attribution")
    return llm.GeneratedSQL(
        needs_clarification=False, interpretations=[], sql=sql, explanation="test",
        confidence=0.9, tables_used=["orders"], columns_used=[],
    )


async def _fake_executor(sql):
    return ExecutionResult(ok=True, columns=["revenue"], rows=[[100]], row_count=1)


async def _unused_repair(question, linked, fewshots, failed_sql, execution, exec_fn):
    raise AssertionError("repair should not have been called")


def _merge_fake_sequence(results):
    """Returns each queued MergedIntent in order across successive calls —
    needed here because one round trip legitimately calls merge_intent twice
    (once ambiguous, once resolved after the reply), unlike
    tests/test_agent/test_graph.py's single-response fakes."""
    queue = list(results)

    def fake_parse_structured(system, user, model_cls, max_tokens):
        return queue.pop(0)

    return fake_parse_structured


def _build(merge_results):
    return build_graph(
        pool=None,
        live_schema=_live_schema(),
        fewshots=[],
        generate_fn=_fake_generate,
        executor_fn=_fake_executor,
        repair_fn=_unused_repair,
        merge_parse_structured=_merge_fake_sequence(merge_results),
    )


def test_clarification_round_trip_resolves_to_revenue_not_campaign_spend():
    graph = _build([AMBIGUOUS_SPEND, RESOLVED_TO_REVENUE])
    cfg = {"configurable": {"thread_id": "clarify-1"}}

    state = new_session("clarify-1")
    state["user_message"] = "Show spend for UAE"
    paused = asyncio.run(graph.ainvoke(state, config=cfg))

    assert "__interrupt__" in paused
    assert paused["status"] == "awaiting_clarification"
    assert paused["pending_clarification"]["explanation"] == (
        "ambiguous between revenue and campaign_spend"
    )
    assert len(paused["pending_clarification"]["interpretations"]) == 2

    resumed = asyncio.run(graph.ainvoke(Command(resume="Revenue"), config=cfg))

    assert resumed["status"] == "ok"
    assert resumed["metrics"] == ["revenue"]
    assert resumed["pending_clarification"] is None
    assert "revenue" in resumed["pipeline_result"]["sql"].lower()
    assert "campaign_spend" not in resumed["pipeline_result"]["sql"].lower()


def test_resume_prompt_carries_pending_interpretations_for_dismissal_handling():
    """Guards the fix for a real bug: a reply like "proceed anyway" or "ok
    leave it" carries no disambiguating content of its own, so clarify_node's
    merge_intent call must hand the LLM the interpretations IT offered,
    otherwise there's nothing for a dismissal reply to resolve against and
    the turn just re-asks. This checks the wiring (pending_clarification's
    interpretations actually reach the system prompt), not the LLM's
    judgment — that's app/agent/merge.py's own prompt, exercised directly in
    tests/test_agent/test_merge.py."""
    captured_system = {}

    def recording_merge(results):
        queue = list(results)

        def fake_parse_structured(system, user, model_cls, max_tokens):
            if queue:
                captured_system["system"] = system
            return queue.pop(0)

        return fake_parse_structured

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=_fake_generate, executor_fn=_fake_executor, repair_fn=_unused_repair,
        merge_parse_structured=recording_merge([AMBIGUOUS_SPEND, RESOLVED_TO_REVENUE]),
    )
    cfg = {"configurable": {"thread_id": "clarify-dismiss"}}

    state = new_session("clarify-dismiss")
    state["user_message"] = "Show spend for UAE"
    asyncio.run(graph.ainvoke(state, config=cfg))

    asyncio.run(graph.ainvoke(Command(resume="ok leave it"), config=cfg))

    system = captured_system["system"]
    assert "Customer purchase revenue" in system
    assert "Marketing campaign budget" in system
    assert "proceed anyway" in system.lower()


def test_still_ambiguous_after_one_reply_stays_ambiguous_not_looping():
    still_ambiguous = MergedIntent(
        domains=[], metrics=[], dimensions=[], filters=[], time_range=None,
        needs_clarification=True, clarification_reason="still unclear", interpretations=[],
    )
    graph = _build([AMBIGUOUS_SPEND, still_ambiguous])
    cfg = {"configurable": {"thread_id": "clarify-2"}}

    state = new_session("clarify-2")
    state["user_message"] = "Show spend for UAE"
    asyncio.run(graph.ainvoke(state, config=cfg))

    resumed = asyncio.run(graph.ainvoke(Command(resume="not sure"), config=cfg))
    assert resumed["status"] == "ambiguous"
    assert "__interrupt__" not in resumed  # terminal, not a second pause
    assert resumed["pending_clarification"]["explanation"] == "still unclear"


# --------------------------------------------------------------------------
# Pipeline-sourced clarification: the generator itself (not merge_intent)
# offered interpretations because it couldn't satisfy the question exactly
# as asked (app/text2sql/pipeline.py no longer auto-accepts a lone one — see
# its own module comment). Resuming here must EXECUTE the chosen
# interpretation's already-shown SQL directly, never re-derive a question
# from merge_intent — that's what actually avoids the dead-end loop a vague
# "proceed anyway" reply used to cause (see app/agent/interpretation.py).
# --------------------------------------------------------------------------

RESOLVED_SIMPLE = MergedIntent(
    domains=["sales_performance"], metrics=["revenue"], dimensions=[], filters=[],
    time_range=None, needs_clarification=False, clarification_reason="", interpretations=[],
)

ONE_INTERPRETATION_SQL = "SELECT SUM(amount) AS revenue FROM orders"
SECOND_INTERPRETATION_SQL = "SELECT SUM(amount) AS revenue FROM orders WHERE status = 'completed'"


def _fake_generate_single_interpretation(question, linked, fewshots):
    return llm.GeneratedSQL(
        needs_clarification=True,
        interpretations=[llm.Interpretation(
            description="Revenue across all available orders",
            example_sql=ONE_INTERPRETATION_SQL,
        )],
        sql="", explanation="No date column is available to filter by year.",
        confidence=0.9, tables_used=["orders"], columns_used=[],
    )


def _fake_generate_two_interpretations(question, linked, fewshots):
    return llm.GeneratedSQL(
        needs_clarification=True,
        interpretations=[
            llm.Interpretation(description="Revenue across all orders",
                                example_sql=ONE_INTERPRETATION_SQL),
            llm.Interpretation(description="Revenue across completed orders only",
                                example_sql=SECOND_INTERPRETATION_SQL),
        ],
        sql="", explanation="Unclear whether to include non-completed orders.",
        confidence=0.7, tables_used=["orders"], columns_used=["status"],
    )


def _choose_fake(index: int):
    def fake_parse_structured(system, user, model_cls, max_tokens):
        return InterpretationChoice(selected_index=index)
    return fake_parse_structured


def test_single_interpretation_from_generator_pauses_instead_of_auto_executing():
    """Guards the reverted auto-accept: a lone fallback interpretation must
    still pause and ask (status=awaiting_clarification, pending_clarification
    sourced from the pipeline), not run straight through to status=ok."""
    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=_fake_generate_single_interpretation, executor_fn=_fake_executor,
        repair_fn=_unused_repair, merge_parse_structured=_merge_fake_sequence([RESOLVED_SIMPLE]),
    )
    cfg = {"configurable": {"thread_id": "pipeline-clarify-1"}}

    state = new_session("pipeline-clarify-1")
    state["user_message"] = "Total revenue in 2025"
    paused = asyncio.run(graph.ainvoke(state, config=cfg))

    assert "__interrupt__" in paused
    assert paused["status"] == "awaiting_clarification"
    pc = paused["pending_clarification"]
    assert pc["source"] == "pipeline"
    assert len(pc["interpretations"]) == 1
    assert pc["interpretations"][0]["example_sql"] == ONE_INTERPRETATION_SQL


def test_resuming_pipeline_clarification_executes_the_chosen_sql_directly():
    """The reply ("proceed anyway") must not re-run merge_intent/generation —
    it executes the exact SQL already shown for the selected interpretation."""
    executed_sql = {}

    async def recording_executor(sql):
        executed_sql["sql"] = sql
        return ExecutionResult(ok=True, columns=["revenue"], rows=[[42]], row_count=1)

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=_fake_generate_single_interpretation, executor_fn=recording_executor,
        repair_fn=_unused_repair, merge_parse_structured=_merge_fake_sequence([RESOLVED_SIMPLE]),
        interpretation_parse_structured=_choose_fake(1),
    )
    cfg = {"configurable": {"thread_id": "pipeline-clarify-2"}}

    state = new_session("pipeline-clarify-2")
    state["user_message"] = "Total revenue in 2025"
    asyncio.run(graph.ainvoke(state, config=cfg))

    resumed = asyncio.run(graph.ainvoke(Command(resume="proceed anyway"), config=cfg))

    assert resumed["status"] == "ok"
    assert resumed["pending_clarification"] is None
    assert "__interrupt__" not in resumed
    # Guardrails inject a LIMIT clause when one is missing (see
    # app/guardrails/sql_validator.py), so the executed SQL starts with —
    # rather than exactly equals — the interpretation's own example_sql.
    assert executed_sql["sql"].startswith(ONE_INTERPRETATION_SQL)
    assert resumed["pipeline_result"]["sql"].startswith(ONE_INTERPRETATION_SQL)


def test_resuming_pipeline_clarification_honors_a_specific_interpretation_choice():
    """Proves selection actually matters — not hardcoded to the first option
    — by picking index 2 of 2 and checking the SECOND interpretation's SQL
    (not the first) is what gets executed."""
    executed_sql = {}

    async def recording_executor(sql):
        executed_sql["sql"] = sql
        return ExecutionResult(ok=True, columns=["revenue"], rows=[[7]], row_count=1)

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=_fake_generate_two_interpretations, executor_fn=recording_executor,
        repair_fn=_unused_repair, merge_parse_structured=_merge_fake_sequence([RESOLVED_SIMPLE]),
        interpretation_parse_structured=_choose_fake(2),
    )
    cfg = {"configurable": {"thread_id": "pipeline-clarify-3"}}

    state = new_session("pipeline-clarify-3")
    state["user_message"] = "Revenue, completed only or all?"
    asyncio.run(graph.ainvoke(state, config=cfg))

    resumed = asyncio.run(graph.ainvoke(Command(resume="the completed-only one"), config=cfg))

    assert resumed["status"] == "ok"
    assert executed_sql["sql"].startswith(SECOND_INTERPRETATION_SQL)


def test_pipeline_clarification_resume_does_not_loop_even_if_execution_fails():
    """If the exact SQL the user confirmed still fails, that's an honest
    terminal error — not a second pause (which would reopen the dead-end-
    loop risk this whole mechanism exists to avoid)."""
    async def failing_executor(sql):
        return ExecutionResult(ok=False, error="relation does not exist", error_code="42P01")

    graph = build_graph(
        pool=None, live_schema=_live_schema(), fewshots=[],
        generate_fn=_fake_generate_single_interpretation, executor_fn=failing_executor,
        repair_fn=_unused_repair, merge_parse_structured=_merge_fake_sequence([RESOLVED_SIMPLE]),
        interpretation_parse_structured=_choose_fake(1),
    )
    cfg = {"configurable": {"thread_id": "pipeline-clarify-4"}}

    state = new_session("pipeline-clarify-4")
    state["user_message"] = "Total revenue in 2025"
    asyncio.run(graph.ainvoke(state, config=cfg))

    resumed = asyncio.run(graph.ainvoke(Command(resume="proceed anyway"), config=cfg))

    assert resumed["status"] == "error"
    assert "__interrupt__" not in resumed
    assert resumed["pending_clarification"] is None


# --------------------------------------------------------------------------
# Live-Postgres integration test. Skipped automatically when no database is
# reachable — run locally after `docker compose up -d`.
# --------------------------------------------------------------------------


def _postgres_reachable() -> bool:
    async def _ping():
        conn = await asyncpg.connect(settings.database_url, timeout=2)
        await conn.close()

    try:
        asyncio.run(_ping())
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _postgres_reachable(), reason="no reachable PostgreSQL instance for live checkpoint tests"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "psycopg's async mode (AsyncPostgresSaver) cannot run under the "
        "default Windows ProactorEventLoop asyncio/pytest-asyncio create "
        "('Please use a compatible event loop ... "
        "WindowsSelectorEventLoopPolicy'); switching the global loop policy "
        "mid-suite to accommodate one test is a worse trade-off than "
        "skipping here — this test runs for real on Linux/Mac CI."
    ),
)
@pytest.mark.asyncio
async def test_resume_survives_a_fresh_graph_instance_via_postgres_checkpoint():
    """Proves persistence actually works, not just in-process MemorySaver
    state: the pause happens on one compiled graph bound to one Postgres
    checkpointer connection, and the resume happens on a SECOND, independently
    compiled graph bound to a SECOND connection — simulating a process
    restart between the two turns."""
    thread_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}

    async with postgres_checkpointer() as cp1:
        graph1 = build_graph(
            pool=None, live_schema=_live_schema(), fewshots=[],
            generate_fn=_fake_generate, executor_fn=_fake_executor, repair_fn=_unused_repair,
            merge_parse_structured=_merge_fake_sequence([AMBIGUOUS_SPEND]),
            checkpointer=cp1,
        )
        state = new_session(thread_id)
        state["user_message"] = "Show spend for UAE"
        paused = await graph1.ainvoke(state, config=cfg)
        assert "__interrupt__" in paused
        assert paused["status"] == "awaiting_clarification"

    async with postgres_checkpointer() as cp2:
        graph2 = build_graph(
            pool=None, live_schema=_live_schema(), fewshots=[],
            generate_fn=_fake_generate, executor_fn=_fake_executor, repair_fn=_unused_repair,
            merge_parse_structured=_merge_fake_sequence([RESOLVED_TO_REVENUE]),
            checkpointer=cp2,
        )
        resumed = await graph2.ainvoke(Command(resume="Revenue"), config=cfg)
        assert resumed["status"] == "ok"
        assert resumed["metrics"] == ["revenue"]
