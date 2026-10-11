"""LangGraph wiring for the governed agentic layer.

Six nodes (each now its own module under app/agent/nodes/ — see that
package's own docstring): `summarize_node` (app/agent/memory.py — a cheap
budget check ahead of everything else, only calling an LLM when the
transcript has actually outgrown the configured token budget),
`classify_message_node` (routes a fresh turn to a data question or a meta
one), `respond_conversationally_node`, `merge_intent_node` (the one new
per-turn LLM call that resolves catalog slots — see app/agent/merge.py),
`clarify_node` (pauses on a real ambiguity via LangGraph's `interrupt()`),
and `run_pipeline_node`, which hands the merged, resolved slot state to the
EXISTING pipeline (app/text2sql/pipeline.py) completely unmodified, via its
one injectable reuse seam, `route_fn`. Everything pipeline.run() already
does — deterministic resolution (app/semantic/resolver.py), value grounding,
schema linking, generation, guardrails, cost guard, execute, repair-on-error
— runs exactly as it does for the non-agentic `/v1/execute`-style path; this
graph only ever decides WHAT QUESTION to ask the pipeline, never how the
pipeline answers it.

This module itself holds only the compiled graph's wiring (`build_graph`,
the `StateGraph` setup, the edges) plus a handful of small helpers shared by
more than one node (`build_routed_intent_from_state`, `_pipeline_result_to_dict`,
`_answer_summary`, `_progress_fn`, `_merged_to_state_update`,
`_clarification_update`) and `postgres_checkpointer`. These helpers are
defined here, ABOVE the node-module imports further down this file, so the
node modules (which import them back from here) see fully-initialized
names rather than a circular, half-built module.

Why ambiguity needs its own node rather than `merge_intent_node` calling
`interrupt()` directly: a LangGraph node that pauses via `interrupt()`
re-executes from its own top on resume (everything before the interrupt call
re-runs), so if `merge_intent_node` itself paused, resuming would re-merge
the ORIGINAL ambiguous message again, not the clarifying reply — the
reply only exists as the value `interrupt()` returns on resume. Splitting
the pause into its own node means `merge_intent_node` commits its
`awaiting_clarification` state update and exits normally; `clarify_node` is
the only thing that re-runs on resume, and it re-merges against the reply
(`interrupt()`'s return value) rather than the stale original message.
"""
from functools import partial

from .. import llm
from ..config import settings
from ..text2sql import pipeline as text2sql_pipeline

from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from .memory import summarize_node
from .state import AgentState


@asynccontextmanager
async def postgres_checkpointer(conn_string: str | None = None):
    """Real, durable checkpointer for production use (app/main.py's
    `lifespan()` wires this into app startup, the same lifespan-scoped-
    resource pattern as app/db/connection.py's ConnectionPool.open()/close())
    — backed by LangGraph's OWN checkpoint tables (checkpoints/
    checkpoint_writes/checkpoint_blobs, created idempotently by `.setup()`
    below), which are deliberately NOT app/db/agent_schema.sql's
    `agent_sessions` table. The two serve different jobs: `agent_sessions` is
    this project's own simple, directly-queryable session row; LangGraph's
    checkpoint tables are its internal replay log, needed to resume a paused
    `interrupt()` exactly where it left off (including values captured
    mid-node) — reimplementing that on top of a plain JSONB column isn't a
    trade this project needs to make when the library already solves it.
    Both tables exist in the same database; neither replaces the other.
    `build_graph`'s default checkpointer stays in-memory (see below) so the
    rest of this module — and its entire offline test suite — keeps working
    with no live Postgres; only a caller that actually needs cross-process
    resume (the real API layer) opens this."""
    async with AsyncPostgresSaver.from_conn_string(conn_string or settings.database_url) as saver:
        await saver.setup()
        yield saver


def build_routed_intent_from_state(state: AgentState, semantic_layer) -> llm.RoutedIntent:
    """Deterministic translation of resolved slots into exactly the shape
    app/semantic/router.py's `route()` would have produced for a single,
    self-contained question — the one place this graph touches the
    pipeline's input, so `SchemaResolver.resolve()` downstream runs exactly
    as it does today, with no awareness that a multi-turn agent exists.

    `time_range` (free-text, e.g. "in 2025" — see app/agent/merge.py) is
    deliberately never modeled as a dimension/filter by merge_intent itself
    (same reasoning as app/semantic/router.py: dates are open-ended, not a
    curated value list). But that means it previously vanished right here —
    SchemaResolver.resolve() only pulls in a table because a metric or
    dimension named it, so a question like "top products by revenue in 2025"
    never got the orders table (and its order_date column) into scope at
    all, leaving the generator unable to see the very column it needed and
    prone to wrongly claiming no such column exists. Appending the catalog's
    designated time_dimension here — mechanically, only when a time_range is
    actually present — fixes that without teaching merge_intent or the
    router anything about dates themselves."""
    domain = state.get("domain")
    filters = [llm.FilterSpec(**f) for f in (state.get("filters") or [])]
    dimensions = list(state.get("dimensions") or [])
    time_dim = semantic_layer.time_dimension
    if state.get("time_range") and time_dim and time_dim not in dimensions:
        dimensions.append(time_dim)
    return llm.RoutedIntent(
        domains=[domain] if domain else [],
        metrics=list(state.get("metrics") or []),
        dimensions=dimensions,
        filters=filters,
        needs_clarification=False,
        clarification_reason="",
        interpretations=[],
    )


def _pipeline_result_to_dict(pr: text2sql_pipeline.PipelineResult) -> dict:
    """Plain-dict projection of the fields a session actually needs to carry
    forward or display — not a full serialization of PipelineResult (whose
    nested objects, e.g. llm.GeneratedSQL, aren't meant to round-trip through
    JSON as the canonical record; the live object is what app/routes/query.py
    builds its HTTP trace from)."""
    return {
        "status": pr.status,
        "sql": pr.sql,
        "metrics": pr.routed.metrics if pr.routed else [],
        "dimensions": pr.routed.dimensions if pr.routed else [],
        "row_count": pr.execution.row_count if pr.execution else None,
        "columns": pr.execution.columns if pr.execution else [],
        "repair_attempts": pr.repair.attempts if pr.repair else 0,
        "violations": list(pr.violations),
    }


def _answer_summary(pr) -> str:
    """Deterministic one-line recap of a pipeline turn for `history` — no
    extra LLM call needed just to summarize fields already on hand, and
    this is what app/agent/intent.py's classifier/responder read to know
    what the last DATA answer actually was."""
    if pr.status == "ok":
        rows = pr.execution.row_count if pr.execution else None
        expl = pr.generated.explanation if pr.generated else ""
        return f"Returned {rows} row(s). {expl}".strip()
    if pr.status == "ambiguous":
        return f"Asked for clarification: {pr.ambiguity.explanation if pr.ambiguity else ''}"
    if pr.status == "blocked":
        return "Blocked by guardrails: " + "; ".join(pr.violations)
    if pr.status == "complexity_rejected":
        return "Rejected as too complex/expensive."
    if pr.status == "error":
        return "Execution error: " + "; ".join(pr.violations)
    return f"Status: {pr.status}"


def _progress_fn(config):
    if config is not None:
        fn = config.get("configurable", {}).get("progress_fn")
        if fn is not None:
            return fn
    return lambda step, detail=None: None


def _merged_to_state_update(state: AgentState, merged) -> dict:
    """Shared by merge_intent_node and clarify_node: a resolved
    (non-ambiguous) MergedIntent applies identically whichever node
    produced it."""
    return {
        "status": "merged",
        # A question is "about" at most one domain for this agent's
        # purposes even though the catalog lets the router list several
        # as context; the first one the merge step names is kept (falling
        # back to whatever was already resolved if this turn's merge
        # didn't name one), same simplification app/text2sql/pipeline.py's
        # callers already make by treating domains as context rather than
        # a hard selector.
        "domain": merged.domains[0] if merged.domains else state.get("domain"),
        "metrics": list(merged.metrics),
        "dimensions": list(merged.dimensions),
        "filters": [f.model_dump() for f in merged.filters],
        "time_range": merged.time_range.model_dump() if merged.time_range else None,
        "pending_clarification": None,
    }


def _clarification_update(merged) -> dict:
    return {
        "pending_clarification": {
            "is_ambiguous": True,
            "explanation": merged.clarification_reason,
            "interpretations": [i.model_dump() for i in merged.interpretations],
        },
    }


# Imported down here, after the shared helpers above are fully defined — each
# node module imports some of those helpers back from this module, so this
# ordering is what lets that work without a circular-import failure (by the
# time Python executes these lines, `app.agent.graph` already has
# build_routed_intent_from_state/_pipeline_result_to_dict/_answer_summary/
# _progress_fn/_merged_to_state_update/_clarification_update set as module
# attributes).
from .nodes.classify import classify_message_node  # noqa: E402
from .nodes.clarify import clarify_node  # noqa: E402
from .nodes.merge_intent import merge_intent_node  # noqa: E402
from .nodes.respond import respond_conversationally_node  # noqa: E402
from .nodes.run_pipeline import run_pipeline_node  # noqa: E402


def build_graph(
    pool,
    live_schema,
    fewshots,
    generate_fn=None,
    executor_fn=None,
    repair_fn=None,
    value_resolver=None,
    merge_parse_structured=None,
    intent_parse_structured=None,
    reformulate_parse_structured=None,
    interpretation_parse_structured=None,
    summarize_parse_structured=None,
    max_agent_steps: int | None = None,
    checkpointer=None,
):
    """Compiles the agent graph bound to fixed, process-level infra
    (pool/live_schema/fewshots — the same values app/routes/query.py already
    passes to pipeline.run() today, not something that varies per turn). The
    per-request `SecurityContext` instead travels through LangGraph's own
    `config["configurable"]` at invoke time, since it must never be
    session-persisted (AgentState has no field for it — see app/agent/state.py)
    and must always come from the real authenticated request, never from the
    agent/LLM. `generate_fn`/`executor_fn`/`repair_fn`/`value_resolver` are
    forwarded straight to pipeline.run() unchanged, so this is unit-testable
    with the exact same fakes tests/test_text2sql/test_pipeline.py already
    uses, no live Postgres/LLM API key required. `merge_parse_structured` is
    forwarded the same way into merge_intent's own `parse_structured` param
    (app/agent/merge.py) — without it, merge_intent_node falls through to
    merge_intent's default (the real app.llm._parse_structured), which would
    make the merge step the one call in this graph that can't be faked from
    the outside; tests always pass one explicitly. Each node function now
    lives in its own module under app/agent/nodes/ and takes its DI
    parameters directly (rather than closing over them) — this function
    binds them in with `functools.partial` when wiring each node, which is
    what keeps the per-node DI seam identical to before the module split."""
    max_steps = max_agent_steps if max_agent_steps is not None else settings.text2sql_max_agent_steps

    graph = StateGraph(AgentState)
    graph.add_node("summarize", partial(summarize_node, parse_structured=summarize_parse_structured))
    graph.add_node(
        "classify_message",
        partial(classify_message_node, intent_parse_structured=intent_parse_structured),
    )
    graph.add_node(
        "respond_conversationally",
        partial(respond_conversationally_node, intent_parse_structured=intent_parse_structured),
    )
    graph.add_node(
        "merge_intent",
        partial(
            merge_intent_node,
            max_steps=max_steps,
            merge_parse_structured=merge_parse_structured,
            reformulate_parse_structured=reformulate_parse_structured,
        ),
    )
    graph.add_node(
        "clarify",
        partial(
            clarify_node,
            pool=pool,
            executor_fn=executor_fn,
            interpretation_parse_structured=interpretation_parse_structured,
            merge_parse_structured=merge_parse_structured,
        ),
    )
    graph.add_node(
        "run_pipeline",
        partial(
            run_pipeline_node,
            pool=pool,
            live_schema=live_schema,
            fewshots=fewshots,
            generate_fn=generate_fn,
            executor_fn=executor_fn,
            repair_fn=repair_fn,
            value_resolver=value_resolver,
        ),
    )
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "classify_message")
    graph.add_conditional_edges(
        "classify_message",
        lambda state: state["status"],
        {"data_question": "merge_intent", "meta": "respond_conversationally"},
    )
    graph.add_edge("respond_conversationally", END)
    graph.add_conditional_edges(
        "merge_intent",
        lambda state: state["status"],
        {"merged": "run_pipeline", "awaiting_clarification": "clarify",
         "error": END, "refused": END},
    )
    graph.add_conditional_edges(
        "clarify",
        lambda state: state["status"],
        # "merged" is merge_intent's own resolution, routed on to run_pipeline
        # as before. "ok"/"blocked"/"complexity_rejected"/"error" are a
        # pipeline-sourced clarification's resume (clarify_node executed the
        # chosen interpretation directly — see clarify_node's "pipeline"
        # branch) and are already terminal pipeline outcomes, not something
        # run_pipeline needs to redo.
        {"merged": "run_pipeline", "ambiguous": END, "refused": END,
         "ok": END, "blocked": END, "complexity_rejected": END, "error": END},
    )
    graph.add_conditional_edges(
        "run_pipeline",
        lambda state: "awaiting_clarification" if state["status"] == "awaiting_clarification" else "done",
        {"awaiting_clarification": "clarify", "done": END},
    )

    # `checkpointer` defaults to in-memory so this graph — and its entire
    # offline test suite — needs no live Postgres; a caller that actually
    # needs interrupt()/resume to survive a process restart (the real API
    # layer) passes one built by `postgres_checkpointer()` above instead.
    return graph.compile(checkpointer=checkpointer or MemorySaver())
