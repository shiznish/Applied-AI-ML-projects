"""The core text-to-SQL request path: POST /v1/query (governed agentic
pipeline), POST /v1/execute (power-user raw-SQL path, no LLM), and GET
/v1/schema.

POST /v1/query runs every turn through the agent graph (app/agent/graph.py)
rather than calling app/text2sql/pipeline.py directly — merging the new
message against the session's prior resolved state (app/agent/merge.py)
before the graph's one reuse seam, `route_fn`, hands it to the exact same
pipeline.run() this endpoint called pre-agent. The rich post-"ok" validation
below (hallucination check, confidence combination) is unchanged from that
pre-agent version — it still operates on the live
PipelineResult for this turn, just now obtained via the agent graph's
`result_sink` (app/agent/graph.py's run_pipeline_node) instead of a direct
pipeline.run() call, so none of that logic needed to move or change shape.
"""
import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from .. import llm
from ..agent.session_store import load_session, save_session
from ..agent.state import new_session, to_persistable
from ..audit import audit_log
from ..audit.audit_log import AuditRecord
from ..auth.rbac import get_current_user, require_permission
from ..config import settings
from ..db import executor as db_executor
from ..db.connection import SecurityContext, pool as db_pool
from ..guardrails import sql_validator as guardrails
from ..observability import tracing
from ..schema_extract import schema_to_dict, schema_to_markdown
from ..state import log_query, state
from ..text2sql import pipeline as text2sql_pipeline
from ..validation import validation

logger = logging.getLogger("text2sql")

router = APIRouter()


class QueryRequest(BaseModel):
    session_id: str | None = None   # omit for a fresh session; echoed back either way
    message: str


class ExecuteRequest(BaseModel):
    sql: str                     # power-user path: raw SQL through guardrails only


def _context_snapshot(state: dict) -> dict:
    """Extracts the slot fields app/agent/state.py's AgentState actually
    carries (not the question text) — what the sidebar's Active Analytical
    Context card and the per-turn context-diff block both render."""
    return {
        "domain": state.get("domain"),
        "metrics": state.get("metrics") or [],
        "dimensions": state.get("dimensions") or [],
        "filters": state.get("filters") or [],
        "time_range": state.get("time_range"),
    }


def _context_diff_chips(prior: dict, current: dict) -> list[dict]:
    """Per-field diff between the slot state a turn started with (`prior`,
    the session as loaded before this turn's merge_intent ran) and what it
    resolved to (`current`) — each entry is one chip the UI renders as
    either carried-forward context or something this turn actually changed,
    replacing the old "Understood as" progress line with something backed by
    the real resolved state instead of a restated sentence."""
    chips: list[dict] = []
    if current.get("domain"):
        state_label = "inherited" if prior.get("domain") == current["domain"] else "changed"
        chips.append({"field": "domain", "label": current["domain"], "state": state_label})
    prior_metrics = set(prior.get("metrics") or [])
    for m in current.get("metrics") or []:
        chips.append({"field": "metric", "label": m, "state": "inherited" if m in prior_metrics else "changed"})
    prior_dims = set(prior.get("dimensions") or [])
    for d in current.get("dimensions") or []:
        chips.append({"field": "dimension", "label": d, "state": "inherited" if d in prior_dims else "changed"})
    prior_filter_keys = {
        (f.get("dimension"), f.get("operator"), f.get("value")) for f in (prior.get("filters") or [])
    }
    for f in current.get("filters") or []:
        key = (f.get("dimension"), f.get("operator"), f.get("value"))
        chips.append({
            "field": "filter", "label": f"{f.get('dimension')}: {f.get('value')}",
            "state": "inherited" if key in prior_filter_keys else "changed",
        })
    if current.get("time_range"):
        label = current["time_range"].get("description", "") if isinstance(current["time_range"], dict) else str(current["time_range"])
        state_label = "inherited" if prior.get("time_range") == current.get("time_range") else "changed"
        chips.append({"field": "time_range", "label": label, "state": state_label})
    return chips


def _write_audit(
    request: Request,
    security_context: SecurityContext,
    query_id: str,
    question: str | None,
    status: str,
    generated_sql: str | None = None,
    executed_sql: str | None = None,
    repair_attempts: int = 0,
    result: db_executor.ExecutionResult | None = None,
    confidence: float | None = None,
    confidence_breakdown: dict | None = None,
) -> None:
    """Writes one immutable AuditRecord for this request — every /v1/query and
    /v1/execute outcome, not just successful ones, since "who ran what SQL as
    whom" is exactly what an audit trail is for regardless of whether it
    succeeded."""
    row_count = result.row_count if result is not None and result.ok else None
    rhash = (
        audit_log.result_hash(result.columns, result.rows)
        if result is not None and result.ok else None
    )
    audit_log.write(AuditRecord(
        correlation_id=getattr(request.state, "correlation_id", ""),
        query_id=query_id,
        timestamp=time.time(),
        username=security_context.username,
        role=security_context.role,
        region=security_context.region,
        question=question,
        status=status,
        generated_sql=generated_sql,
        executed_sql=executed_sql,
        repair_attempts=repair_attempts,
        row_count=row_count,
        result_hash=rhash,
        confidence=confidence,
        confidence_breakdown=confidence_breakdown,
    ))


@router.post("/v1/query")
async def query(
    req: QueryRequest,
    request: Request,
    security_context: SecurityContext = Depends(require_permission("query:execute")),
):
    if not req.message.strip():
        raise HTTPException(400, "Message is empty")

    session_id = req.session_id or uuid.uuid4().hex
    query_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    # Every span created for the remainder of this request (in pipeline.py,
    # generator.py, llm.py, and the steps below that still live directly in
    # this handler) picks up this correlation id via the contextvar in
    # app/observability/tracing.py — it's the same id the audit trail
    # (app/audit/audit_log.py) and request logs already key on, so a trace,
    # an audit record, and log lines for one request can be cross-referenced.
    correlation_id = getattr(request.state, "correlation_id", "")
    cid_token = tracing.set_correlation_id(correlation_id)
    try:
        return await _handle_query(req, request, security_context, query_id, t0, session_id)
    finally:
        tracing.reset_correlation_id(cid_token)


# Single source of truth for progress-step labels — both this endpoint's own
# post-pipeline steps and app/text2sql/pipeline.py's internal ones (which
# report a bare step key via the progress_fn threaded through
# app/agent/graph.py) are looked up here, so the text the UI shows always
# describes a step that's actually running right now, not a guess.
PROGRESS_LABELS = {
    "classifying": "Figuring out what you're asking…",
    "responding": "Thinking about how to reply…",
    "reformulating": "Understood as:",
    "merging": "Merging this message into the conversation…",
    "resolving": "Resolving metrics, dimensions & filters…",
    "linking": "Linking schema & joins…",
    "generating": "Generating SQL…",
    "generated": "Draft query:",
    "guardrails": "Checking guardrails & cost limits…",
    "cost_guard": "Checking cost guard…",
    "executing": "Running the query…",
    "repairing": "Repairing the query after an error…",
    "hallucination_check": "Double-checking the SQL answers your question…",
    "confidence": "Scoring confidence…",
}


@router.post("/v1/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    security_context: SecurityContext = Depends(require_permission("query:execute")),
):
    """Same contract as POST /v1/query, but as Server-Sent Events: a
    `data:` line per real progress step as app/text2sql/pipeline.py and this
    module's own post-pipeline validation steps actually reach them, then one
    final `event: done` line carrying the exact JSON body POST /v1/query
    would have returned. Exists purely so the UI can show real progress
    instead of a timed guess — POST /v1/query itself is unchanged and still
    the right choice for any non-browser caller that just wants the result.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message is empty")

    session_id = req.session_id or uuid.uuid4().hex
    query_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    correlation_id = getattr(request.state, "correlation_id", "")

    queue: asyncio.Queue = asyncio.Queue()

    def progress_fn(step: str, detail: str | None = None) -> None:
        queue.put_nowait((step, detail))

    async def run_and_capture():
        cid_token = tracing.set_correlation_id(correlation_id)
        try:
            return await _handle_query(
                req, request, security_context, query_id, t0, session_id, progress_fn=progress_fn,
            )
        finally:
            tracing.reset_correlation_id(cid_token)

    async def event_stream():
        task = asyncio.create_task(run_and_capture())
        # A real gap between two progress steps (schema resolution,
        # execution, and repair-on-error can each legitimately take several
        # seconds with no step boundary in between) left this connection
        # fully idle for 10+ seconds — long enough that something in the
        # local network path (observed: not this server, not FastAPI/
        # Starlette; reproduced identically via a direct httpx client
        # against this same process, which tolerated the gap fine) silently
        # dropped it, surfacing in the browser as
        # net::ERR_INCOMPLETE_CHUNKED_ENCODING. A `:`-prefixed line is an
        # SSE comment — ignored by any real event parser, including ours —
        # sent here purely to keep bytes flowing often enough that no gap
        # is long enough to look idle.
        last_sent = time.monotonic()
        def _event(step: str, detail: str | None) -> str:
            payload = {"step": step, "label": PROGRESS_LABELS.get(step, step)}
            if detail:
                payload["detail"] = detail
            return f"data: {json.dumps(payload)}\n\n"

        try:
            while not task.done():
                try:
                    step, detail = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_sent > 2:
                        yield ": heartbeat\n\n"
                        last_sent = time.monotonic()
                    continue
                yield _event(step, detail)
                last_sent = time.monotonic()
            # The task may have queued a final step just before finishing —
            # drain it so the UI sees it even though the loop above already
            # noticed task.done() on the same tick.
            while not queue.empty():
                step, detail = queue.get_nowait()
                yield _event(step, detail)
            result = task.result()
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps({'status': e.status_code, 'detail': e.detail})}\n\n"
            return
        except Exception as e:
            logger.exception("POST /v1/query/stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return
        # default=str: e.g. ExecutionResult.elapsed_ms is a float, fine as-is,
        # but a row value can be a Decimal/datetime straight from asyncpg —
        # same reasoning as app/audit/audit_log.py's own default=str usage.
        yield f"event: done\ndata: {json.dumps(result, default=str)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _handle_query(
    req: "QueryRequest",
    request: Request,
    security_context: SecurityContext,
    query_id: str,
    t0: float,
    session_id: str,
    progress_fn=None,
):
    # Purely observational, same no-op-by-default shape as
    # app/text2sql/pipeline.py's own progress_fn — POST /v1/query/stream
    # (below) is the only caller that passes a real one.
    report = progress_fn or (lambda step, detail=None: None)
    # A session row belongs to exactly one user (app/agent/session_store.py
    # enforces this with WHERE user_id = $1, never just by convention) — a
    # caller supplying someone else's session_id transparently gets a fresh
    # session instead of that user's state — fail-safe (a blank session),
    # never fail-open (someone else's data).
    prior_state = await load_session(db_pool, session_id, security_context.username)
    is_resume = prior_state is not None and prior_state["status"] == "awaiting_clarification"
    if prior_state is None:
        prior_state = new_session(session_id)

    # `result_sink` is how app/agent/graph.py's run_pipeline_node hands back
    # the live PipelineResult for this turn (when the agent graph actually
    # reached it) without putting a non-JSON-serializable object into the
    # AgentState that gets persisted below — see that module's docstring.
    result_sink: dict = {}
    config = {
        "configurable": {
            "thread_id": session_id,
            "security_context": security_context,
            "result_sink": result_sink,
            "progress_fn": report,
        },
    }

    # ---- Merge this message into the session's resolved state, then (unless
    #      it's ambiguous) run it through app/text2sql/pipeline.py exactly as
    #      a single-turn question would — all via the compiled agent graph
    #      built once at startup (app/main.py's lifespan). A resume uses
    #      LangGraph's own Command(resume=...) to continue the SAME paused
    #      clarify_node with this message as the reply (see app/agent/
    #      graph.py's module docstring for why that must be a different node
    #      than the one that first classified the message as ambiguous);
    #      anything else is a plain turn merged against the session row
    #      `load_session` just loaded above, not against whatever
    #      LangGraph's own checkpointer happens to still hold for this
    #      thread_id. -------------------------------------------------------
    try:
        if is_resume:
            raw_state = await state["agent_graph"].ainvoke(Command(resume=req.message), config=config)
        else:
            turn_input = dict(prior_state)
            turn_input["user_message"] = req.message
            raw_state = await state["agent_graph"].ainvoke(turn_input, config=config)
    except llm.LLMAuthError as e:
        raise HTTPException(502, f"{e} — set the API key in .env and restart")
    except llm.LLMError as e:
        raise HTTPException(502, str(e))

    agent_state = to_persistable(raw_state)
    await save_session(db_pool, session_id, security_context.username, agent_state)

    current_context = _context_snapshot(agent_state)
    context_diff = _context_diff_chips(_context_snapshot(prior_state), current_context)

    # ---- Paused on a real clarification (LangGraph interrupt(), or still
    #      ambiguous after one clarifying reply — app/agent/graph.py's
    #      clarify_node treats that as terminal rather than pausing again).
    #      Neither reaches run_pipeline_node, so result_sink is empty. -------
    if "__interrupt__" in raw_state or (
        agent_state["status"] == "ambiguous" and result_sink.get("pr") is None
    ):
        pc = agent_state["pending_clarification"] or {}
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(),
            "message": req.message, "username": security_context.username,
            "status": "clarification",
            "interpretations": pc.get("interpretations", []),
            "explanation": pc.get("explanation", ""),
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(request, security_context, query_id, req.message, status="ambiguous")
        return record

    # ---- Not a legitimate data request at all (a destructive command,
    #      something with no possible mapping onto the catalog) —
    #      app/agent/merge.py's MergedIntent.answerable=false, which
    #      app/agent/graph.py routes straight to END, never through
    #      clarify_node's interrupt() — there is nothing to offer a choice
    #      between, so this is terminal like an error, not a pause. -------
    if agent_state["status"] == "refused":
        pc = agent_state["pending_clarification"] or {}
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(),
            "message": req.message, "username": security_context.username,
            "status": "refused", "explanation": pc.get("explanation", ""),
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(request, security_context, query_id, req.message, status="refused")
        return record

    # ---- A meta question about the conversation itself ("explain that",
    #      "why did you do X") — app/agent/graph.py's classify_message_node
    #      routed it to respond_conversationally_node instead of merge_intent,
    #      so there's no new SQL/PipelineResult for this turn at all; the
    #      reply is plain text grounded in the last real answer. -------------
    if agent_state["status"] == "conversational":
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(),
            "message": req.message, "username": security_context.username,
            "status": "conversational", "reply": agent_state.get("last_reply") or "",
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(request, security_context, query_id, req.message, status="conversational")
        return record

    # ---- Too many clarification rounds (app/config.py's
    #      TEXT2SQL_MAX_AGENT_STEPS) — app/agent/graph.py's merge_intent_node
    #      short-circuits here instead of looping forever. --------------------
    if agent_state["status"] == "error" and result_sink.get("pr") is None:
        err = (agent_state["pipeline_result"] or {}).get("error", "Too many clarification rounds")
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(),
            "message": req.message, "username": security_context.username,
            "status": "error", "error": err,
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(request, security_context, query_id, req.message, status="error")
        return record

    # ---- From here on the agent graph actually reached run_pipeline_node
    #      for this turn, so `pr` is the live PipelineResult (not a dict) and
    #      `question` is the question synthesized from resolved slot state —
    #      everything below is otherwise unchanged from the pre-agent,
    #      single-turn version of this handler. --------------------------
    pr = result_sink["pr"]
    question = result_sink["question"]

    if pr.status == "ambiguous":
        # pr.ambiguity is populated identically regardless of which of the
        # three possible sources caught it (semantic router, deterministic
        # resolver/value-grounding, or the SQL generator's own
        # needs_clarification) — see app/text2sql/pipeline.py.
        amb = pr.ambiguity
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(), "question": question,
            "username": security_context.username,
            "status": "clarification",
            "interpretations": amb.interpretations,
            "explanation": amb.explanation,
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(
            request, security_context, query_id, question, status="ambiguous",
            generated_sql=pr.generated.sql if pr.generated is not None else None,
        )
        return record

    if pr.status == "blocked":
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(), "question": question,
            "username": security_context.username,
            "status": "blocked", "sql": pr.sql, "violations": pr.violations,
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(
            request, security_context, query_id, question, status="blocked",
            generated_sql=pr.generated.sql if pr.generated is not None else None,
            executed_sql=pr.sql,
        )
        return record

    if pr.status == "complexity_rejected":
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(), "question": question,
            "username": security_context.username,
            "status": "complexity_rejected", "sql": pr.sql,
            "error": pr.execution.error if pr.execution is not None else "Query rejected as too complex/expensive",
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(
            request, security_context, query_id, question, status="complexity_rejected",
            generated_sql=pr.generated.sql if pr.generated is not None else None,
            executed_sql=pr.sql, repair_attempts=pr.repair.attempts if pr.repair else 0,
        )
        return record

    if pr.status == "error":
        record = {
            "query_id": query_id, "session_id": session_id, "ts": time.time(), "question": question,
            "username": security_context.username,
            "status": "error", "sql": pr.sql,
            "error": pr.execution.error if pr.execution is not None else "; ".join(pr.violations),
            "context": current_context, "context_diff": context_diff,
        }
        log_query(record)
        _write_audit(
            request, security_context, query_id, question, status="error",
            generated_sql=pr.generated.sql if pr.generated is not None else None,
            executed_sql=pr.sql, repair_attempts=pr.repair.attempts if pr.repair else 0,
        )
        return record

    # ---- status == "ok" from here on -----------------------------------------
    schema = state["schema"]
    gen = pr.generated
    guard_sql = pr.sql
    result = pr.execution
    schema_md = schema_to_markdown(pr.linked.tables) if pr.linked else schema_to_markdown(schema)

    # ---- 4. Hallucination detection ------------------------------------------
    report("hallucination_check")
    alignment_score = None
    back_translated = None
    alignment_reasoning = None
    with tracing.traced_span("text2sql.hallucination_check") as span:
        try:
            back_translated = llm.back_translate(guard_sql, schema_md)
            judgment = llm.judge_alignment(question, back_translated)
            alignment_score = judgment.score
            alignment_reasoning = judgment.reasoning
            span.set_attribute("text2sql.hallucination_check.alignment_score", alignment_score)
        except llm.LLMError as e:
            logger.warning("Back-translation skipped: %s", e)
            span.set_attribute("text2sql.hallucination_check.skipped", True)

    sanity = validation.sanity_check(result, state["date_range"])
    # Check coverage against the same table identities the model was actually
    # shown/generated against, not the role-agnostic introspected schema —
    # for a non-admin caller that's `customers_masked`, not `customers` (see
    # app/db/masking.py), and using the raw introspected schema here would
    # flag that as a phantom table on every non-admin customer query.
    coverage_schema = pr.linked.tables if pr.linked else schema
    coverage, coverage_problems = validation.schema_coverage(
        gen.tables_used, gen.columns_used, schema_to_dict(coverage_schema)
    )

    # ---- 5. Confidence --------------------------------------------------------
    report("confidence")
    # Always 1.0: by construction, execution only reaches this point via
    # pipeline.py's "ok" status, which never carries an AmbiguityCheck (every
    # ambiguity source — semantic router, deterministic resolver/value
    # grounding, or the SQL generator's own needs_clarification — short-
    # circuits to status="ambiguous" before generation/execution runs). Kept
    # as an explicit named signal (not just inlined as 1.0 everywhere) so
    # app/validation/validation.py's combine_confidence still documents
    # ambiguity as one of its inputs, and so a future pipeline change that
    # proceeds despite flagged ambiguity has an obvious place to wire a real
    # partial score back in.
    amb_score = 1.0
    with tracing.traced_span("text2sql.confidence_combination") as span:
        confidence, breakdown = validation.combine_confidence(
            alignment_score, sanity.score, coverage, amb_score,
            model_confidence=gen.confidence,
        )
        span.set_attribute("text2sql.confidence.score", confidence)

    now = time.perf_counter()
    total_ms = round((now - t0) * 1000, 1)

    # The session's own `pipeline_result` (written by run_pipeline_node,
    # before any of this post-pipeline validation runs — see
    # app/agent/graph.py's _pipeline_result_to_dict) only ever had the raw
    # PipelineResult fields. Enriching it here and re-saving means a later
    # "meta" turn (app/agent/intent.py's respond_conversationally, via
    # app/prompts/intent/v2.py's last_answer_prompt) can actually discuss
    # confidence/the hallucination check instead of truthfully claiming it
    # has no visibility into them — it used to be correct about that, which
    # was itself the bug.
    agent_state["pipeline_result"] = {
        **(agent_state.get("pipeline_result") or {}),
        "confidence": confidence,
        "low_confidence": confidence < settings.low_confidence_threshold,
        "back_translated_question": back_translated,
        "alignment_reasoning": alignment_reasoning,
        "sanity_warnings": sanity.warnings,
        "coverage_problems": coverage_problems,
    }
    await save_session(db_pool, session_id, security_context.username, agent_state)

    record = {
        "query_id": query_id,
        "session_id": session_id,
        "ts": time.time(),
        "question": question,
        "username": security_context.username,
        "status": "ok",
        "domains": pr.routed.domains if pr.routed is not None else [],
        "metrics": pr.routed.metrics if pr.routed is not None else [],
        "dimensions": pr.routed.dimensions if pr.routed is not None else [],
        "sql": guard_sql,
        "explanation": gen.explanation,
        "guardrail_notes": pr.guardrail_notes,
        "repair_attempts": pr.repair.attempts if pr.repair else 0,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "elapsed_ms": result.elapsed_ms,
        "explain_plan": result.explain_plan,
        "confidence": confidence,
        "confidence_breakdown": breakdown,
        "low_confidence": confidence < settings.low_confidence_threshold,
        "back_translated_question": back_translated,
        "alignment_reasoning": alignment_reasoning,
        "sanity_warnings": sanity.warnings,
        "coverage_problems": coverage_problems,
        "total_ms": total_ms,
        "context": current_context, "context_diff": context_diff,
    }
    log_query(record)
    _write_audit(
        request, security_context, query_id, question, status="ok",
        generated_sql=gen.sql, executed_sql=guard_sql,
        repair_attempts=pr.repair.attempts if pr.repair else 0,
        result=result, confidence=confidence, confidence_breakdown=breakdown,
    )
    return record


@router.post("/v1/execute")
async def execute_raw(
    req: ExecuteRequest,
    request: Request,
    security_context: SecurityContext = Depends(require_permission("query:execute")),
):
    """Power-user path: run (edited) SQL through guardrails + sandbox, no LLM.
    No generated_sql/confidence exist for this path (there's no LLM call) —
    but it still ran SQL against a real user's identity, which is exactly
    what the audit trail exists to capture, so every outcome is audited too."""
    query_id = uuid.uuid4().hex[:12]
    guard = guardrails.check(req.sql)
    if not guard.allowed:
        _write_audit(
            request, security_context, query_id, None, status="blocked",
            executed_sql=req.sql,
        )
        return {"status": "blocked", "sql": req.sql, "violations": guard.violations}
    result = await db_executor.execute(db_pool, guard.sql, security_context=security_context)
    if not result.ok:
        _write_audit(
            request, security_context, query_id, None, status="error",
            executed_sql=guard.sql,
        )
        return {"status": "error", "sql": guard.sql, "error": result.error}
    _write_audit(
        request, security_context, query_id, None, status="ok",
        executed_sql=guard.sql, result=result,
    )
    return {
        "status": "ok", "sql": guard.sql, "guardrail_notes": guard.notes,
        "columns": result.columns, "rows": result.rows,
        "row_count": result.row_count, "elapsed_ms": result.elapsed_ms,
        "explain_plan": result.explain_plan,
    }


@router.get("/v1/schema")
async def get_schema(security_context: SecurityContext = Depends(get_current_user)):
    return {"date_range": state["date_range"], "tables": schema_to_dict(state["schema"])}
