"""Pauses on a real ambiguity (via LangGraph's `interrupt()`) and resumes
against the clarifying reply — see app/agent/graph.py's module docstring for
why this has to be a separate node from merge_intent_node rather than
merge_intent_node interrupting directly. Moved out of app/agent/graph.py's
build_graph(); `pool`, `executor_fn`, `interpretation_parse_structured` and
`merge_parse_structured` are the same DI seams build_graph used to close
over directly, now passed in and bound at node-registration time via
`functools.partial` (see build_graph).
"""
from langgraph.types import interrupt

from ... import llm
from ...semantic import resolver
from ...text2sql import pipeline as text2sql_pipeline
from .. import interpretation, merge
from ..graph import (
    _answer_summary, _clarification_update, _merged_to_state_update, _pipeline_result_to_dict,
    _progress_fn,
)
from ..state import AgentState, append_history


async def clarify_node(
    state: AgentState,
    config=None,
    pool=None,
    executor_fn=None,
    interpretation_parse_structured=None,
    merge_parse_structured=None,
) -> dict:
    # Pauses here on first entry (see app/agent/graph.py's module docstring
    # for why this is a separate node from merge_intent_node); on resume,
    # `interrupt()` returns the clarifying reply a caller supplied via
    # `graph.ainvoke(Command(resume=<reply>), config=...)` instead of
    # pausing again.
    reply = interrupt(state["pending_clarification"])
    pending = state.get("pending_clarification") or {}
    # A resume bypasses classify_message_node entirely (it re-enters
    # directly here via interrupt()/Command(resume=...) — see this
    # function's own docstring and app/agent/intent.py's module
    # docstring), so the reply itself has never been recorded in
    # `history` yet; this is the only place that can do it.
    history_with_reply = append_history(state.get("history") or [], "user", reply)

    if pending.get("source") == "pipeline":
        # The generator already produced and showed real SQL per
        # interpretation (app/text2sql/pipeline.py's needs_clarification
        # path, no longer auto-accepted) — resolving the reply here means
        # picking which one, not re-merging catalog slots (there's
        # nothing about domain/metric/dimension to resolve; the gap was
        # in how the generator expressed a condition in SQL). Executing
        # the CHOSEN interpretation's already-shown SQL directly — rather
        # than re-running merge_intent + generation from scratch — is
        # what actually avoids the old dead end: a vague "proceed
        # anyway" reply used to re-derive the identical question against
        # unchanged slots and hit the identical wall every time; here it
        # just resolves to index 1 (app/agent/interpretation.py) and
        # executes, guaranteed forward progress on any reply.
        _progress_fn(config)("executing")
        interpretations = pending.get("interpretations") or []
        interp_kwargs = (
            {"parse_structured": interpretation_parse_structured}
            if interpretation_parse_structured is not None else {}
        )
        idx = interpretation.choose_interpretation(
            reply, interpretations, state.get("history") or [], **interp_kwargs,
        )
        chosen = interpretations[idx - 1]
        security_context = None
        result_sink: dict | None = None
        if config is not None:
            configurable = config.get("configurable", {})
            security_context = configurable.get("security_context")
            result_sink = configurable.get("result_sink")
        pr = await text2sql_pipeline.run_selected_interpretation(
            chosen["example_sql"], chosen["description"],
            pending.get("tables_used") or [], pending.get("columns_used") or [],
            pool, executor_fn=executor_fn, security_context=security_context,
            progress_fn=_progress_fn(config),
        )
        if result_sink is not None:
            result_sink["pr"] = pr
            result_sink["question"] = pending.get("original_question") or chosen["description"]
        return {
            "status": pr.status,
            "pending_clarification": None,
            "pipeline_result": _pipeline_result_to_dict(pr),
            "history": append_history(history_with_reply, "assistant", _answer_summary(pr)),
        }

    _progress_fn(config)("merging")
    semantic_layer = resolver.get()
    pending_interpretations = [
        llm.Interpretation(**i) for i in (pending.get("interpretations") or [])
    ]
    merge_kwargs = (
        {"parse_structured": merge_parse_structured} if merge_parse_structured is not None else {}
    )
    merged = merge.merge_intent(
        state, reply, semantic_layer,
        pending_interpretations=pending_interpretations,
        **merge_kwargs,
    )

    if merged.needs_clarification:
        # Still unresolved after one reply — stays ambiguous rather than
        # interrupting again indefinitely in this same node; a further
        # reply needs a new top-level turn. answerable=false means the
        # reply itself turned out to be something that can't be fulfilled
        # at all (same terminal "refused" outcome merge_intent_node
        # produces, not more ambiguity to keep bouncing on).
        return {
            "status": "ambiguous" if merged.answerable else "refused",
            **_clarification_update(merged),
            "history": append_history(
                history_with_reply, "assistant",
                f"Asked for clarification: {merged.clarification_reason}"
                if merged.answerable else f"Declined: {merged.clarification_reason}",
            ),
        }

    return {"agent_steps": 0, **_merged_to_state_update(state, merged), "history": history_with_reply}
