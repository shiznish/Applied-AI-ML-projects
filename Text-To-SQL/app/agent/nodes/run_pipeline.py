"""Hands the merged, resolved slot state to the EXISTING pipeline
(app/text2sql/pipeline.py) completely unmodified, via its one injectable
reuse seam, `route_fn`. Moved out of app/agent/graph.py's build_graph();
`pool`/`live_schema`/`fewshots`/`generate_fn`/`executor_fn`/`repair_fn`/
`value_resolver` are the same DI seams build_graph used to close over
directly, now passed in and bound at node-registration time via
`functools.partial` (see build_graph).
"""
from ...db.connection import SecurityContext
from ...text2sql import pipeline as text2sql_pipeline
from .. import merge
from ..graph import _answer_summary, _pipeline_result_to_dict, _progress_fn, build_routed_intent_from_state
from ..state import AgentState, append_history


async def run_pipeline_node(
    state: AgentState,
    config=None,
    pool=None,
    live_schema=None,
    fewshots=None,
    generate_fn=None,
    executor_fn=None,
    repair_fn=None,
    value_resolver=None,
) -> dict:
    security_context: SecurityContext | None = None
    result_sink: dict | None = None
    if config is not None:
        configurable = config.get("configurable", {})
        security_context = configurable.get("security_context")
        result_sink = configurable.get("result_sink")

    # synthesize_question() renders only what the resolved SLOTS capture
    # (domain/metrics/dimensions/filters/time_range) — a deterministic,
    # catalog-constrained paraphrase. That's necessary to carry prior
    # turns' context into one coherent question, but it's also lossy: any
    # nuance in the user's own phrasing that the catalog can't represent
    # as a slot (e.g. "rate", "percentage", "trend") gets silently
    # dropped, because slot-filling can only approximate with the closest
    # available metric/dimension (see app/semantic/layer.yaml's
    # cancellation_rate metric and app/prompts/sql_generation.py's rate/
    # ratio rule — this is the other half of that fix: neither helps if
    # the generator never sees the word "rate" in the first place).
    # Appending the user's exact wording alongside the synthesized summary
    # costs nothing when they already agree, and recovers this case
    # without having to detect "is this a follow-up" to decide which to
    # send.
    synthesized = merge.synthesize_question(state)
    user_message = state.get("user_message") or ""
    question = (
        synthesized if synthesized.strip().lower() == user_message.strip().lower()
        else f'{synthesized}\n\n(User\'s exact wording for this turn: "{user_message}")'
    )
    pr = await text2sql_pipeline.run(
        question,
        live_schema,
        pool,
        fewshots,
        generate_fn=generate_fn,
        executor_fn=executor_fn,
        repair_fn=repair_fn,
        route_fn=lambda q, layer: build_routed_intent_from_state(state, layer),
        value_resolver=value_resolver,
        security_context=security_context,
        progress_fn=_progress_fn(config),
    )
    if pr.status == "ambiguous" and pr.generated is not None:
        # Generator-level ambiguity — as opposed to router/resolver-level
        # ambiguity (pr.generated is None, a catalog term never even made
        # it to generation) — comes with real, executable SQL per
        # interpretation (app/text2sql/pipeline.py no longer auto-accepts
        # a lone one). That's resumable the same way merge-level
        # ambiguity already is: pause via clarify_node, carrying enough
        # plain-JSON state (interpretations, the original question,
        # tables/columns used) to directly EXECUTE whichever one the
        # reply selects on resume, rather than re-deriving a question
        # from scratch and risking the exact dead end this guards against
        # (see app/agent/interpretation.py and app/agent/nodes/clarify.py).
        # Router/resolver-level ambiguity has no SQL to execute at all —
        # that stays a terminal "ambiguous" status, unchanged.
        return {
            "status": "awaiting_clarification",
            "pending_clarification": {
                "source": "pipeline",
                "is_ambiguous": True,
                "explanation": pr.generated.explanation,
                "interpretations": [i.model_dump() for i in pr.generated.interpretations],
                "original_question": question,
                "tables_used": list(pr.generated.tables_used),
                "columns_used": list(pr.generated.columns_used),
            },
            "history": append_history(
                state.get("history") or [], "assistant",
                f"Asked for clarification: {pr.generated.explanation}",
            ),
        }

    if result_sink is not None:
        # Lets a caller (app/routes/query.py) get at the live PipelineResult
        # for this turn — e.g. to build the rich "ok"-status response
        # (hallucination check, multi-query agreement, confidence
        # breakdown) the same way the pre-agent /v1/query handler did —
        # without putting a non-JSON-serializable object into AgentState,
        # which app/agent/session_store.py persists as plain JSONB and
        # LangGraph's own checkpointer serializes independently. Passed
        # via config rather than a build_graph-time closure because it
        # must be fresh per request, not shared across concurrent callers
        # of one compiled graph.
        result_sink["pr"] = pr
        result_sink["question"] = question
    return {
        "status": pr.status,
        "pipeline_result": _pipeline_result_to_dict(pr),
        "history": append_history(state.get("history") or [], "assistant", _answer_summary(pr)),
    }
