"""The one new per-turn LLM call that resolves catalog slots for a fresh
DATA-question turn — see app/agent/merge.py for the actual classification
logic and prompt. Moved out of app/agent/graph.py's build_graph(); `max_steps`,
`merge_parse_structured` and `reformulate_parse_structured` are the same DI
seams build_graph used to close over directly, now passed in and bound at
node-registration time via `functools.partial` (see build_graph).
"""
from ...semantic import resolver
from .. import merge, reformulate
from ..graph import _clarification_update, _merged_to_state_update, _progress_fn
from ..state import AgentState, append_history


async def merge_intent_node(
    state: AgentState,
    config=None,
    max_steps: int = 6,
    merge_parse_structured=None,
    reformulate_parse_structured=None,
) -> dict:
    steps = state["agent_steps"] + 1
    if steps > max_steps:
        return {
            "agent_steps": steps,
            "status": "error",
            "pipeline_result": {"status": "error", "error": "too many clarification rounds"},
        }

    # Rewrite an elliptical follow-up ("less order value") into one
    # standalone, complete question ("Which country has the lowest
    # average order value?") BEFORE any catalog/slot matching — see
    # app/agent/reformulate.py's module docstring for why this has to
    # happen here rather than being left to merge_intent's own
    # slot-carrying, and why hand-curating a metric per phrase (as this
    # project did for cancellation_rate/average_order_value) doesn't
    # scale to every possible elliptical question on its own.
    reformulate_kwargs = (
        {"parse_structured": reformulate_parse_structured}
        if reformulate_parse_structured is not None else {}
    )
    reformulated = reformulate.reformulate_question(
        state, state["user_message"], **reformulate_kwargs
    )
    _progress_fn(config)("reformulating")
    _progress_fn(config)("merging")

    semantic_layer = resolver.get()
    merge_kwargs = (
        {"parse_structured": merge_parse_structured} if merge_parse_structured is not None else {}
    )
    merged = merge.merge_intent(state, reformulated, semantic_layer, **merge_kwargs)

    if merged.needs_clarification:
        # answerable=false means there is no choice to offer at all — a
        # destructive command, something with no legitimate mapping onto
        # the catalog. That's terminal (status="refused", no interrupt,
        # no point waiting for a reply the same way genuine ambiguity
        # does — see app/agent/merge.py's MergedIntent.answerable), not
        # a pause to resume: "awaiting_clarification" is reserved for
        # real ambiguity where the user has something to choose between.
        return {
            "agent_steps": steps if merged.answerable else 0,
            "status": "awaiting_clarification" if merged.answerable else "refused",
            **_clarification_update(merged),
            "user_message": reformulated,
            "history": append_history(
                state.get("history") or [], "assistant",
                f"Asked for clarification: {merged.clarification_reason}"
                if merged.answerable else f"Declined: {merged.clarification_reason}",
            ),
        }

    # Resolved on the first try — reset the counter rather than leaving it
    # at `steps`, which would otherwise ratchet up by one on every
    # question this session ever asks (ambiguous or not) and eventually
    # trip `max_steps` for good. The cap exists to stop one question's
    # clarification loop from bouncing forever, not to budget a lifetime
    # number of questions per session — see the test this guards,
    # `test_agent_steps_cap_stops_instead_of_looping_forever`.
    return {
        "agent_steps": 0, **_merged_to_state_update(state, merged),
        "user_message": reformulated,
    }
