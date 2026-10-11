"""Entry point (after app/agent/memory.py's summarize_node) for every FRESH
top-level turn — a clarification resume re-enters directly at
app/agent/nodes/clarify.py's clarify_node instead, so this never runs
mid-clarification. Decides whether the message is a new/follow-up DATA
question (routed on to merge_intent) or a META one about the conversation/
last answer itself (see app/agent/intent.py for why that distinction needs
to exist at all — it's the fix for a message like "explain the previous
answer" otherwise being merged against resolved slots like any other
follow-up and silently re-asking whatever it was already resolved to).

Moved out of app/agent/graph.py's build_graph(); `intent_parse_structured`
is the same DI seam app/agent/graph.py used to forward via a closure, now
passed in directly and bound at node-registration time via
`functools.partial` (see build_graph).
"""
from .. import intent
from ..graph import _progress_fn
from ..state import AgentState, append_history


async def classify_message_node(state: AgentState, config=None, intent_parse_structured=None) -> dict:
    _progress_fn(config)("classifying")
    if not state.get("pipeline_result"):
        # Nothing to be "meta" about yet — the very first message of a
        # session is always a data question, and classifying it would
        # just be an LLM call with nothing useful to decide.
        kind = "data_question"
    else:
        kwargs = (
            {"parse_structured": intent_parse_structured}
            if intent_parse_structured is not None else {}
        )
        result = intent.classify_message(state, state["user_message"], **kwargs)
        kind = "meta" if result.kind == "meta" else "data_question"
    return {
        "status": kind,
        "history": append_history(state.get("history") or [], "user", state["user_message"]),
    }
