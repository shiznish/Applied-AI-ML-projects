"""Replies to a META turn (see app/agent/nodes/classify.py) directly from
state — plain text, grounded only in the conversation transcript and the
last real DATA answer, never a new SQL/pipeline call. Moved out of
app/agent/graph.py's build_graph().
"""
from .. import intent
from ..graph import _progress_fn
from ..state import AgentState, append_history


async def respond_conversationally_node(
    state: AgentState, config=None, intent_parse_structured=None
) -> dict:
    _progress_fn(config)("responding")
    kwargs = (
        {"parse_structured": intent_parse_structured}
        if intent_parse_structured is not None else {}
    )
    reply = intent.respond_conversationally(state, state["user_message"], **kwargs)
    return {
        "status": "conversational",
        "last_reply": reply,
        "history": append_history(state.get("history") or [], "assistant", reply),
    }
