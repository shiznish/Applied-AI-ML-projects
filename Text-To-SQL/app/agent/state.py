"""Shared state for the governed agentic layer sitting on top of
`app.text2sql.pipeline`. This module holds only the slot/session shape; the
LangGraph wiring that reads and writes it lives in `app.agent.graph` (and,
for history-budget/summarization concerns, `app.agent.memory`).

`AgentState` is a `TypedDict` (not a dataclass or Pydantic model) because
LangGraph's `StateGraph` expects a `TypedDict`-shaped state so it can merge
per-node partial updates by key.
"""
from typing import TypedDict


class AgentState(TypedDict):
    session_id: str
    user_message: str
    # The actual conversation transcript — [{"role": "user"|"assistant",
    # "content": str}, ...], oldest first, NEVER truncated: the full history
    # is what gets persisted verbatim to `agent_sessions.state_jsonb` via
    # app/agent/session_store.py.save_session, so dropping entries here would
    # mean losing them from Postgres too, not just from this turn's prompts.
    # Distinct from domain/metrics/dimensions/filters/time_range below: those
    # are the deterministic SLOT summary app/agent/merge.py resolves a
    # question down to (what SQL generation actually acts on, and the only
    # thing a catalog-constrained metric/dimension vocabulary can represent);
    # `history` is what the user and assistant actually SAID, verbatim, which
    # is what app/agent/intent.py's classifier and conversational-reply node
    # need to behave like an ordinary chat rather than a slot machine, and
    # what merge_intent now also sees so nuance the slot vocabulary can't
    # capture (e.g. "rate", "why", a passing remark) isn't silently dropped
    # just because it doesn't map onto a catalog field.
    history: list[dict]
    # Running LLM-produced summary of whatever part of `history` no longer
    # fits app/config.py's `agent_history_token_budget` — None until a
    # session's transcript actually grows past that budget. Maintained by
    # app/agent/memory.py's summarize_node and rendered as a leading line by
    # render_transcript (also in app/agent/memory.py); never a substitute for
    # the full `history` above, only a budget-friendly stand-in for the part
    # of it this turn's prompts don't render raw.
    history_summary: str | None
    domain: str | None
    metrics: list[str]
    dimensions: list[str]
    filters: list[dict]
    time_range: dict | None
    pending_clarification: dict | None
    agent_steps: int
    repair_attempts: int
    status: str
    pipeline_result: dict | None
    # Plain-text reply for a turn classified as conversational/meta (see
    # app/agent/intent.py) — distinct from pipeline_result, which always
    # holds the last real DATA answer so a later meta turn still has
    # something concrete to explain even though it didn't produce one itself.
    last_reply: str | None


def new_session(session_id: str) -> AgentState:
    """The empty/default state a brand-new session starts from."""
    return AgentState(
        session_id=session_id,
        user_message="",
        history=[],
        history_summary=None,
        domain=None,
        metrics=[],
        dimensions=[],
        filters=[],
        time_range=None,
        pending_clarification=None,
        agent_steps=0,
        repair_attempts=0,
        status="new",
        pipeline_result=None,
        last_reply=None,
    )


def append_history(history: list[dict], role: str, content: str) -> list[dict]:
    """Returns a NEW list (never mutates `history` in place — state updates
    in a LangGraph node must be treated as immutable, same discipline as
    every other field here) with one more {"role", "content"} entry
    appended. Never truncates: the complete transcript must always make it
    into `agent_sessions.state_jsonb` (see app/agent/session_store.py), full
    stop — budgeting what a given prompt actually RENDERS from this list is
    a separate, LLM-summarization-backed concern handled entirely by
    app/agent/memory.py's build_context/summarize_node/render_transcript, not
    by dropping data here."""
    return history + [{"role": role, "content": content}]


def to_persistable(state: dict) -> AgentState:
    """Strips anything LangGraph's `ainvoke` result carries beyond
    `AgentState`'s own fields (e.g. `__interrupt__`, present on a turn that
    paused — a list of `Interrupt` objects, not JSON-serializable) before
    handing the state to app/agent/session_store.py, which persists it as
    plain JSONB."""
    return AgentState(**{k: state[k] for k in AgentState.__annotations__ if k in state})
