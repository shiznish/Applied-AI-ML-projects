"""Rewrites a (possibly elliptical) follow-up message into one standalone,
complete natural-language question, using the real conversation transcript
for context — BEFORE any catalog/slot resolution happens.

Why this needs to exist as its own step rather than relying on
app/agent/merge.py's slot-carrying alone: slot-carrying can only preserve
context that already fits the catalog's metric/dimension vocabulary. A
follow-up like "less order value" has no slot of its own to merge into — it
has to be recognized, in plain language, as "the same question as before,
but asking for the lowest instead of the highest" before that recognition
can even be attempted against the catalog. Expanding the catalog one metric
at a time (as this project did for cancellation_rate and
average_order_value in app/semantic/layer.yaml) fixes specific named
business metrics, which is still worth doing for governance/consistency —
but it doesn't scale to every possible elliptical phrasing, and this step is
the general fix for that: resolve what's being asked in plain English
first, then hand the fully-specified standalone question to the existing
resolution path exactly as if it had been typed that way from the start.

The system prompt itself lives in app/prompts/reformulate/ — this module
only orchestrates the LLM call and defines the structured-output contract.
"""
from pydantic import BaseModel

from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import reformulate as reformulate_prompts
from .state import AgentState


class ReformulatedQuestion(BaseModel):
    question: str


def reformulate_question(
    state: AgentState,
    user_message: str,
    parse_structured=llm._parse_structured,
) -> str:
    """Returns a standalone, complete question — unchanged if `user_message`
    is already self-contained (a fresh, unrelated question), expanded using
    conversation context when it's an elliptical follow-up. Deliberately NOT
    called from app/agent/graph.py's clarify_node: a clarification REPLY
    ("proceed anyway", "the second one") is answering a specific question
    already on the table, not itself a new data question to restate — that
    case is handled by merge_intent's own pending-interpretations prompt
    (see app/agent/merge.py), not this one."""
    # app/agent/graph.py's classify_message_node always appends this turn's
    # own message to history before routing here — what matters for "is
    # there anything to reformulate against" is whatever came BEFORE that,
    # so it's excluded rather than counted as prior context.
    history = state.get("history") or []
    prior_history = history[:-1]
    if not prior_history:
        # Nothing to be a follow-up TO yet — the first message of a session
        # is always already standalone; skip the LLM call entirely.
        return user_message

    prompts = reformulate_prompts.active()
    system = prompts.system_prompt(state.get("history_summary"), prior_history)
    with llm_call_span("reformulate_question", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, user_message, ReformulatedQuestion, 200)
    return out.question
