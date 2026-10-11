"""Routes a fresh top-level message to one of two modes: a new/follow-up DATA
question (the existing merge_intent -> pipeline path, unchanged) or a META
message about the conversation itself — "explain that", "why did you pick
X", "what does this column mean" — which should be answered directly from
what's already in state, not funneled through the metric/dimension
classifier as if it were a new data request.

This closes a real bug: without it, a message like "please explain the
previous answer" got merged against the prior turn's resolved slots like any
other follow-up (nothing in it contradicts them, so they're kept unchanged),
silently re-asking whatever data question had already been resolved instead
of answering what was actually asked. See app/agent/graph.py's module
docstring for where this fits in the graph — only reached for a FRESH
top-level turn; a clarification reply resumes directly inside clarify_node
via LangGraph's interrupt()/Command(resume=...) and never re-enters here.

Both functions are given the real conversation transcript (AgentState's
`history` — see app/agent/state.py), not just the resolved slot summary —
that's what lets a meta question reference something several turns back, and
is the actual fix for this feeling like a slot machine rather than a
conversation: app/agent/merge.py's slot vocabulary is deliberately
catalog-constrained (governed, auditable SQL generation), but classification
and conversational replies have no such constraint and should read like an
ordinary chat.

The system prompts themselves live in app/prompts/intent/ — this module
only orchestrates the LLM call and defines the structured-output contracts.
"""
from pydantic import BaseModel

from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import intent as intent_prompts
from .state import AgentState


class MessageIntent(BaseModel):
    kind: str  # "data_question" | "meta"
    reasoning: str


class ConversationalReply(BaseModel):
    reply: str


def classify_message(
    state: AgentState,
    user_message: str,
    parse_structured=llm._parse_structured,
) -> MessageIntent:
    """Only called when there's already a prior data answer to possibly be
    "meta" about — app/agent/graph.py's classify_message_node skips this
    entirely (always data_question) on a session's very first message."""
    prompts = intent_prompts.active()
    system = prompts.classify_message_prompt(state)
    with llm_call_span("classify_message", prompt_version=prompts.PROMPT_VERSION):
        return parse_structured(system, user_message, MessageIntent, 200)


def respond_conversationally(
    state: AgentState,
    user_message: str,
    parse_structured=llm._parse_structured,
) -> str:
    prompts = intent_prompts.active()
    system = prompts.respond_conversationally_prompt(state)
    with llm_call_span("respond_conversationally", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, user_message, ConversationalReply, 400)
    return out.reply
