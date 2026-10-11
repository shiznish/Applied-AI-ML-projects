"""Long-term conversation memory: the full transcript always stays in
Postgres (see app/agent/state.py's `history` field — `append_history` never
truncates), but what a given LLM call actually RENDERS into its prompt is
budgeted — once the raw transcript would blow past a configurable token
budget, the oldest turns are folded into one running LLM-maintained summary
instead of being shown verbatim, and only a recent tail is still rendered
raw. This is a budget, not a drop: nothing here ever removes anything from
`AgentState.history` itself, only from what a given prompt-render pass
chooses to include.

Three pieces:
- `estimate_tokens`/`build_context`: pure, no-LLM budget bookkeeping — decide
  whether the current history fits, and if not, which turns are "recent"
  (still rendered raw) vs. "older" (need folding into the summary).
- `summarize_node`: a real LangGraph node, inserted ahead of
  classify_message_node in app/agent/graph.py's build_graph(). On every turn
  it re-checks the budget via build_context; most turns it's a no-op (no LLM
  call at all) because the transcript still fits. Only when the budget is
  actually exceeded does it call the small summarization prompt
  (app/prompts/summarize/) and update `history_summary`.
- `render_transcript`: what app/prompts/merge/, app/prompts/intent/, and
  app/prompts/reformulate/ actually call to render `(history_summary,
  history)` into one prompt-ready string — applies the same budget logic so
  the rendered text always matches what summarize_node decided, with no risk
  of the two drifting out of sync.
"""
from pydantic import BaseModel

from .. import llm
from ..config import settings
from ..observability.tracing import llm_call_span
from ..prompts import summarize as summarize_prompts
from .state import AgentState


def estimate_tokens(text: str) -> int:
    """Cheap `len(text) // 4` heuristic — no tokenizer dependency added just
    for a budget trip-wire. This is NOT a billing-accurate token count (real
    tokenizers vary by model/provider and this project talks to several —
    see app/llm.py); it only needs to be good enough, consistently, to decide
    "is this transcript roughly too big for the prompt budget", which a
    rough character-based estimate does fine for."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _render_entry(entry: dict) -> str:
    return f'{entry["role"]}: {entry["content"]}'


def build_context(
    history: list[dict], history_summary: str | None, token_budget: int
) -> tuple[str | list[dict] | None, list[dict]]:
    """Decides how much of `history` fits `token_budget`.

    Returns `(first, recent)`:
    - Under budget: the full transcript's estimated tokens already fit, so
      nothing needs folding — returns `(history_summary, history)` completely
      unchanged (the summary stays available but isn't needed).
    - Over budget: splits `history` into an "older" slice (the turns that no
      longer fit and need folding into the running summary) and a "recent"
      tail that does fit on its own — returns `(older, recent)`. The first
      element is a `list[dict]` in this branch (not a string) — callers tell
      the two cases apart with `isinstance(first, list)`: `summarize_node`
      uses that to know there's something to fold; `render_transcript`
      ignores `first` entirely in this branch and renders the CURRENT
      `history_summary` instead (summarize_node's job is keeping that
      current, not this function's).

    Split logic: walk `history` from the most recent message backwards,
    greedily keeping messages in the "recent" tail while they still fit
    `token_budget`; everything before that cutoff becomes "older". This is
    the simplest policy that satisfies "keep folding the oldest messages into
    the summary until what's left fits the budget" — it always keeps the
    most recent, most contextually relevant turns verbatim, and the
    chronological fold order means a later summarization call is always
    extending the same running summary with the next-oldest batch, never
    re-ordering what it already folded.
    """
    if not history:
        return history_summary, history

    full_text = "\n".join(_render_entry(h) for h in history)
    if estimate_tokens(full_text) <= token_budget:
        return history_summary, history

    recent: list[dict] = []
    used = 0
    cutoff = len(history)
    for i in range(len(history) - 1, -1, -1):
        t = estimate_tokens(_render_entry(history[i]))
        if recent and used + t > token_budget:
            break
        used += t
        recent.insert(0, history[i])
        cutoff = i
    older = history[:cutoff]
    return older, recent


class ConversationSummary(BaseModel):
    summary: str


def _default_summarize(
    previous_summary: str, older_messages: list[dict], parse_structured=llm._parse_structured
) -> str:
    prompts = summarize_prompts.active()
    system = prompts.system_prompt(previous_summary)
    user = "\n".join(_render_entry(m) for m in older_messages)
    with llm_call_span("summarize_history", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, user, ConversationSummary, 400)
    return out.summary


def summarize_node(state: AgentState, config=None, parse_structured=None) -> dict:
    """LangGraph node: decides FOR ITSELF, every turn, whether the budget is
    actually exceeded (via `build_context`) and only calls the LLM when it
    is — a cheap no-op on the overwhelming majority of turns, where the
    transcript still fits. `parse_structured` is the usual DI seam
    (app/agent/graph.py's build_graph forwards its own
    `summarize_parse_structured` kwarg here), defaulting to the real
    app.llm._parse_structured when not overridden.

    Only ever returns a `history_summary` update — `history` itself is never
    touched here, so the full transcript app/agent/session_store.py persists
    is completely unaffected by summarization; only what later prompt-render
    calls choose to show (via `render_transcript` below) changes."""
    history = state.get("history") or []
    history_summary = state.get("history_summary")
    older, _recent = build_context(history, history_summary, settings.agent_history_token_budget)
    if not isinstance(older, list) or not older:
        return {}

    kwargs = {"parse_structured": parse_structured} if parse_structured is not None else {}
    new_summary = _default_summarize(history_summary or "none yet", older, **kwargs)
    return {"history_summary": new_summary}


def render_transcript(history_summary: str | None, history: list[dict]) -> str:
    """Shared rendering used by app/prompts/merge/, app/prompts/intent/
    and app/prompts/reformulate/ so every LLM call in this layer sees the
    conversation the same way. Renders a leading "Summary of earlier
    conversation: ..." line when `history_summary` is set, followed by the
    recent raw turns `build_context` says still fit the configured budget —
    the summary itself is always whatever `summarize_node` most recently
    produced; this function only decides how much of the RAW tail to show
    alongside it, via the identical budget logic, so the two never disagree
    about where the cutoff is."""
    _, recent = build_context(history, history_summary, settings.agent_history_token_budget)

    if not recent and not history_summary:
        return "No prior conversation this session."

    parts = []
    if history_summary:
        parts.append(f"Summary of earlier conversation: {history_summary}")
    if recent:
        lines = [_render_entry(h) for h in recent]
        parts.append("Conversation so far, oldest first:\n" + "\n".join(lines))
    return "\n\n".join(parts)
