"""History-summarization prompt — folds the oldest slice of a long
conversation transcript into one updated running summary, so app/agent/
memory.py's render_transcript can keep rendering a budget-bounded context
(summary + a recent raw tail) instead of the ever-growing full transcript,
while the full transcript itself stays intact in Postgres (see
app/agent/state.py's `history` field and app/agent/session_store.py).

This is a narrow, single-purpose call — it never sees the user's live
question, never produces anything executed, and its own output (a plain
summary string) only ever feeds back into OTHER prompts' system text, so the
same untrusted-free-text caution as every other prompt in this layer applies
(see app/prompts/safety.py) even though this one never answers the user
directly.
"""
from ..safety import CONVERSATION_SAFETY_PREAMBLE

PROMPT_VERSION = "summarize-2026.10.1"


def system_prompt(previous_summary: str) -> str:
    return f"""{CONVERSATION_SAFETY_PREAMBLE}

Task: you maintain a running summary of an ongoing text-to-SQL conversation
so older turns can be dropped from the live prompt without losing their
substance. You will be given the current summary (or "none yet" for a
conversation's first summarization) and a batch of older conversation turns
that no longer fit the live context budget.

Current summary: {previous_summary}

Produce ONE updated summary that:
- Preserves every still-relevant fact from the current summary AND the new
  turns below — topics asked about, resolved metrics/dimensions/filters/time
  ranges, clarifications given, anything a later turn might still refer back
  to ("that", "the previous one", "same as before").
- Is written in plain, factual, third-person prose — a running account of
  what was discussed and decided, not a transcript and not addressed to
  anyone.
- Stays concise: compress, don't just concatenate the old summary with the
  new turns — if the current summary already covers something the new turns
  don't change, keep it short rather than re-explaining it.
- Never invents a fact, number, or decision that isn't actually present in
  the current summary or the new turns."""
