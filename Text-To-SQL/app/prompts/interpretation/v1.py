"""Interpretation-choice system prompt — resolves a clarifying reply to one
of the SQL-level interpretations app/text2sql/pipeline.py's generator
offered. Moved from app/agent/interpretation.py; the InterpretationChoice
model and the choose_interpretation() orchestration stay there.
"""
from ...agent.memory import render_transcript
from ..safety import CONVERSATION_SAFETY_PREAMBLE

PROMPT_VERSION = "interpretation-2026.10.1"


def system_prompt(interpretations: list[dict], history: list[dict]) -> str:
    options = "\n".join(
        f"{i}. {interp['description']}" for i, interp in enumerate(interpretations, start=1)
    )
    # No history_summary is threaded in here (unlike merge/intent/reformulate)
    # — choose_interpretation only ever receives a raw `history` list, not
    # full AgentState, and this is the one prompt family the long-term-memory
    # work deliberately left alone (see app/agent/memory.py's module
    # docstring); render_transcript still accepts a bare history list fine
    # with no summary.
    transcript = render_transcript(None, history)
    return f"""{CONVERSATION_SAFETY_PREAMBLE}

Task: the user was offered these numbered options as the closest available
answers to a question the system could not answer exactly as asked:
{options}

{transcript}

The user's latest message is their reply to that offer. Decide which option
it means:
- If it clearly names or narrows to one option (a number, a distinguishing
  phrase drawn from its description), return that option's index.
- If it does not pick a specific option but signals the user wants to
  proceed anyway with the system's best judgment ("fine", "any", "whatever",
  "proceed", "yes", "ok", "sure", "you decide"), return 1 — option 1 is
  always the generator's own best-confidence offer, so there is nothing
  further to ask.
- Return only a 1-based index; it must be one of the numbers listed above."""
