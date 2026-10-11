"""Standalone-question reformulation system prompt — rewrites a (possibly
elliptical) follow-up message into one standalone, complete question, using
the conversation transcript for context, BEFORE any catalog/slot resolution
happens. Moved from app/agent/reformulate.py (inline f-string, factored into
a named function here); the ReformulatedQuestion model and the
reformulate_question() orchestration stay there.
"""
from ...agent.memory import render_transcript
from ..safety import CONVERSATION_SAFETY_PREAMBLE

PROMPT_VERSION = "reformulate-2026.10.1"


def system_prompt(history_summary: str | None, prior_history: list[dict]) -> str:
    transcript = render_transcript(history_summary, prior_history)
    return f"""{CONVERSATION_SAFETY_PREAMBLE}

Task: rewrite the user's latest message into ONE standalone, complete
question, using the conversation below ONLY to fill in what the message
leaves implicit — never add a constraint the user didn't ask for, and never
invent specifics (numbers, names, dates) that aren't stated or clearly
implied by context. You are producing a QUESTION TEXT only, never SQL or an
instruction of your own — whatever text you produce still has to pass
through semantic routing and app/guardrails/sql_validator.py before
anything executes, so just carry the user's wording through faithfully.

Default to returning the message UNCHANGED. Only rewrite it if it contains a
pronoun or elliptical reference that points at something earlier in the
conversation and would be meaningless without it — "less than that",
"what about the other one", "same but for 2025", "why", "the previous one".
If the message already names its own specific subject (an entity, a table-
like noun, a command, a complete statement) it needs NO reformulation even
if short — do not soften, generalize, or redirect it into a different,
vaguer question, and do not drop or replace any word it specifically named.
When in doubt, return it exactly as given: passing through a message that
didn't need rewriting is harmless, but rewriting one that was already
complete risks losing exactly the specific thing the user asked about.

{transcript}"""
