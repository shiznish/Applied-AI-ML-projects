"""Resolves a clarifying reply to one of the SQL-level interpretations
app/text2sql/pipeline.py's generator offered — distinct from app/agent/
merge.py's merge_intent, which resolves a reply against catalog slots
(domain/metric/dimension). A generator-level interpretation is already a
concrete SQL + description pair, not a slot to re-merge, so picking one is a
much narrower problem: which numbered option (or none, defaulting to the
generator's own first/best-confidence one) does this reply mean.

Used only by app/agent/graph.py's clarify_node, and only for a
pending_clarification whose `source` is "pipeline" (one whose `source` is
"merge" — or missing, for state persisted before this field existed — still
goes through merge_intent as before).

The system prompt itself lives in app/prompts/interpretation/ — this
module only orchestrates the LLM call and defines the structured-output
contract.
"""
from pydantic import BaseModel

from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import interpretation as interpretation_prompts


class InterpretationChoice(BaseModel):
    selected_index: int  # 1-based index into the offered interpretations


def choose_interpretation(
    reply: str,
    interpretations: list[dict],
    history: list[dict],
    parse_structured=llm._parse_structured,
) -> int:
    """Returns the 1-based index of the interpretation `reply` selects.
    Defensive against an out-of-range or non-numeric model response (clamps
    to index 1, the same default the prompt itself asks for) so a malformed
    structured-output reply can't crash the resume instead of just picking
    the safest default."""
    prompts = interpretation_prompts.active()
    system = prompts.system_prompt(interpretations, history)
    with llm_call_span("choose_interpretation", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, reply, InterpretationChoice, 200)
    idx = out.selected_index
    if not isinstance(idx, int) or idx < 1 or idx > len(interpretations):
        return 1
    return idx
