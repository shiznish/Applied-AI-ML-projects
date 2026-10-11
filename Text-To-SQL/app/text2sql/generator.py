"""SQL generation from a LinkedSchema (the curated semantic layer) instead of
a raw schema dump.

Mirrors the old app/llm.generate_sql signature but takes a LinkedSchema +
fewshot list and builds the prompt via
app/prompts/sql_generation/. The actual provider call is delegated to
app.llm._parse_structured — this module never talks to a provider directly,
and the call point is injectable so tests can substitute a fake without a
real API key/network access.
"""
from .. import llm
from ..observability.tracing import llm_call_span
from ..prompts import sql_generation as sql_generation_prompts
from .schema_linker import LinkedSchema


def generate_sql(
    question: str,
    linked: LinkedSchema,
    fewshots: list[dict],
    parse_structured=llm._parse_structured,
) -> llm.GeneratedSQL:
    prompts = sql_generation_prompts.active()
    system = prompts.system_prompt(linked, fewshots)
    with llm_call_span("generate_content", prompt_version=prompts.PROMPT_VERSION):
        out = parse_structured(system, question, llm.GeneratedSQL, 2000)
    out.confidence = min(1.0, max(0.0, out.confidence))
    return out
