"""SQL repair — on a Postgres execution error, sanitize the error message,
feed it back to the LLM along with the original question + linked schema,
regenerate, revalidate through guardrails, and retry at most `max_attempts`
times before giving up.

A repair attempt is never allowed to skip the guardrail check that the first
attempt went through — a regenerated query is just as capable of drifting
into a blocked shape (e.g. inventing a DELETE, or an unbounded query) as the
original one.
"""
from dataclasses import dataclass

from .. import llm
from ..db.executor import ExecutionResult
from ..guardrails import sql_validator
from ..observability.tracing import llm_call_span
from ..prompts import sql_generation as sql_generation_prompts
from .schema_linker import LinkedSchema


@dataclass
class RepairResult:
    sql: str
    attempts: int
    succeeded: bool
    final_error: str = ""
    # Mirrors ExecutionResult.error_code so callers (pipeline.py) can tell a
    # cost-guard rejection apart from a genuine give-up-after-max-attempts
    # correctness failure — repair is for correctness, not for arguing a query
    # down to a cheaper shape, so a "COMPLEXITY_REJECTED" here means repair
    # stopped early rather than exhausting max_attempts.
    error_code: str = ""


def _default_generate_fn(
    question: str,
    linked: LinkedSchema,
    fewshots: list[dict],
    failed_sql: str,
    sanitized_error: str,
) -> llm.GeneratedSQL:
    prompts = sql_generation_prompts.active()
    system = prompts.system_prompt(linked, fewshots)
    user = prompts.repair_user_prompt(question, failed_sql, sanitized_error)
    with llm_call_span("repair", prompt_version=prompts.PROMPT_VERSION):
        out = llm._parse_structured(system, user, llm.GeneratedSQL, 2000)
    out.confidence = min(1.0, max(0.0, out.confidence))
    return out


async def repair(
    question: str,
    linked: LinkedSchema,
    fewshots: list[dict],
    failed_sql: str,
    execution_result: ExecutionResult,
    executor_fn,
    max_attempts: int = 2,
    generate_fn=None,
) -> RepairResult:
    generate_fn = generate_fn or _default_generate_fn

    current_sql = failed_sql
    current_error = llm._redact(execution_result.error)

    for attempt in range(1, max_attempts + 1):
        try:
            gen = generate_fn(question, linked, fewshots, current_sql, current_error)
        except llm.LLMError as e:
            current_error = llm._redact(str(e))
            if attempt == max_attempts:
                return RepairResult(current_sql, attempt, False, current_error)
            continue

        if gen.needs_clarification or not gen.sql.strip():
            current_error = "Model could not produce a corrected query (needs_clarification)"
            if attempt == max_attempts:
                return RepairResult(current_sql, attempt, False, current_error)
            continue

        guard = sql_validator.check(gen.sql)
        if not guard.allowed:
            current_sql = gen.sql
            current_error = "Guardrail violation: " + "; ".join(guard.violations)
            if attempt == max_attempts:
                return RepairResult(current_sql, attempt, False, current_error)
            continue

        current_sql = guard.sql
        result = await executor_fn(current_sql)
        if result.ok:
            return RepairResult(current_sql, attempt, True)

        if result.error_code == "COMPLEXITY_REJECTED":
            # Not a correctness error the LLM can fix by regenerating — stop
            # immediately rather than burning remaining attempts on a query
            # shape that was rejected for being too expensive, not wrong.
            return RepairResult(current_sql, attempt, False, result.error, "COMPLEXITY_REJECTED")

        current_error = llm._redact(result.error)
        if attempt == max_attempts:
            return RepairResult(current_sql, attempt, False, current_error)

    return RepairResult(current_sql, max_attempts, False, current_error)
