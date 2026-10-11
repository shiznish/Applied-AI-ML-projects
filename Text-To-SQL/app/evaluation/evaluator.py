"""Runs one golden text-to-SQL case through app/text2sql/pipeline.py and
scores it by *execution accuracy* against a reference query — never by SQL
string comparison, since two differently-worded queries can be equally
correct.

Two cooperating comparisons:
  - Normal case (has "expected_sql"): run the pipeline, then execute
    expected_sql against the same pool, then compare the two result sets via
    app/validation/validation.py:results_agree (order-insensitive,
    float-tolerant — the same comparator already used for multi-query
    agreement elsewhere in the project).
  - Ambiguous case (has "expect_ambiguous": true): correctness means the
    pipeline stopped at status == "ambiguous" — either the semantic router's
    own needs_clarification (app/semantic/router.py), the deterministic
    resolver's synonym-collision/unresolved-concept gate
    (app/semantic/resolver.py:SchemaResolver), or the LLM's own
    needs_clarification flag during generation — not an execution match.

A case that also carries expected_domains/expected_metrics/expected_dimensions
is additionally scored for *router accuracy* — whether
app/semantic/router.py's RoutedIntent matched what a human curating the
golden set expected, independent of whether the final SQL also executed
correctly. This is a separate signal from execution accuracy: a case can
have the right SQL via a router near-miss the resolver happened to still
resolve correctly (or vice versa), so both are reported rather than only
the end-to-end result.

Both the pipeline call and the reference-SQL execution are injectable
(`pipeline_run_fn`, `executor_fn`) so this is fully testable with canned
PipelineResult / ExecutionResult fakes, matching the dependency-injection
pattern already used in tests/test_text2sql/ — no live Postgres instance or
LLM API key is needed to verify the scoring logic itself.
"""
from dataclasses import dataclass

from ..db.connection import SecurityContext
from ..db.executor import ExecutionResult, execute as db_execute
from ..text2sql.pipeline import PipelineResult, run as pipeline_run
from ..validation.validation import results_agree

# Evaluation measures whether the generated SQL is CORRECT, not whether RLS
# correctly restricts a given role (that's tested separately, live, in
# tests/test_security/test_rls.py) — so both the generated and reference SQL
# run as this admin-equivalent context, otherwise a case like "top 5 products
# by revenue" would spuriously fail for a non-admin role for a reason that has
# nothing to do with whether the model got the SQL right (RLS silently
# narrowing "orders" to one region on both sides equally wouldn't even be a
# fair fight, since golden reference queries are written unscoped).
EVAL_SECURITY_CONTEXT = SecurityContext(
    username="_eval", role="admin", region="*", permissions=frozenset({"admin:*"})
)


@dataclass
class CaseOutcome:
    """One case's scored result — enough detail to debug a failure without
    re-running anything."""
    id: str
    category: str
    difficulty: str
    passed: bool
    reason: str = ""
    status: str = ""          # pipeline status: "ambiguous" | "blocked" | "error" | "ok"
    generated_sql: str = ""
    expected_sql: str = ""
    # Router-accuracy signal — only populated when the case declares
    # expected_domains/expected_metrics/expected_dimensions in golden.json.
    # None means "not scored for this case", distinct from False ("scored
    # and wrong") — see app/evaluation/metrics.py's routing_accuracy().
    routing_correct: bool | None = None


def _score_routing(case: dict, result: PipelineResult) -> bool | None:
    """None when the case doesn't declare any expected_* routing fields, or
    the router itself never ran (offline/no-LLM-key fallback took over —
    result.routed is None). Domains are checked as "expected is a subset of
    what the router returned" (the router is deliberately told to be
    inclusive on domains); metrics/dimensions are checked for exact set
    equality (the router is told to be precise there) — matching
    app/semantic/router.py's own prompt instructions."""
    expected_domains = case.get("expected_domains")
    expected_metrics = case.get("expected_metrics")
    expected_dimensions = case.get("expected_dimensions")
    if not (expected_domains or expected_metrics or expected_dimensions):
        return None
    if result.routed is None:
        return None

    domains_ok = (set(expected_domains or []) <= set(result.routed.domains))
    metrics_ok = (expected_metrics is None or set(result.routed.metrics) == set(expected_metrics))
    dimensions_ok = (expected_dimensions is None or set(result.routed.dimensions) == set(expected_dimensions))
    return domains_ok and metrics_ok and dimensions_ok


async def evaluate_case(
    case: dict,
    pool,
    live_schema,
    fewshots: list[dict],
    semantic_layer=None,
    pipeline_run_fn=None,
    executor_fn=None,
) -> CaseOutcome:
    """Run `case` (one entry of eval/golden.json's "text2sql_cases") through
    the pipeline and score it.

    `pipeline_run_fn` defaults to app.text2sql.pipeline.run and is called as
    `pipeline_run_fn(question, live_schema, pool, fewshots, semantic_layer=...,
    security_context=...)`. `executor_fn` defaults to app.db.executor.execute
    and is called as `executor_fn(pool, sql, security_context=...)` to run the
    reference query. Both use `EVAL_SECURITY_CONTEXT` (admin-equivalent) —
    see the module-level comment for why evaluation deliberately bypasses RLS
    rather than testing it here.
    """
    run_fn = pipeline_run_fn or pipeline_run
    exec_fn = executor_fn or db_execute

    case_id = case["id"]
    category = case["category"]
    difficulty = case.get("difficulty", "")
    expect_ambiguous = bool(case.get("expect_ambiguous"))
    expected_sql = case.get("expected_sql", "")

    result: PipelineResult = await run_fn(
        case["question"], live_schema, pool, fewshots, semantic_layer=semantic_layer,
        security_context=EVAL_SECURITY_CONTEXT,
    )
    generated_sql = result.sql or (result.generated.sql if result.generated else "")
    routing_correct = _score_routing(case, result)

    if expect_ambiguous:
        passed = result.status == "ambiguous"
        reason = "" if passed else (
            f"expected the pipeline to stop as ambiguous, got status={result.status!r}"
        )
        return CaseOutcome(
            id=case_id, category=category, difficulty=difficulty, passed=passed,
            reason=reason, status=result.status, generated_sql=generated_sql,
            expected_sql=expected_sql, routing_correct=routing_correct,
        )

    if result.status != "ok":
        detail = result.violations or (
            [result.generated.explanation] if result.generated and result.generated.explanation else []
        )
        return CaseOutcome(
            id=case_id, category=category, difficulty=difficulty, passed=False,
            reason=f"pipeline did not reach status 'ok' (got {result.status!r}): {detail}",
            status=result.status, generated_sql=generated_sql, expected_sql=expected_sql,
            routing_correct=routing_correct,
        )

    reference: ExecutionResult = await exec_fn(pool, expected_sql, security_context=EVAL_SECURITY_CONTEXT)
    if not reference.ok:
        return CaseOutcome(
            id=case_id, category=category, difficulty=difficulty, passed=False,
            reason=f"reference (golden) SQL failed to execute: {reference.error}",
            status=result.status, generated_sql=generated_sql, expected_sql=expected_sql,
            routing_correct=routing_correct,
        )

    agree = results_agree(result.execution, reference)
    reason = "" if agree else (
        f"result mismatch: generated {result.execution.row_count} row(s) "
        f"columns={result.execution.columns} vs reference {reference.row_count} "
        f"row(s) columns={reference.columns}"
    )
    return CaseOutcome(
        id=case_id, category=category, difficulty=difficulty, passed=agree, reason=reason,
        status=result.status, generated_sql=generated_sql, expected_sql=expected_sql,
        routing_correct=routing_correct,
    )


async def evaluate_cases(
    cases: list[dict],
    pool,
    live_schema,
    fewshots: list[dict],
    semantic_layer=None,
    pipeline_run_fn=None,
    executor_fn=None,
) -> list[CaseOutcome]:
    """Sequentially evaluate every case. Sequential (not gathered) on purpose —
    keeps LLM-provider rate limits and connection-pool pressure predictable
    for a full run against a real API key/quota."""
    outcomes = []
    for case in cases:
        outcomes.append(await evaluate_case(
            case, pool, live_schema, fewshots, semantic_layer=semantic_layer,
            pipeline_run_fn=pipeline_run_fn, executor_fn=executor_fn,
        ))
    return outcomes
