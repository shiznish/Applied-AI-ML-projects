"""Orchestrates the text-to-SQL core: semantic routing -> deterministic
resolution -> value grounding -> schema linking -> generation -> guardrails
-> cost guard -> execute -> repair-on-error.

Deliberately scoped to that core — the multi-query-agreement / confidence-blend
logic in app/validation/validation.py stays orchestrated from app/main.py,
since it operates across multiple pipeline runs rather than within one; this
module just gives it a clean, structured result to work from.

The LLM call and DB execution points are injectable so this is unit-testable
without a live Postgres instance or an LLM API key — the defaults are the
real things (app.llm via generator.generate_sql, app.semantic.router.route,
and app.db.executor.execute against the given pool).

Retrieval used to be a keyword-overlap table selection
(app/text2sql/schema_retriever.py) gated behind a flat `ambiguous_terms`
lexical floor (app/text2sql/ambiguity.py, now removed). Both are replaced by
a semantic router (one structured-output LLM call classifying the question
against the curated business catalog) plus a deterministic resolver that
maps that classification to physical tables/columns — see
app/semantic/router.py and app/semantic/resolver.py's module docstrings for
the full reasoning. schema_retriever.py is kept, unchanged, as an explicit
offline/no-LLM-key fallback only (used here when the router call itself
raises llm.LLMError) — it is not the normal path anymore.
"""
from dataclasses import dataclass, field

from .. import llm
from ..db.connection import SecurityContext
from ..db.executor import ExecutionResult, execute as db_execute
from ..guardrails import cost_guard, sql_validator
from ..observability.tracing import traced_span
from ..schema_extract import Table
from ..semantic import resolver
from ..semantic.resolver import AmbiguityCheck, ResolvedFilter, SchemaResolver
from ..semantic.router import route as route_question
from ..semantic.value_resolver import ValueResolver
from . import generator, sql_repair, verified_queries
from .schema_linker import LinkedSchema, link as link_schema
from .schema_retriever import RetrievalResult, retrieve as retrieve_schema
from .sql_repair import RepairResult


@dataclass
class PipelineResult:
    status: str  # "ambiguous" | "blocked" | "complexity_rejected" | "error" | "ok"
    sql: str = ""
    generated: llm.GeneratedSQL | None = None
    execution: ExecutionResult | None = None
    repair: RepairResult | None = None
    routed: llm.RoutedIntent | None = None
    retrieval: RetrievalResult | None = None  # only set on the offline-fallback path
    linked: LinkedSchema | None = None
    ambiguity: AmbiguityCheck | None = None
    violations: list[str] = field(default_factory=list)
    guardrail_notes: list[str] = field(default_factory=list)


def _value_ambiguity(dimension: str, value: str, candidates: list[str]) -> AmbiguityCheck:
    if candidates:
        options = ", ".join(candidates)
        description = f"\"{value}\" for {dimension} matches more than one stored value: {options}. Which did you mean?"
    else:
        description = f"\"{value}\" for {dimension} doesn't match anything in the data."
    return AmbiguityCheck(
        is_ambiguous=True,
        explanation=f"Couldn't confidently match the value \"{value}\" you gave for {dimension}.",
        interpretations=[{"description": description, "example_sql": ""}],
    )


async def run(
    question: str,
    live_schema: dict[str, Table],
    pool,
    fewshots: list[dict],
    semantic_layer=None,
    generate_fn=None,
    executor_fn=None,
    repair_fn=None,
    route_fn=None,
    value_resolver=None,
    security_context: SecurityContext | None = None,
    progress_fn=None,
) -> PipelineResult:
    semantic_layer = semantic_layer or resolver.get()
    generate_fn = generate_fn or generator.generate_sql
    exec_fn = executor_fn or (lambda sql: db_execute(pool, sql, security_context=security_context))
    rep_fn = repair_fn or sql_repair.repair
    route_fn = route_fn or route_question
    val_resolver = value_resolver or ValueResolver(pool=pool)
    schema_resolver = SchemaResolver(semantic_layer)
    # Optional, purely observational: a callable invoked with a short step
    # name (e.g. "linking", "generating") right before that step starts, so
    # a caller holding a live connection to the end user (an SSE stream —
    # see app/routes/query.py's POST /v1/query/stream) can report real
    # progress instead of guessing at a fixed cadence. Defaults to a no-op
    # so every existing caller/test is unaffected.
    report = progress_fn or (lambda step, detail=None: None)

    pipeline_attrs = {}
    if security_context is not None:
        pipeline_attrs["text2sql.rls.region"] = security_context.region

    with traced_span("text2sql.pipeline.run", pipeline_attrs):
        routed: llm.RoutedIntent | None = None
        retrieval: RetrievalResult | None = None
        grounded_filters: list[ResolvedFilter] = []

        # ---- 1. Semantic routing ---------------------------------------------
        # No-LLM-key / provider-error environments fall back to the old
        # keyword-overlap retriever below rather than failing the request —
        # this is what keeps `python eval/run_eval.py --offline` and the unit
        # test suite working without any live API key.
        with traced_span("text2sql.semantic_routing") as span:
            try:
                routed = route_fn(question, semantic_layer)
                span.set_attribute("text2sql.routing.needs_clarification", routed.needs_clarification)
            except llm.LLMError:
                span.set_attribute("text2sql.routing.fallback_to_keyword_retrieval", True)

        if routed is not None and routed.needs_clarification:
            return PipelineResult(
                status="ambiguous", routed=routed,
                ambiguity=AmbiguityCheck(
                    is_ambiguous=True,
                    explanation=routed.clarification_reason,
                    interpretations=[i.model_dump() for i in routed.interpretations],
                ),
            )

        if routed is not None:
            # ---- 2. Deterministic resolution ---------------------------------
            report("resolving")
            with traced_span("text2sql.schema_resolution") as span:
                outcome = schema_resolver.resolve(routed)
                span.set_attribute("text2sql.resolution.ambiguous", isinstance(outcome, AmbiguityCheck))
            if isinstance(outcome, AmbiguityCheck):
                return PipelineResult(status="ambiguous", routed=routed, ambiguity=outcome)

            # ---- 3. Value grounding -------------------------------------------
            with traced_span("text2sql.value_resolution") as span:
                for f in outcome.filters:
                    dim = semantic_layer.dimensions[f.dimension]
                    rv = await val_resolver.resolve(dim, f.value, security_context=security_context)
                    if not rv.resolved:
                        return PipelineResult(
                            status="ambiguous", routed=routed,
                            ambiguity=_value_ambiguity(f.dimension, f.value, rv.candidates),
                        )
                    grounded_filters.append(ResolvedFilter(
                        dimension=f.dimension, column=f.column, operator=f.operator, value=rv.value,
                    ))
                span.set_attribute("text2sql.value_resolution.filters_grounded", len(grounded_filters))

            selected_tables = outcome.tables
            base_fewshots = list(fewshots)
        else:
            # ---- Offline fallback: keyword-overlap retrieval, no filters ------
            with traced_span("text2sql.schema_retrieval_fallback") as span:
                retrieval = retrieve_schema(question, live_schema, semantic_layer)
                span.set_attribute("text2sql.retrieval.tables_selected", len(retrieval.selected_tables))
            selected_tables = retrieval.selected_tables
            base_fewshots = list(fewshots)

        verified = verified_queries.select(routed, llm._load_learned_examples())
        effective_fewshots = base_fewshots + verified

        # ---- 4. Schema linking ----------------------------------------------
        report("linking")
        with traced_span("text2sql.schema_linking"):
            linked = link_schema(question, selected_tables, live_schema, semantic_layer,
                                  filters=grounded_filters, security_context=security_context)

        # ---- 5. Generation ----------------------------------------------------
        # Note: llm.LLMError / LLMAuthError are intentionally NOT caught here —
        # the caller (app/main.py) distinguishes auth failures from other
        # provider errors and maps them to distinct HTTP responses; catching
        # them here would collapse that distinction into an opaque "error"
        # status. (The generate_fn implementation itself opens the GenAI
        # gen_ai.generate_content span — see app/text2sql/generator.py.)
        report("generating")
        with traced_span("text2sql.generation"):
            generated = generate_fn(question, linked, effective_fewshots)

        if generated.needs_clarification:
            # Even a single interpretation is a real choice, not a rubber
            # stamp: silently executing the generator's own best-guess
            # fallback (the prior behavior here) means a dropped date filter,
            # a relaxed condition, or any other gap the generator couldn't
            # satisfy gets presented as an ordinary success, with the caveat
            # buried in small print. Always pausing to ask — one offered
            # option or several — is both more honest and is what app/agent/
            # graph.py's clarify_node now has a real resume path for (see its
            # "pipeline"-sourced pending_clarification branch): a reply that
            # picks or confirms the one interpretation directly EXECUTES that
            # already-shown SQL rather than re-deriving it from scratch, so
            # there is no dead end the way there used to be when a vague
            # "proceed anyway" reply couldn't change anything about how the
            # question got re-resolved.
            return PipelineResult(
                status="ambiguous", generated=generated, routed=routed, retrieval=retrieval,
                linked=linked,
                ambiguity=AmbiguityCheck(
                    is_ambiguous=True, explanation=generated.explanation,
                    interpretations=[i.model_dump() for i in generated.interpretations],
                ),
            )

        # Surfaces the SQL the generator actually produced, before it's
        # known whether guardrails/execution will accept it — lets the UI
        # show what the agent is trying to resolve the question into while
        # the remaining steps (which can take several seconds with no other
        # step boundary in between) are still running, instead of only
        # revealing it once the whole turn finishes.
        report("generated", generated.sql)

        # ---- 6. Guardrails ---------------------------------------------------
        report("guardrails")
        with traced_span("text2sql.guardrail_check") as span:
            guard = sql_validator.check(generated.sql)
            span.set_attribute("text2sql.guardrail.blocked", not guard.allowed)
            span.set_attribute("text2sql.guardrail.violations", len(guard.violations))
        if not guard.allowed:
            return PipelineResult(
                status="blocked", sql=generated.sql, generated=generated, routed=routed,
                retrieval=retrieval, linked=linked, violations=guard.violations,
            )

        # ---- 7. Cost guard (static half) --------------------------------------
        # Runs on the guardrail-transformed SQL (post LIMIT injection/clamp),
        # before anything touches Postgres. This is a shape heuristic only
        # (join count, subquery depth) — the EXPLAIN-based half runs inside
        # db/executor.py, right after it captures the plan and before it calls
        # .fetch(), since that's the only place a real plan exists to check.
        report("cost_guard")
        with traced_span("text2sql.cost_guard") as span:
            complexity = cost_guard.static_complexity_check(guard.sql)
            span.set_attribute("text2sql.cost_guard.rejected", not complexity.allowed)
        if not complexity.allowed:
            return PipelineResult(
                status="complexity_rejected", sql=guard.sql, generated=generated, routed=routed,
                retrieval=retrieval, linked=linked,
                violations=complexity.violations,
                guardrail_notes=guard.notes,
            )

        # ---- 8. Execute ---------------------------------------------------------
        report("executing")
        with traced_span("text2sql.execute"):
            execution = await exec_fn(guard.sql)
        if execution.ok:
            return PipelineResult(
                status="ok", sql=guard.sql, generated=generated, execution=execution, routed=routed,
                retrieval=retrieval, linked=linked,
                guardrail_notes=guard.notes,
            )

        # A cost-guard rejection from the EXPLAIN-based check (db/executor.py)
        # is a complexity verdict, not a correctness error — surface it
        # directly rather than handing it to sql_repair.py, which exists to
        # fix wrong SQL (undefined column, syntax), not to argue a query down
        # to a cheaper shape.
        if execution.error_code == "COMPLEXITY_REJECTED":
            return PipelineResult(
                status="complexity_rejected", sql=guard.sql, generated=generated, routed=routed,
                execution=execution, retrieval=retrieval, linked=linked,
                violations=[execution.error],
            )

        # ---- 9. Repair on error ---------------------------------------------------
        report("repairing")
        with traced_span("text2sql.repair") as span:
            repair_result = await rep_fn(question, linked, effective_fewshots, guard.sql, execution, exec_fn)
            span.set_attribute("text2sql.repair.attempts", repair_result.attempts)
        if repair_result.succeeded:
            final_execution = await exec_fn(repair_result.sql)
            return PipelineResult(
                status="ok", sql=repair_result.sql, generated=generated, execution=final_execution,
                repair=repair_result, routed=routed, retrieval=retrieval, linked=linked,
            )

        if repair_result.error_code == "COMPLEXITY_REJECTED":
            return PipelineResult(
                status="complexity_rejected", sql=repair_result.sql, generated=generated,
                execution=execution, repair=repair_result, routed=routed, retrieval=retrieval,
                linked=linked, violations=[repair_result.final_error],
            )

        return PipelineResult(
            status="error", sql=repair_result.sql, generated=generated, execution=execution,
            repair=repair_result, routed=routed, retrieval=retrieval, linked=linked,
            violations=[repair_result.final_error],
        )


async def run_selected_interpretation(
    sql: str,
    description: str,
    tables_used: list[str],
    columns_used: list[str],
    pool,
    executor_fn=None,
    security_context: SecurityContext | None = None,
    progress_fn=None,
) -> PipelineResult:
    """Executes a SQL interpretation the generator already offered (one entry
    of `GeneratedSQL.interpretations` from a `needs_clarification=true`
    result above) and the user has now confirmed — app/agent/graph.py's
    clarify_node, resuming a pipeline-sourced pending_clarification.

    Deliberately skips routing, schema linking, AND generation: the SQL was
    already produced and shown to the user verbatim as `example_sql`, so
    regenerating it from scratch on resume would risk a different, unreviewed
    query silently replacing the one they actually agreed to. Still runs the
    SAME guardrails -> cost guard -> execute stages as the normal path — a
    confirmed interpretation is not exempt from those. No repair-on-error:
    if the exact SQL the user confirmed fails, that is an honest error to
    surface, not something to silently rewrite into a third query they never
    saw.
    """
    exec_fn = executor_fn or (lambda s: db_execute(pool, s, security_context=security_context))
    report = progress_fn or (lambda step, detail=None: None)
    generated = llm.GeneratedSQL(
        needs_clarification=False, interpretations=[], sql=sql, explanation=description,
        confidence=0.6, tables_used=tables_used, columns_used=columns_used,
    )

    with traced_span("text2sql.guardrail_check") as span:
        guard = sql_validator.check(generated.sql)
        span.set_attribute("text2sql.guardrail.blocked", not guard.allowed)
    if not guard.allowed:
        return PipelineResult(
            status="blocked", sql=generated.sql, generated=generated, violations=guard.violations,
        )

    with traced_span("text2sql.cost_guard") as span:
        complexity = cost_guard.static_complexity_check(guard.sql)
        span.set_attribute("text2sql.cost_guard.rejected", not complexity.allowed)
    if not complexity.allowed:
        return PipelineResult(
            status="complexity_rejected", sql=guard.sql, generated=generated,
            violations=complexity.violations, guardrail_notes=guard.notes,
        )

    report("executing")
    with traced_span("text2sql.execute"):
        execution = await exec_fn(guard.sql)
    if execution.ok:
        return PipelineResult(
            status="ok", sql=guard.sql, generated=generated, execution=execution,
            guardrail_notes=guard.notes,
        )
    if execution.error_code == "COMPLEXITY_REJECTED":
        return PipelineResult(
            status="complexity_rejected", sql=guard.sql, generated=generated, execution=execution,
            violations=[execution.error],
        )
    return PipelineResult(
        status="error", sql=guard.sql, generated=generated, execution=execution,
        violations=[execution.error],
    )
