"""Query-complexity governance — heuristics, not sophisticated cost prediction.

This module does NOT estimate real query cost (that would require statistics,
selectivity estimation, and a proper cost model — none of which live here). It
does two much blunter things, both framed as complexity *heuristics*:

  1. `static_complexity_check(sql, config)` — a pure sqlglot-based check that
     needs no DB connection at all: counts JOINs and subquery nesting depth
     (the same depth-counting approach `sql_validator.py` already uses for its
     own subquery-depth rule) and flags a query above configurable
     join-count/depth thresholds. Runs immediately after generation, before
     anything touches Postgres.

  2. `parse_explain_cost(explain_plan)` / `explain_complexity_check(...)` — a
     pure function over the `EXPLAIN (FORMAT TEXT)` plan text that
     `app/db/executor.py` already captures for auditability. Postgres's own
     planner estimate (`cost=startup..total rows=N` on the plan's top line) is
     parsed out and compared against configurable ceilings. This trusts
     Postgres's own estimates — it is not a second opinion on them.

Neither check replaces `app/db/connection.py`'s `SET LOCAL statement_timeout`,
which remains the actual backstop for anything that slips past both of these
(the planner's estimate can simply be wrong). The point of this module is to
reject *obviously* oversized queries before spending that timeout budget.

`check(sql, explain_plan, config)` combines both: `explain_plan` is optional
because the static check can — and does, in app/text2sql/pipeline.py — run
before execution, while the EXPLAIN-based half only has something to check
once app/db/executor.py has captured a plan.
"""
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..config import GuardrailConfig, settings
from .sql_validator import SQL_DIALECT, _subquery_depth


@dataclass
class ComplexityResult:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    join_count: int = 0
    subquery_depth: int = 0
    estimated_cost: float | None = None
    estimated_rows: int | None = None


def static_complexity_check(sql: str, config: GuardrailConfig | None = None) -> ComplexityResult:
    """Pure sqlglot heuristic — no DB connection required. Counts joins and
    subquery nesting depth against configurable thresholds."""
    config = config or settings.guardrails

    try:
        statements = [s for s in sqlglot.parse(sql, read=SQL_DIALECT) if s is not None]
    except Exception as e:
        return ComplexityResult(False, [f"SQL failed to parse: {e}"])

    if not statements:
        return ComplexityResult(False, ["No SQL statement to analyze"])

    root = statements[0]
    join_count = len(list(root.find_all(exp.Join)))
    depth = _subquery_depth(root)

    violations = []
    if join_count > config.max_join_count:
        violations.append(f"Join count {join_count} exceeds limit of {config.max_join_count}")
    if depth > config.max_subquery_depth:
        violations.append(f"Subquery depth {depth} exceeds limit of {config.max_subquery_depth}")

    return ComplexityResult(
        allowed=not violations, violations=violations,
        join_count=join_count, subquery_depth=depth,
    )


# Matches the `cost=<startup>..<total> rows=<n>` fragment on the top line of
# Postgres's EXPLAIN (FORMAT TEXT) output — this fragment's format is the same
# across plan node types (Seq Scan, Nested Loop, Hash Join, ...), so matching
# only the first line is sufficient and avoids picking up a child node's cost.
_EXPLAIN_COST_RE = re.compile(r"cost=[\d.]+\.\.([\d.]+)\s+rows=(\d+)")


def parse_explain_cost(explain_plan: str) -> tuple[float, int] | None:
    """Parse the estimated total cost and row count off the top line of a
    Postgres `EXPLAIN (FORMAT TEXT)` plan. Returns None if the text is empty
    or doesn't match the expected shape (e.g. a plan format this wasn't
    written against, or no plan at all)."""
    if not explain_plan or not explain_plan.strip():
        return None
    first_line = explain_plan.strip().splitlines()[0]
    match = _EXPLAIN_COST_RE.search(first_line)
    if not match:
        return None
    total_cost = float(match.group(1))
    rows = int(match.group(2))
    return total_cost, rows


def explain_complexity_check(
    explain_plan: str | None, config: GuardrailConfig | None = None
) -> ComplexityResult:
    """Check a captured EXPLAIN plan's estimated cost/rows against configurable
    ceilings. Returns allowed=True (no violations) when there's no plan to
    check, rather than failing closed — an absent/unparseable plan is not
    itself evidence of an expensive query."""
    config = config or settings.guardrails
    parsed = parse_explain_cost(explain_plan or "")
    if parsed is None:
        return ComplexityResult(True, [])

    cost, rows = parsed
    violations = []
    if cost > config.max_query_cost:
        violations.append(f"Estimated query cost {cost} exceeds limit of {config.max_query_cost}")
    if rows > config.max_query_rows:
        violations.append(f"Estimated row count {rows} exceeds limit of {config.max_query_rows}")

    return ComplexityResult(
        allowed=not violations, violations=violations,
        estimated_cost=cost, estimated_rows=rows,
    )


def check(
    sql: str,
    explain_plan: str | None = None,
    config: GuardrailConfig | None = None,
) -> ComplexityResult:
    """Combine the static and (optional) EXPLAIN-based checks. `explain_plan`
    is optional so this is usable both before execution (static only) and
    after (static + EXPLAIN)."""
    config = config or settings.guardrails

    static = static_complexity_check(sql, config)
    if explain_plan is None:
        return static

    explained = explain_complexity_check(explain_plan, config)
    return ComplexityResult(
        allowed=static.allowed and explained.allowed,
        violations=static.violations + explained.violations,
        join_count=static.join_count,
        subquery_depth=static.subquery_depth,
        estimated_cost=explained.estimated_cost,
        estimated_rows=explained.estimated_rows,
    )
