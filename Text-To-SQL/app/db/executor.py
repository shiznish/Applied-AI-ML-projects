"""Sandboxed, async query execution against PostgreSQL.

Ported from the original synchronous DuckDB executor. The two-layer defense is
unchanged in spirit — guardrails block destructive SQL before it gets here, and
the connection itself is scoped (role + statement_timeout) before this ever
runs — but execution is now async (asyncpg) and the timeout is enforced by
Postgres itself via `SET LOCAL statement_timeout` (see app/db/connection.py)
rather than a client-side `threading.Timer` interrupt, since asyncpg has no
equivalent to DuckDB's `interrupt()`.

There is a third layer between the EXPLAIN capture and the actual fetch:
app/guardrails/cost_guard.py's EXPLAIN-based complexity check. It runs here
(rather than only in the pipeline) because this is the one place that already
has the plan text before any work is done — rejecting here means an oversized
query never spends any of the statement_timeout budget at all.
"""
import datetime
import decimal
from dataclasses import dataclass, field

import asyncpg

from ..guardrails import cost_guard
from .connection import ConnectionPool, SecurityContext


@dataclass
class ExecutionResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    elapsed_ms: float = 0.0
    explain_plan: str = ""
    error: str = ""
    error_code: str = ""  # Postgres SQLSTATE, used by text2sql/sql_repair.py


async def execute(
    pool: ConnectionPool,
    sql: str,
    security_context: SecurityContext | None = None,
    max_rows: int = 1000,
) -> ExecutionResult:
    import time

    async with pool.scoped_connection(security_context) as conn:
        try:
            plan_rows = await conn.fetch(f"EXPLAIN (FORMAT TEXT) {sql}")
            explain_plan = "\n".join(r[0] for r in plan_rows)
        except asyncpg.PostgresError as e:
            return ExecutionResult(ok=False, error=str(e), error_code=e.sqlstate or "")

        # Cost guard: the EXPLAIN-based half of app/guardrails/cost_guard.py
        # runs here, right after the plan is captured and before .fetch() is
        # ever called, so an oversized query is rejected before it spends any
        # of the statement_timeout budget set in connection.py. A distinct
        # error_code ("COMPLEXITY_REJECTED") keeps this from being mistaken by
        # sql_repair.py for a correctness error worth regenerating over.
        complexity = cost_guard.explain_complexity_check(explain_plan)
        if not complexity.allowed:
            return ExecutionResult(
                ok=False, explain_plan=explain_plan,
                error="Query rejected by cost guard: " + "; ".join(complexity.violations),
                error_code="COMPLEXITY_REJECTED",
            )

        start = time.perf_counter()
        try:
            stmt = await conn.prepare(sql)
            columns = [a.name for a in stmt.get_attributes()] if stmt.get_attributes() else []
            records = await stmt.fetch(*[], timeout=None)
            records = records[:max_rows]
        except asyncpg.QueryCanceledError:
            return ExecutionResult(
                ok=False, explain_plan=explain_plan,
                error="Query cancelled after exceeding the statement timeout",
                error_code="57014",
            )
        except asyncpg.PostgresError as e:
            return ExecutionResult(
                ok=False, explain_plan=explain_plan, error=str(e),
                error_code=e.sqlstate or "",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

    rows = [[_jsonable(v) for v in rec.values()] for rec in records]
    columns = columns or (list(records[0].keys()) if records else [])
    return ExecutionResult(
        ok=True,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        elapsed_ms=round(elapsed_ms, 2),
        explain_plan=explain_plan,
    )


def _jsonable(v):
    """Convert asyncpg values (Decimal, date, datetime) to JSON-safe types."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v
