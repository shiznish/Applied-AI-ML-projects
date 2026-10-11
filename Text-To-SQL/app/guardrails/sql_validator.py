"""Guardrail middleware — every generated query passes through here before execution.

Rules (each configurable via GuardrailConfig):
  1. Must parse as exactly ONE statement.
  2. Whitelist: only SELECT (incl. CTEs / UNION / INTERSECT / EXCEPT). All DDL
     (CREATE/ALTER/DROP), DML writes (INSERT/UPDATE/DELETE), and admin commands
     (COPY/GRANT/SET/...) are blocked.
  3. No file-reading / superuser / info-disclosure functions.
  4. Subquery nesting depth <= max_subquery_depth.
  5. A LIMIT is enforced: if the outer query has none, LIMIT <default_row_limit>
     is injected; an existing larger LIMIT is clamped.
Every blocked query is logged with the failing rule.

Ported from a DuckDB-specific version to PostgreSQL. The blocked-function list is
NOT a mechanical rename — DuckDB's introspection functions (duckdb_secrets(), ...)
don't exist in Postgres, and Postgres has its own equivalent attack surface:
large-object file I/O (lo_import/lo_export), the pg_read_file family (superuser
file access), dblink (cross-database/network access), and session-hijacking via
SET ROLE / SET SESSION AUTHORIZATION (which would defeat the RLS session wiring
this project relies on for access control — see app/db/connection.py).
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..config import GuardrailConfig, settings

logger = logging.getLogger("guardrails")
_BLOCK_LOG = os.path.join(settings.log_dir, "blocked.jsonl")

SQL_DIALECT = "postgres"

# Root node types that are read-only queries
try:
    _QUERY_ROOTS: tuple = (exp.Select, exp.SetOperation)
except AttributeError:  # older sqlglot
    _QUERY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass
class GuardrailResult:
    allowed: bool
    sql: str                       # transformed SQL (LIMIT injected) if allowed
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # non-fatal transforms applied


def _log_block(sql: str, violations: list[str]) -> None:
    record = {"ts": time.time(), "sql": sql, "violations": violations}
    with open(_BLOCK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.warning("Blocked query: %s | %s", violations, sql[:200])


def _subquery_depth(node: exp.Expression) -> int:
    """Maximum nesting depth of SELECTs (outer query = depth 1)."""
    deepest = 0
    for select in node.find_all(exp.Select):
        depth = 1
        parent = select.parent
        while parent is not None:
            if isinstance(parent, exp.Select):
                depth += 1
            parent = parent.parent
        deepest = max(deepest, depth)
    return deepest


def _blocked_function_calls(node: exp.Expression, blocked: frozenset) -> list[str]:
    hits = []
    for func in node.find_all(exp.Func):
        if isinstance(func, exp.Anonymous):
            name = (func.name or "").lower()
        else:
            name = func.sql_name().lower()
        if name in blocked:
            hits.append(name)
    return sorted(set(hits))


def _blocked_identifiers(node: exp.Expression, blocked: frozenset) -> list[str]:
    """Catch blocked names referenced without call syntax, e.g. bare `SET ROLE admin`
    or a table-function reference `FROM pg_read_file(...)` that sqlglot may not
    classify as exp.Func in every dialect edge case."""
    hits = []
    for ident in node.find_all(exp.Column, exp.Table):
        name = (ident.name or "").lower()
        if name in blocked:
            hits.append(name)
    return sorted(set(hits))


def check(sql: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    config = config or settings.guardrails
    violations: list[str] = []
    notes: list[str] = []

    try:
        statements = sqlglot.parse(sql, read=SQL_DIALECT)
    except Exception as e:
        result = GuardrailResult(False, sql, [f"SQL failed to parse: {e}"])
        _log_block(sql, result.violations)
        return result

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        violations.append(f"Expected exactly 1 statement, got {len(statements)}")
        _log_block(sql, violations)
        return GuardrailResult(False, sql, violations)

    root = statements[0]

    if config.allow_only_select and not isinstance(root, _QUERY_ROOTS):
        violations.append(
            f"Only SELECT queries are allowed (got {root.__class__.__name__.upper()})"
        )

    # Defense in depth: no write/DDL/admin node anywhere in the tree (e.g. hidden in a CTE).
    # exp.Command covers statements sqlglot doesn't have a dedicated node for in the
    # postgres dialect (SET, GRANT, REVOKE, COPY, VACUUM, ...) — these parse as
    # exp.Command with the keyword captured as `this`.
    write_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Merge,
                   exp.Alter, exp.Copy, exp.Grant)
    for cls in write_nodes:
        if root.find(cls) is not None:
            violations.append(f"Destructive operation {cls.__name__.upper()} is not allowed")

    for cmd in root.find_all(exp.Command):
        kw = str(cmd.this).upper() if cmd.this else ""
        violations.append(f"Admin/session command {kw or 'COMMAND'} is not allowed")

    blocked_fns = _blocked_function_calls(root, config.blocked_functions)
    blocked_fns += _blocked_identifiers(root, config.blocked_functions)
    blocked_fns = sorted(set(blocked_fns))
    if blocked_fns:
        violations.append(f"Blocked function(s)/identifier(s): {', '.join(blocked_fns)}")

    depth = _subquery_depth(root)
    if depth > config.max_subquery_depth:
        violations.append(
            f"Subquery depth {depth} exceeds limit of {config.max_subquery_depth}"
        )

    if violations:
        _log_block(sql, violations)
        return GuardrailResult(False, sql, violations)

    transformed = root
    limit_expr = root.args.get("limit")
    if limit_expr is None:
        transformed = root.limit(config.default_row_limit)
        notes.append(f"LIMIT {config.default_row_limit} injected")
    else:
        try:
            current = int(limit_expr.expression.this)
            if current > config.default_row_limit:
                transformed = root.limit(config.default_row_limit)
                notes.append(
                    f"LIMIT clamped from {current} to {config.default_row_limit}"
                )
        except (AttributeError, TypeError, ValueError):
            pass  # non-literal LIMIT (rare) — leave as-is, row cap applies at fetch

    return GuardrailResult(True, transformed.sql(dialect=SQL_DIALECT), [], notes)
