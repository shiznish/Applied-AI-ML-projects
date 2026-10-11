"""Offline unit tests for app/guardrails/cost_guard.py. Everything here is a
pure function over SQL text or hand-written EXPLAIN plan text — no DB
connection or LLM call needed, same testing posture as tests/test_guardrails.py.

Kept as a top-level file (not a tests/test_guardrails/ package) to avoid a
name collision with the existing tests/test_guardrails.py module.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import GuardrailConfig
from app.guardrails import cost_guard


def _config(**overrides) -> GuardrailConfig:
    cfg = GuardrailConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# static_complexity_check
# ---------------------------------------------------------------------------

def test_static_check_allows_query_within_join_limit():
    sql = (
        "SELECT o.id FROM orders o "
        "JOIN customers c ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id"
    )
    result = cost_guard.static_complexity_check(sql, _config(max_join_count=4))
    assert result.allowed, result.violations
    assert result.join_count == 2


def test_static_check_rejects_query_above_join_limit():
    sql = (
        "SELECT o.id FROM orders o "
        "JOIN customers c ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id "
        "JOIN suppliers s ON s.id = p.supplier_id"
    )
    result = cost_guard.static_complexity_check(sql, _config(max_join_count=3))
    assert not result.allowed
    assert result.join_count == 4
    assert any("join" in v.lower() for v in result.violations)


def test_static_check_rejects_query_above_depth_limit():
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM customers) a) b) c"
    result = cost_guard.static_complexity_check(sql, _config(max_subquery_depth=3))
    assert not result.allowed
    assert any("depth" in v.lower() for v in result.violations)


def test_static_check_allows_simple_query():
    result = cost_guard.static_complexity_check("SELECT * FROM customers WHERE id = 1")
    assert result.allowed
    assert result.join_count == 0


def test_static_check_reports_parse_failure():
    result = cost_guard.static_complexity_check("not sql at all ;;;")
    assert not result.allowed
    assert result.violations


# ---------------------------------------------------------------------------
# parse_explain_cost
# ---------------------------------------------------------------------------

SEQ_SCAN_PLAN = "Seq Scan on orders  (cost=0.00..123.45 rows=1000 width=64)"

NESTED_LOOP_PLAN = (
    "Nested Loop  (cost=0.29..847.12 rows=250 width=128)\n"
    "  ->  Seq Scan on customers c  (cost=0.00..12.50 rows=500 width=64)\n"
    "  ->  Index Scan using orders_customer_id_idx on orders o  "
    "(cost=0.29..1.65 rows=1 width=64)\n"
    "        Index Cond: (customer_id = c.id)"
)

HASH_JOIN_PLAN = (
    "Hash Join  (cost=45.50..9821.99 rows=15000 width=96)\n"
    "  Hash Cond: (oi.product_id = p.id)\n"
    "  ->  Seq Scan on order_items oi  (cost=0.00..8500.00 rows=200000 width=32)\n"
    "  ->  Hash  (cost=30.00..30.00 rows=1240 width=64)\n"
    "        ->  Seq Scan on products p  (cost=0.00..30.00 rows=1240 width=64)"
)

AGGREGATE_PLAN = (
    "Aggregate  (cost=200.00..200.01 rows=1 width=8)\n"
    "  ->  Seq Scan on orders  (cost=0.00..180.00 rows=8000 width=0)"
)


def test_parse_explain_cost_seq_scan():
    parsed = cost_guard.parse_explain_cost(SEQ_SCAN_PLAN)
    assert parsed == (123.45, 1000)


def test_parse_explain_cost_nested_loop():
    parsed = cost_guard.parse_explain_cost(NESTED_LOOP_PLAN)
    assert parsed == (847.12, 250)


def test_parse_explain_cost_hash_join():
    parsed = cost_guard.parse_explain_cost(HASH_JOIN_PLAN)
    assert parsed == (9821.99, 15000)


def test_parse_explain_cost_aggregate():
    parsed = cost_guard.parse_explain_cost(AGGREGATE_PLAN)
    assert parsed == (200.01, 1)


def test_parse_explain_cost_empty_returns_none():
    assert cost_guard.parse_explain_cost("") is None
    assert cost_guard.parse_explain_cost(None) is None


def test_parse_explain_cost_unrecognized_text_returns_none():
    assert cost_guard.parse_explain_cost("not an explain plan") is None


# ---------------------------------------------------------------------------
# explain_complexity_check / check (combined)
# ---------------------------------------------------------------------------

def test_explain_check_allows_cheap_plan():
    result = cost_guard.explain_complexity_check(
        SEQ_SCAN_PLAN, _config(max_query_cost=100000, max_query_rows=1000000)
    )
    assert result.allowed
    assert result.estimated_cost == 123.45
    assert result.estimated_rows == 1000


def test_explain_check_rejects_expensive_plan():
    result = cost_guard.explain_complexity_check(
        HASH_JOIN_PLAN, _config(max_query_cost=5000, max_query_rows=1000000)
    )
    assert not result.allowed
    assert any("cost" in v.lower() for v in result.violations)


def test_explain_check_rejects_high_row_estimate():
    result = cost_guard.explain_complexity_check(
        HASH_JOIN_PLAN, _config(max_query_cost=100000, max_query_rows=10000)
    )
    assert not result.allowed
    assert any("row" in v.lower() for v in result.violations)


def test_explain_check_allows_when_no_plan_given():
    result = cost_guard.explain_complexity_check(None)
    assert result.allowed


def test_combined_check_allows_cheap_query_without_plan():
    result = cost_guard.check("SELECT * FROM customers WHERE id = 1")
    assert result.allowed


def test_combined_check_allows_cheap_query_with_cheap_plan():
    result = cost_guard.check(
        "SELECT * FROM orders", SEQ_SCAN_PLAN,
        _config(max_join_count=4, max_query_cost=100000, max_query_rows=1000000),
    )
    assert result.allowed


def test_combined_check_rejects_expensive_query_via_plan():
    result = cost_guard.check(
        "SELECT * FROM order_items oi JOIN products p ON oi.product_id = p.id",
        HASH_JOIN_PLAN,
        _config(max_join_count=4, max_query_cost=5000, max_query_rows=1000000),
    )
    assert not result.allowed
    assert any("cost" in v.lower() for v in result.violations)


def test_combined_check_rejects_on_static_even_without_plan():
    sql = (
        "SELECT o.id FROM orders o "
        "JOIN customers c ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id "
        "JOIN suppliers s ON s.id = p.supplier_id"
    )
    result = cost_guard.check(sql, None, _config(max_join_count=3))
    assert not result.allowed
    assert any("join" in v.lower() for v in result.violations)
