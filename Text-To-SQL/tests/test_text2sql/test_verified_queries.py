"""Unit tests for app/text2sql/verified_queries.py's structured-overlap
scoring — the Verified Query Retrieval mechanism that replaced
llm._load_fewshots()'s old unconditional "newest 3 learned examples"
behavior for the routed path.

Key behaviors under test, matching the reviewed plan exactly:
  - score = 3*domain_overlap + 2*metric_overlap + 1*dimension_overlap
  - top 0-3 by score, ties broken by recency
  - when every example scores 0, select() returns [] — no fallback to
    "most recent" (an irrelevant example is worse than none)
  - routed=None (offline/no-router path) also returns []
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.text2sql import verified_queries


def _routed(domains=None, metrics=None, dimensions=None):
    return llm.RoutedIntent(
        domains=domains or [], metrics=metrics or [], dimensions=dimensions or [],
        filters=[], needs_clarification=False, clarification_reason="", interpretations=[],
    )


def _example(question, domains=None, metrics=None, dimensions=None, ts=0):
    return {
        "question": question, "sql": "SELECT 1", "ts": ts,
        "domains": domains or [], "metrics": metrics or [], "dimensions": dimensions or [],
    }


def test_score_weights_domain_over_metric_over_dimension():
    routed = _routed(domains=["customer_analytics"], metrics=["revenue"], dimensions=["customer_country"])
    domain_only = _example("q1", domains=["customer_analytics"])
    metric_only = _example("q2", metrics=["revenue"])
    dimension_only = _example("q3", dimensions=["customer_country"])
    assert verified_queries.score(routed, domain_only) == 3
    assert verified_queries.score(routed, metric_only) == 2
    assert verified_queries.score(routed, dimension_only) == 1


def test_zero_score_examples_are_excluded_entirely():
    routed = _routed(domains=["customer_analytics"], metrics=["revenue"])
    unrelated = _example("totally unrelated", domains=["returns_analytics"], metrics=["refund_amount"])
    result = verified_queries.select(routed, [unrelated])
    assert result == []


def test_all_zero_scores_returns_empty_not_a_fallback_to_recent():
    """The exact behavior the review insisted on: no example beats an
    irrelevant one — never fall back to 'most recent' when nothing scores."""
    routed = _routed(metrics=["revenue"])
    recent_but_irrelevant = _example("recent question", metrics=["campaign_spend"], ts=99999)
    result = verified_queries.select(routed, [recent_but_irrelevant])
    assert result == []


def test_routed_none_returns_empty():
    assert verified_queries.select(None, [_example("q", metrics=["revenue"])]) == []


def test_top_n_ordered_by_score_then_recency():
    routed = _routed(domains=["customer_analytics"], metrics=["revenue"], dimensions=["customer_country"])
    low = _example("low", dimensions=["customer_country"], ts=1)          # score 1
    high_old = _example("high_old", domains=["customer_analytics"], metrics=["revenue"], ts=1)   # score 5
    high_new = _example("high_new", domains=["customer_analytics"], metrics=["revenue"], ts=2)   # score 5, newer
    mid = _example("mid", metrics=["revenue"], ts=1)                       # score 2

    result = verified_queries.select(routed, [low, high_old, mid, high_new], limit=3)

    assert [ex["question"] for ex in result] == ["high_new", "high_old", "mid"]
