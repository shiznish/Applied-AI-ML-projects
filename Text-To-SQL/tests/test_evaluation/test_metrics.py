"""Offline unit tests for app/evaluation/metrics.py's per-category rollup.
Cross-checked against the same eval.run_eval.clopper_pearson values verified
in tests/test_stats.py, since metrics.py imports (not reimplements) that
function.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.evaluation.evaluator import CaseOutcome
from app.evaluation.metrics import per_category_accuracy
from eval.run_eval import clopper_pearson


def _outcome(category, passed):
    return CaseOutcome(id="x", category=category, difficulty="easy", passed=passed)


def test_per_category_k_and_n():
    results = [
        _outcome("simple", True), _outcome("simple", True), _outcome("simple", False),
        _outcome("join", True), _outcome("join", False),
    ]
    report = per_category_accuracy(results)

    assert report["simple"]["k"] == 2
    assert report["simple"]["n"] == 3
    assert report["join"]["k"] == 1
    assert report["join"]["n"] == 2


def test_per_category_ci_matches_clopper_pearson():
    results = [_outcome("simple", True)] * 12 + [_outcome("simple", False)] * 2
    report = per_category_accuracy(results)

    expected_lo, expected_hi = clopper_pearson(12, 14)
    lo, hi = report["simple"]["ci_95"]
    assert abs(lo - expected_lo) < 1e-9
    assert abs(hi - expected_hi) < 1e-9
    assert abs(report["simple"]["accuracy"] - 12 / 14) < 1e-9


def test_overall_rollup_combines_every_category():
    results = [
        _outcome("simple", True), _outcome("simple", False),
        _outcome("join", True), _outcome("join", True),
    ]
    report = per_category_accuracy(results)

    assert report["overall"]["k"] == 3
    assert report["overall"]["n"] == 4
    expected_lo, expected_hi = clopper_pearson(3, 4)
    lo, hi = report["overall"]["ci_95"]
    assert abs(lo - expected_lo) < 1e-9
    assert abs(hi - expected_hi) < 1e-9


def test_accepts_plain_dicts_too():
    results = [
        {"category": "temporal", "passed": True},
        {"category": "temporal", "passed": False},
    ]
    report = per_category_accuracy(results)
    assert report["temporal"]["k"] == 1
    assert report["temporal"]["n"] == 2


def test_empty_results_produce_empty_overall():
    report = per_category_accuracy([])
    assert report["overall"] == {"k": 0, "n": 0, "accuracy": None, "ci_95": None}
