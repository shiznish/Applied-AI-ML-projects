"""Per-category execution-accuracy rollup with exact Clopper-Pearson CIs.

Generalizes eval/run_eval.py's clopper_pearson beyond its original
guardrail-only use so every text2sql question category gets a real
confidence interval instead of a bare percentage — importing it rather than
reimplementing it, since it's the one statistically load-bearing function
this project's numbers depend on and two copies would drift.

Note on import direction: this module imports FROM eval.run_eval (not the
other way around at module scope) to avoid a circular import — eval/run_eval.py
imports this module lazily, inside the functions that need it.
"""
from collections import defaultdict
from typing import Any, Iterable

from eval.run_eval import clopper_pearson


def _category_and_pass(result: Any) -> tuple[str, bool]:
    if isinstance(result, dict):
        return result["category"], bool(result["passed"])
    return result.category, bool(result.passed)


def _stats(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    return {
        "k": k,
        "n": n,
        "accuracy": (k / n) if n else None,
        "ci_95": clopper_pearson(k, n) if n else None,
    }


def per_category_accuracy(results: Iterable[Any]) -> dict[str, dict]:
    """`results` is any iterable of per-case outcomes — either
    app.evaluation.evaluator.CaseOutcome instances or plain dicts with
    "category"/"passed" keys.

    Returns e.g.:
        {
          "simple":      {"k": 11, "n": 12, "accuracy": 0.917, "ci_95": (lo, hi)},
          "aggregation": {"k": 9,  "n": 13, "accuracy": 0.692, "ci_95": (lo, hi)},
          ...
          "overall":     {"k": 48, "n": 60, "accuracy": 0.8,   "ci_95": (lo, hi)},
        }
    "overall" rolls up every category so a README table can be built straight
    from this dict's items (one row per key) without extra bookkeeping.
    """
    buckets: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        category, passed = _category_and_pass(result)
        buckets[category].append(passed)

    out = {category: _stats(flags) for category, flags in buckets.items()}
    all_flags = [flag for flags in buckets.values() for flag in flags]
    out["overall"] = _stats(all_flags)
    return out


def routing_accuracy(results: Iterable[Any]) -> dict:
    """Same shape as per_category_accuracy's "overall" entry, but scored on
    CaseOutcome.routing_correct (whether app/semantic/router.py's
    RoutedIntent matched a case's expected_domains/metrics/dimensions) rather
    than end-to-end execution correctness — a separate signal, see
    app/evaluation/evaluator.py:_score_routing. Cases with routing_correct
    left as None (no expected_* fields declared, or the router never ran)
    are excluded, not counted as failures."""
    flags = [
        bool(r.routing_correct if not isinstance(r, dict) else r.get("routing_correct"))
        for r in results
        if (r.routing_correct if not isinstance(r, dict) else r.get("routing_correct")) is not None
    ]
    return _stats(flags)
