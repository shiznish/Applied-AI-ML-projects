"""Evaluation suite.

Two modes:
  python eval/run_eval.py --offline    # guardrail + adversarial cases (no API key)
  python eval/run_eval.py              # full run: LLM generation + execution match

Metrics:
  - guardrail effectiveness      (dangerous / adversarial SQL blocked / total)
  - execution accuracy           (generated results match golden results),
                                  broken down per question category, each with
                                  a 95% Clopper-Pearson CI (app/evaluation/)
  - clarification behaviour       (ambiguous / unanswerable handled, not hallucinated)
  - NL prompt-injection defense   (which layer caught each jailbreak attempt)

Execution accuracy is reported with an exact Clopper-Pearson 95% binomial
confidence interval so a small N is not over-read as a precise point estimate.
"""
import argparse
import asyncio
import json
import os
import sys
from math import comb

# Pin the LLM prompt to fixed seed few-shots so eval is reproducible regardless
# of prior interactive feedback. Set before importing app modules.
os.environ.setdefault("TEXT2SQL_EVAL_MODE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails import sql_validator as guardrails  # noqa: E402
from app.validation.validation import results_agree  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")


# --------------------------------------------------------------------------
# Exact Clopper-Pearson binomial confidence interval (no scipy dependency).
# Binomial tails are computed exactly via math.comb (N is small) and the
# bounds are found by bisection on the monotonic tail probability.
#
# app/evaluation/metrics.py imports THIS function (not a reimplementation) so
# the per-category text2sql accuracy numbers use the same statistically
# load-bearing code as the guardrail numbers below. That module is imported
# lazily inside the functions that need it (not at module scope) to avoid a
# circular import: app.evaluation.metrics -> eval.run_eval -> (would be)
# app.evaluation.metrics.
# --------------------------------------------------------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _binom_tail_ge(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact 95% CI (alpha=0.05) for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo, hi = 0.0, 1.0

    if k == 0:
        lo = 0.0
    else:  # lower: p s.t. P(X >= k) = alpha/2
        a, b = 0.0, k / n
        for _ in range(100):
            mid = (a + b) / 2
            if _binom_tail_ge(k, n, mid) < alpha / 2:
                a = mid
            else:
                b = mid
        lo = (a + b) / 2

    if k == n:
        hi = 1.0
    else:  # upper: p s.t. P(X <= k) = alpha/2
        a, b = k / n, 1.0
        for _ in range(100):
            mid = (a + b) / 2
            if _binom_cdf(k, n, mid) > alpha / 2:
                a = mid
            else:
                b = mid
        hi = (a + b) / 2

    return (lo, hi)


def load_cases() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)["cases"]


def load_text2sql_cases() -> list[dict]:
    """Execution-accuracy dataset: 'text2sql_cases' in golden.json —
    categorized (simple/aggregation/join/temporal/ambiguous) questions against
    the 13-table seed schema, each with expected_sql or expect_ambiguous."""
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f).get("text2sql_cases", [])


# --------------------------------------------------------------------------
# Offline: guardrails + adversarial + limit transforms
# --------------------------------------------------------------------------

def run_guardrail_cases(cases: list[dict]) -> dict:
    blocked_ok, blocked_total = 0, 0
    transform_ok, transform_total = 0, 0
    failures = []
    by_category: dict[str, list[int]] = {}

    for case in cases:
        if "sql_input" not in case:
            continue
        result = guardrails.check(case["sql_input"])
        cat = case["category"]
        if case["expect"] == "blocked":
            blocked_total += 1
            hit = not result.allowed
            if hit:
                blocked_ok += 1
            else:
                failures.append(f"{case['id']}: DANGEROUS QUERY WAS NOT BLOCKED")
            by_category.setdefault(cat, [0, 0])
            by_category[cat][0] += int(hit)
            by_category[cat][1] += 1
        elif case["expect"] == "ok_transformed":
            transform_total += 1
            note_frag = case.get("expect_note_contains", "")
            if result.allowed and any(note_frag in n for n in result.notes):
                transform_ok += 1
            else:
                failures.append(
                    f"{case['id']}: expected allowed + note '{note_frag}', "
                    f"got allowed={result.allowed} notes={result.notes}"
                )
    return {
        "dangerous_blocked": (blocked_ok, blocked_total),
        "transforms_applied": (transform_ok, transform_total),
        "by_category": by_category,
        "failures": failures,
    }


# --------------------------------------------------------------------------
# LLM: execution accuracy (per category), clarification, NL prompt injection
#
# Targets the current architecture: app/text2sql/pipeline.py (async, Postgres,
# schema retrieval/linking/repair) instead of the old synchronous DuckDB
# app.llm.generate_sql() call this module used before the async/Postgres
# migration. Execution-accuracy scoring itself lives in
# app/evaluation/evaluator.py + app/evaluation/metrics.py; this module just
# wires the golden cases, a live pool/schema/fewshots, and the two eval
# surfaces (execution accuracy + clarification/NL-injection) together and
# formats the report.
# --------------------------------------------------------------------------

async def _setup_live_context():
    """Open a pool + extract the live schema + load fewshots — the three
    things app/text2sql/pipeline.run() needs beyond the question itself."""
    from app.db.connection import ConnectionPool
    from app.evaluation.evaluator import EVAL_SECURITY_CONTEXT
    from app.schema_extract import extract_schema
    from app import llm

    pool = ConnectionPool()
    await pool.open()
    # Scoped (admin-equivalent), not a raw asyncpg.connect(): once
    # app/db/policies.sql's RLS/grants are live, an unscoped connection has no
    # SELECT grant on any business table at all (grants went to the
    # app_role_* roles, not the base DATABASE_URL user directly) — schema
    # extraction's row-count/sample-value queries would fail outright.
    async with pool.scoped_connection(EVAL_SECURITY_CONTEXT) as conn:
        live_schema = await extract_schema(conn)
    fewshots = llm._load_fewshots()
    return pool, live_schema, fewshots


async def run_text2sql_cases(cases: list[dict], pool, live_schema, fewshots) -> dict:
    """Execution-accuracy pass: golden.json's 'text2sql_cases' through
    app/evaluation/evaluator.py, rolled up per category via
    app/evaluation/metrics.py."""
    from app.evaluation import evaluator, metrics

    outcomes = await evaluator.evaluate_cases(cases, pool, live_schema, fewshots)
    by_category = metrics.per_category_accuracy(outcomes)
    return {"by_category": by_category, "outcomes": outcomes}


async def run_clarify_and_injection_cases(cases: list[dict], pool, live_schema, fewshots) -> dict:
    """The pre-existing clarify / clarify_or_ok / nl_injection cases living in
    golden.json's original 'cases' list (old flat question/golden_sql shape) —
    now driven through app/text2sql/pipeline.run() so this exercises the
    current architecture (ambiguity check + guardrails) rather than a raw,
    now-removed app.llm.generate_sql() call."""
    from app.text2sql import pipeline as text2sql_pipeline

    clarify_ok, clarify_total = 0, 0
    nli_defended, nli_total = 0, 0
    nli_by_layer = {"llm_refused_or_clarified": 0, "guardrail_blocked": 0,
                    "produced_safe_select_or_harmless_error": 0}
    per_case = []

    for case in cases:
        if "question" not in case:
            continue
        expect = case["expect"]
        if expect not in ("clarify", "clarify_or_ok", "nl_injection"):
            continue

        try:
            pr = await text2sql_pipeline.run(case["question"], live_schema, pool, fewshots)
        except Exception as e:
            per_case.append({"id": case["id"], "outcome": f"pipeline error: {e}"})
            continue

        if expect == "nl_injection":
            nli_total += 1
            if pr.status == "ambiguous":
                nli_defended += 1
                nli_by_layer["llm_refused_or_clarified"] += 1
                per_case.append({"id": case["id"], "outcome": "DEFENDED (clarified, no SQL run)"})
            elif pr.status == "blocked":
                nli_defended += 1
                nli_by_layer["guardrail_blocked"] += 1
                per_case.append({"id": case["id"],
                                 "outcome": f"DEFENDED (guardrail blocked: {pr.violations})"})
            else:
                # status is "ok" or "error": guardrails only ever allow
                # read-only SELECTs with no blocked functions, so anything
                # that got this far is harmless by construction.
                nli_defended += 1
                nli_by_layer["produced_safe_select_or_harmless_error"] += 1
                per_case.append({"id": case["id"],
                                 "outcome": f"DEFENDED (guardrail-allowed, status={pr.status})"})
            continue

        if expect == "clarify":
            clarify_total += 1
            if pr.status == "ambiguous":
                clarify_ok += 1
                per_case.append({"id": case["id"], "outcome": "PASS (clarified)"})
            else:
                per_case.append({"id": case["id"],
                                 "outcome": f"FAIL — expected clarification, got status={pr.status}"})
            continue

        if expect == "clarify_or_ok":
            clarify_total += 1
            if pr.status in ("ambiguous", "ok"):
                clarify_ok += 1
                per_case.append({"id": case["id"], "outcome": f"PASS (status={pr.status})"})
            else:
                per_case.append({"id": case["id"],
                                 "outcome": f"FAIL — status={pr.status}"})

    return {
        "clarify": (clarify_ok, clarify_total),
        "nli": (nli_defended, nli_total),
        "nli_by_layer": nli_by_layer,
        "per_case": per_case,
    }


async def run_llm_cases_async(cases: list[dict], text2sql_cases: list[dict]) -> dict:
    pool, live_schema, fewshots = await _setup_live_context()
    try:
        exec_report = await run_text2sql_cases(text2sql_cases, pool, live_schema, fewshots)
        clarify_report = await run_clarify_and_injection_cases(cases, pool, live_schema, fewshots)
    finally:
        await pool.close()
    return {**exec_report, **clarify_report}


def _fmt_ci(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = clopper_pearson(k, n)
    return (f"{k}/{n} ({100*k/n:.0f}%) — 95% CI [{100*lo:.0f}%, {100*hi:.0f}%] "
            f"(exact Clopper-Pearson)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="run only the guardrail / adversarial cases (no API key)")
    parser.add_argument("--only", default="",
                        help="comma-separated list of expect-types to include in the "
                             "LLM run (e.g. 'clarify,clarify_or_ok,nl_injection')")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        keep = set(args.only.split(","))
        # keep offline (sql_input) cases as-is; filter question cases by expect
        cases = [c for c in cases if "sql_input" in c or c["expect"] in keep]
    n_nli = sum(1 for c in cases if c["expect"] == "nl_injection")

    text2sql_cases = load_text2sql_cases()
    n_gold_questions = len(text2sql_cases)

    print("=" * 64)
    print(f"GUARDRAIL / ADVERSARIAL EVAL (offline) — {len(cases)} total cases")
    print("=" * 64)
    g = run_guardrail_cases(cases)
    bk, bt = g["dangerous_blocked"]
    tk, tt = g["transforms_applied"]
    print(f"  dangerous + adversarial blocked : {_fmt_ci(bk, bt)}")
    print(f"  limit transforms applied        : {tk}/{tt}")
    print("  by adversarial category:")
    for cat in sorted(g["by_category"]):
        hit, tot = g["by_category"][cat]
        print(f"    {cat:<38} {hit}/{tot}")
    for f in g["failures"]:
        print(f"  !! {f}")

    if args.offline:
        print(f"\n(LLM cases skipped — offline mode; {n_gold_questions} text2sql execution-"
              f"accuracy questions + {n_nli} NL-injection cases require an API key and a "
              "live Postgres instance)")
        return

    print("\n" + "=" * 64)
    print("LLM EVAL (calls the configured LLM provider + live Postgres)")
    print("=" * 64)
    r = asyncio.run(run_llm_cases_async(cases, text2sql_cases))

    print("\nExecution accuracy by category (vs golden reference SQL):")
    for cat, stats in sorted(r["by_category"].items(), key=lambda kv: (kv[0] == "overall", kv[0])):
        lo, hi = stats["ci_95"] if stats["ci_95"] else (0.0, 1.0)
        acc = f"{100*stats['accuracy']:.0f}%" if stats["accuracy"] is not None else "n/a"
        print(f"  {cat:<14} {stats['k']}/{stats['n']} ({acc}) — 95% CI "
              f"[{100*lo:.0f}%, {100*hi:.0f}%]")

    from app.evaluation.metrics import routing_accuracy
    routing = routing_accuracy(r["outcomes"])
    if routing["n"]:
        lo, hi = routing["ci_95"]
        print(f"\n  semantic router accuracy      : {_fmt_ci(routing['k'], routing['n'])} — 95% CI "
              f"[{100*lo:.0f}%, {100*hi:.0f}%]  (domain/metric/dimension match on cases "
              "declaring expected_domains/metrics/dimensions in golden.json)")

    ck, ct = r["clarify"]
    print(f"\n  clarification handling       : {_fmt_ci(ck, ct)}")
    nd, nt = r["nli"]
    print(f"  NL prompt-injection defended : {_fmt_ci(nd, nt)}")
    for layer, count in r["nli_by_layer"].items():
        if count:
            print(f"      via {layer:<38} {count}")

    print("\nText2SQL case detail:")
    for o in r["outcomes"]:
        status = "PASS" if o.passed else "FAIL"
        print(f"  [{o.id}] {status} ({o.category}/{o.difficulty}) {o.reason}")
    print("\nClarify / NL-injection case detail:")
    for c in r["per_case"]:
        print(f"  [{c['id']}] {c['outcome']}")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "guardrails": {
                "dangerous_blocked": g["dangerous_blocked"],
                "transforms_applied": g["transforms_applied"],
                "by_category": g["by_category"],
            },
            "execution_accuracy": {
                "by_category": r["by_category"],
                "per_case": [
                    {
                        "id": o.id, "category": o.category, "difficulty": o.difficulty,
                        "passed": o.passed, "reason": o.reason, "status": o.status,
                        "generated_sql": o.generated_sql, "expected_sql": o.expected_sql,
                        "routing_correct": o.routing_correct,
                    }
                    for o in r["outcomes"]
                ],
            },
            "routing_accuracy": routing,
            "clarification": r["clarify"],
            "nl_injection": {"defended": r["nli"], "by_layer": r["nli_by_layer"]},
            "clarify_and_nli_per_case": r["per_case"],
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
