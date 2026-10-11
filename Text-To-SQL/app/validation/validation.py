"""Hallucination detection & confidence scoring.

Signals combined into one score:
  - syntax/guardrail pass          (implicit — pipeline stops otherwise)
  - back-translation alignment (SQL -> question -> compare with original)
  - result sanity checks (NULL-heavy columns, empty results, dates out of range)
  - schema coverage (do the tables/columns the model claims actually exist?)
  - ambiguity score (did the question contain a term the semantic layer
    flags as ambiguous, that execution proceeded past anyway? — see
    app/semantic/resolver.py's ambiguity handling)

A fifth signal, multi-query agreement (generate a second, independent SQL
query and compare its result to the first), used to be part of this blend —
removed because the cost (a second full LLM generation + execution on every
query) wasn't worth the signal for this project's scope, and it produced a
confusing user-facing result (two SQL queries shown for one question) more
often than it caught a real hallucination the other four signals missed.
`results_agree` below is kept — it's also what eval/run_eval.py uses to
score execution accuracy against the golden reference SQL, an unrelated use.

The model's own self-reported `confidence` (GeneratedSQL.confidence) is
deliberately NOT one of the weighted inputs above — "the model said 0.92" is
not evidence of anything on its own. It is still
threaded through combine_confidence() and surfaced in the returned
`breakdown` (as `model_confidence`, weight 0, included=False) purely as a
diagnostic: how far did the model's self-report diverge from the computed
score?
"""
import datetime
from dataclasses import dataclass, field

from ..db.executor import ExecutionResult

WEIGHTS = {
    "alignment": 0.40,
    "sanity": 0.20,
    "schema_coverage": 0.20,
    "ambiguity": 0.20,
}


@dataclass
class SanityReport:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.passed) + len(self.warnings)
        return 1.0 if total == 0 else len(self.passed) / total


def sanity_check(result: ExecutionResult,
                 data_date_range: tuple[str, str] | None = None) -> SanityReport:
    report = SanityReport()

    if result.row_count == 0:
        report.warnings.append("Query returned 0 rows — possibly an over-restrictive filter or bad JOIN")
    else:
        report.passed.append(f"Returned {result.row_count} row(s)")

    # NULL-heavy columns often indicate a bad LEFT JOIN or wrong key
    if result.rows:
        for i, col in enumerate(result.columns):
            nulls = sum(1 for row in result.rows if row[i] is None)
            frac = nulls / len(result.rows)
            if frac > 0.5:
                report.warnings.append(
                    f"Column '{col}' is {frac:.0%} NULL — possible bad JOIN or missing data"
                )
            else:
                report.passed.append(f"Column '{col}' mostly non-NULL")

    # Dates in results should fall inside the data's actual timespan
    if data_date_range and result.rows:
        lo, hi = data_date_range
        for i, col in enumerate(result.columns):
            vals = [row[i] for row in result.rows if isinstance(row[i], str)]
            date_vals = [v for v in vals if _looks_like_date(v)]
            if date_vals:
                out_of_range = [v for v in date_vals if v[:10] < lo or v[:10] > hi]
                if out_of_range:
                    report.warnings.append(
                        f"Column '{col}' has dates outside data range {lo}..{hi}"
                    )
                else:
                    report.passed.append(f"Column '{col}' dates within data range")
    return report


def _looks_like_date(v: str) -> bool:
    try:
        datetime.date.fromisoformat(v[:10])
        return True
    except (ValueError, TypeError):
        return False


def results_agree(a: ExecutionResult, b: ExecutionResult, float_tol: float = 1e-6) -> bool:
    """Order-insensitive comparison of two result sets with float tolerance."""
    if not (a.ok and b.ok):
        return False
    if len(a.rows) != len(b.rows) or len(a.columns) != len(b.columns):
        return False

    def normalize(rows):
        out = []
        for row in rows:
            norm = []
            for v in row:
                if isinstance(v, float):
                    norm.append(round(v, 6))
                else:
                    norm.append(v)
            out.append(tuple(norm))
        return sorted(out, key=repr)

    for ra, rb in zip(normalize(a.rows), normalize(b.rows)):
        for va, vb in zip(ra, rb):
            if isinstance(va, float) and isinstance(vb, (int, float)):
                if abs(va - float(vb)) > max(float_tol, abs(va) * 1e-6):
                    return False
            elif va != vb:
                return False
    return True


def schema_coverage(tables_used: list[str], columns_used: list[str],
                    schema: dict) -> tuple[float, list[str]]:
    """Fraction of tables/columns the model claims to use that actually exist.
    Anything below 1.0 means the model referenced phantom schema — a strong
    hallucination signal."""
    problems: list[str] = []
    checks = 0
    hits = 0

    known_columns = {
        f"{tname}.{c['name']}"
        for tname, t in schema.items() for c in t["columns"]
    }
    bare_columns = {c["name"] for t in schema.values() for c in t["columns"]}

    for t in tables_used:
        checks += 1
        if t in schema:
            hits += 1
        else:
            problems.append(f"Table '{t}' does not exist")

    for c in columns_used:
        checks += 1
        if ("." in c and c in known_columns) or c in bare_columns:
            hits += 1
        else:
            problems.append(f"Column '{c}' does not exist")

    return (hits / checks if checks else 1.0), problems


def combine_confidence(alignment: float | None, sanity: float,
                       coverage: float, ambiguity: float,
                       model_confidence: float | None = None,
                       ) -> tuple[float, dict]:
    """Weighted blend of the four computed signals; a skipped signal
    (alignment unavailable because back-translation errored, say) redistributes
    its weight over the remaining active ones.

    `model_confidence` (the LLM's own self-reported confidence) is accepted
    for display purposes only — see the module docstring for why it is not
    one of the weighted `signals` below. It is never included in the score."""
    signals = {
        "alignment": alignment,
        "sanity": sanity,
        "schema_coverage": coverage,
        "ambiguity": ambiguity,
    }
    active = {k: v for k, v in signals.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in active)
    score = sum(WEIGHTS[k] * v for k, v in active.items()) / total_weight

    breakdown = {
        k: {"value": (round(v, 3) if v is not None else None),
            "weight": WEIGHTS[k],
            "included": v is not None}
        for k, v in signals.items()
    }
    # Diagnostic-only: never weighted, never counted as "included".
    breakdown["model_confidence"] = {
        "value": round(model_confidence, 3) if model_confidence is not None else None,
        "weight": 0.0,
        "included": False,
    }
    return round(score, 3), breakdown
