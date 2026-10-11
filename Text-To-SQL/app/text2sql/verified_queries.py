"""Verified Query Retrieval — approved past question->SQL pairs used as
few-shot exemplars for generation, selected by deterministic structured
overlap against the current question's RoutedIntent rather than by domain
equality alone (a question can legitimately span multiple domains) or by
raw text similarity (no vector search / fuzzy matching at this stage — see
app/semantic/router.py's own docstring for why this project stays lexical-
free at the routing layer but doesn't reach for embeddings here either: set
overlap on already-structured fields is enough to find genuinely similar
past questions).

score = 3 * |domain overlap| + 2 * |metric overlap| + 1 * |dimension overlap|

Deliberately does NOT fall back to "most recent learned examples" when every
approved example scores 0 — an irrelevant few-shot can actively degrade
generation quality, so no example is preferable to an unrelated one. This
replaces (not supplements) the old llm._load_fewshots()'s unconditional
"newest 3 learned examples" behavior for the routed path; see
app/text2sql/pipeline.py for how the two fewshot sources (seed examples +
this) are combined.
"""
from .. import llm


def _overlap(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))


def score(routed: "llm.RoutedIntent", example: dict) -> int:
    return (
        3 * _overlap(routed.domains, example.get("domains") or [])
        + 2 * _overlap(routed.metrics, example.get("metrics") or [])
        + 1 * _overlap(routed.dimensions, example.get("dimensions") or [])
    )


def select(routed: "llm.RoutedIntent | None", approved: list[dict], limit: int = 3) -> list[dict]:
    """`approved` is the raw learned-example list as stored by
    llm.add_fewshot() (app/llm.py._load_learned_examples()) — each carries
    question/sql/domains/metrics/dimensions/ts. Returns [] when `routed` is
    None (offline/no-router path — nothing structured to score against) or
    when nothing scores above 0."""
    if routed is None or not approved:
        return []
    scored = [(score(routed, ex), ex) for ex in approved]
    scored = [(s, ex) for s, ex in scored if s > 0]
    scored.sort(key=lambda pair: (-pair[0], -pair[1].get("ts", 0)))
    return [ex for _, ex in scored[:limit]]
