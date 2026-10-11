"""Schema retrieval — select a relevant table subset out of the full catalog
instead of dumping every table into the prompt.

At 13 tables this is already a real feature, not a toy: a question like
"revenue by product category from UAE customers" should retrieve
{orders, order_items, products, customers} (4 tables) out of 13, not all of
them — retrieving from a schema too small to need this would prove nothing.

Two signals, both against the curated semantic layer rather than raw column
names — deliberately, since raw-schema prompting is what this retriever
replaces:
  1. keyword overlap between the question and each table's name/description
  2. metric/dimension names implied by the question (layer.yaml maps a metric
     like "revenue" to its base_table, so asking about "revenue" pulls in
     order_items even if the word "order_items" never appears in the question)
Selected tables are then connected via SemanticLayer.connect_tables so any
join-bridge table gets pulled in too — a join path can't be computed later
against tables that were never retrieved.
"""
import re
from dataclasses import dataclass, field

from ..schema_extract import Table
from ..semantic.resolver import SemanticLayer


@dataclass
class RetrievalResult:
    selected_tables: set[str]
    scores: dict[str, int] = field(default_factory=dict)
    bridge_tables: set[str] = field(default_factory=set)  # pulled in only for connectivity


def _tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words | {w + "s" for w in words} | {w.rstrip("s") for w in words}


def retrieve(
    question: str,
    live_schema: dict[str, Table],
    semantic_layer: SemanticLayer,
    max_tables: int = 6,
) -> RetrievalResult:
    if len(live_schema) <= max_tables:
        return RetrievalResult(selected_tables=set(live_schema), scores={})

    q_tokens = _tokenize(question)

    scores: dict[str, int] = {}
    for name in live_schema:
        desc = semantic_layer.tables.get(name, "")
        table_tokens = _tokenize(name) | _tokenize(desc)
        scores[name] = len(q_tokens & table_tokens)

    for name in semantic_layer.tables_for_terms(_tokenize(question)):
        if name in scores:
            scores[name] += 2  # a metric/dimension hit is a stronger signal than a bare word overlap

    ranked = sorted((n for n, s in scores.items() if s > 0), key=lambda n: -scores[n])
    initial = set(ranked[:max_tables])

    if not initial:
        # Nothing scored — fail toward showing something small rather than
        # nothing, same fallback philosophy as the original keyword filter.
        initial = set(sorted(live_schema)[:max_tables])

    connected, _ = semantic_layer.connect_tables(initial)
    bridges = connected - initial
    final = (initial | bridges)
    if len(final) > max_tables:
        # Keep the highest-scored tables plus whatever bridges are needed to
        # keep them joinable, even if that slightly exceeds max_tables — an
        # unjoinable selection is worse than a marginally larger one.
        keep = set(sorted(initial, key=lambda n: -scores.get(n, 0))[:max_tables]) | bridges
        final = keep

    return RetrievalResult(selected_tables=final, scores=scores, bridge_tables=bridges)
