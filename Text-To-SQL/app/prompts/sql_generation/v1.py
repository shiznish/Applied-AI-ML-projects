"""Prompt construction for SQL generation, built from a LinkedSchema (the
curated semantic layer) instead of a raw schema dump — the semantic layer's
metrics/dimensions/join path are stated explicitly so the model doesn't have
to reconstruct them. Moved from app/text2sql/prompts.py (same contents);
app/text2sql/generator.py and app/text2sql/sql_repair.py are its callers.
"""
from ...schema_extract import schema_to_markdown
from ...text2sql.schema_linker import (
    LinkedSchema, render_dimensions, render_filters, render_join_path, render_metrics,
)

PROMPT_VERSION = "sql_generation-2026.10.1"


def system_prompt(linked: LinkedSchema, fewshots: list[dict]) -> str:
    schema_md = schema_to_markdown(linked.tables)
    join_section = render_join_path(linked.join_edges)
    metrics_section = render_metrics(linked.metrics)
    dims_section = render_dimensions(linked.dimensions)
    filters_section = render_filters(linked.filters)
    shots = "\n\n".join(f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in fewshots)

    sections = [f"""You are an expert PostgreSQL SQL analyst. Translate the user's natural
language question into a single read-only PostgreSQL SELECT query, using only
the tables below (already narrowed to what's relevant to this question):

{schema_md}

Join path connecting these tables:
{join_section}""".strip()]

    if metrics_section:
        sections.append(metrics_section)
    if dims_section:
        sections.append(dims_section)
    if filters_section:
        sections.append(filters_section)

    sections.append("""Rules:
- SELECT queries only. Never write INSERT/UPDATE/DELETE/DDL.
- Use only the tables, columns, metrics and joins given above.
- Use the pre-defined metric expressions exactly as given — do not invent an
  alternative definition of a metric that's already defined above.
- If the question asks for a RATE, RATIO, or PERCENTAGE of one category out
  of a total (e.g. "cancellation rate", "% of orders that are late") and no
  pre-defined metric above already expresses that ratio, compute it yourself
  as a conditional aggregate over the relevant dimension's column — e.g.
  `COUNT(*) FILTER (WHERE orders.status = 'cancelled')::numeric /
  NULLIF(COUNT(*), 0)` — rather than just grouping by that dimension and
  returning raw counts per category, which answers a different, easier
  question than the one asked.
- Prefer explicit JOINs following the join path shown above.
- Include a sensible LIMIT for queries that could return many rows.
- If the question is genuinely ambiguous (a term maps to multiple plausible
  interpretations not resolved by the metrics above), set
  needs_clarification=true, leave sql empty, and list each interpretation with
  an example query. Only do this for real ambiguity — not for minor wording.
- If the question cannot be answered exactly as asked (e.g. it names a
  condition — a date range, a status, a category — that nothing in the
  tables above can express), that is still needs_clarification=true, but with
  exactly ONE interpretation: the closest query you CAN run after dropping or
  adjusting just the part that doesn't fit. Before claiming any column or
  condition is unavailable, check the schema given above carefully — name
  the specific table/column it's missing from, and only say that if it is
  genuinely absent from what's shown, not because you didn't look. `explanation` and each
  interpretation's `description` serve DIFFERENT purposes and must not
  repeat each other:
    - `explanation`: one sentence naming the specific condition that can't be
      satisfied and why (e.g. "no column here records order date, so this
      can't be filtered to 2025").
    - interpretation `description`: states ONLY what the fallback query
      actually computes, as a plain claim with no caveats or repetition of
      `explanation` — e.g. "Revenue by product across all available orders"
      — and that description must accurately describe what `example_sql`
      computes (never describe a filter the SQL doesn't actually apply).
- Report an honest confidence (0-1) that your SQL answers the question.""")

    if shots:
        sections.append(f"Examples:\n{shots}")

    return "\n\n".join(sections)


def repair_user_prompt(question: str, failed_sql: str, sanitized_error: str) -> str:
    return (
        f"{question}\n\n"
        f"Your previous SQL failed to execute:\n{failed_sql}\n\n"
        f"Database error:\n{sanitized_error}\n\n"
        "Fix the query and return corrected SQL. Use only the tables/columns/"
        "joins/metrics given in the schema above — the error above is the only "
        "thing to fix; do not otherwise change the query's intent."
    )
