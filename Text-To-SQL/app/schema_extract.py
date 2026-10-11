"""Automatic schema introspection for the prompt engine (async, PostgreSQL).

Extracts tables, columns + types, PK/FK relationships, and sample values for
low-cardinality text columns. Also provides a lightweight relevance filter so
large schemas don't blow up the prompt — superseded for the LLM prompt itself
by app/text2sql/schema_retriever.py's curated semantic layer, but this module
remains the source of ground truth used to validate the semantic layer
against the live database.

Ported from DuckDB: Postgres has no duckdb_constraints() equivalent, so PK/FK
extraction goes through the standard information_schema constraint views
instead of a single convenience table function.
"""
import re
from dataclasses import dataclass, field

import asyncpg


@dataclass
class Column:
    name: str
    dtype: str
    samples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_keys: list[str]
    foreign_keys: list[str]  # human-readable, e.g. "customer_id -> customers.customer_id"
    row_count: int


_PK_SQL = """
SELECT kcu.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'
"""

_FK_SQL = """
SELECT
    kcu.table_name AS from_table, kcu.column_name AS from_column,
    ccu.table_name AS to_table, ccu.column_name AS to_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
"""


async def extract_schema(conn: asyncpg.Connection, sample_max_distinct: int = 20) -> dict[str, Table]:
    tables: dict[str, Table] = {}

    table_rows = await conn.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    pk_rows = await conn.fetch(_PK_SQL)
    fk_rows = await conn.fetch(_FK_SQL)

    pks_by_table: dict[str, list[str]] = {}
    for row in pk_rows:
        pks_by_table.setdefault(row["table_name"], []).append(row["column_name"])

    fks_by_table: dict[str, list[str]] = {}
    for row in fk_rows:
        fks_by_table.setdefault(row["from_table"], []).append(
            f"{row['from_column']} -> {row['to_table']}.{row['to_column']}"
        )

    for row in table_rows:
        tname = row["table_name"]
        col_rows = await conn.fetch(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1 ORDER BY ordinal_position",
            tname,
        )
        cols = [Column(name=c["column_name"], dtype=c["data_type"]) for c in col_rows]

        for col in cols:
            if "character" not in col.dtype and col.dtype != "text":
                continue
            n_distinct = await conn.fetchval(
                f'SELECT COUNT(DISTINCT "{col.name}") FROM "{tname}"'
            )
            if 0 < n_distinct <= sample_max_distinct:
                sample_rows = await conn.fetch(
                    f'SELECT DISTINCT "{col.name}" FROM "{tname}" '
                    f'WHERE "{col.name}" IS NOT NULL ORDER BY 1 LIMIT {sample_max_distinct}'
                )
                col.samples = [str(r[0]) for r in sample_rows]

        row_count = await conn.fetchval(f'SELECT COUNT(*) FROM "{tname}"')
        tables[tname] = Table(
            tname, cols, pks_by_table.get(tname, []), fks_by_table.get(tname, []), row_count
        )
    return tables


def filter_relevant_tables(tables: dict[str, Table], question: str,
                           always_include_if_at_most: int = 8) -> dict[str, Table]:
    """Keyword-overlap relevance filter. Small schemas are passed through whole;
    for large ones, keep tables whose name/columns overlap the question, plus
    any table reachable via a foreign key from a kept table (JOIN paths).

    This is the original DuckDB-era heuristic, kept as a fallback and as the
    baseline that app/text2sql/schema_retriever.py generalizes: instead of
    scoring against raw column names, it scores against the curated semantic
    layer.
    """
    if len(tables) <= always_include_if_at_most:
        return tables

    tokens = set(re.findall(r"[a-z]+", question.lower()))
    scored: dict[str, int] = {}
    for name, t in tables.items():
        words = set(re.findall(r"[a-z]+", name.lower()))
        for c in t.columns:
            words |= set(re.findall(r"[a-z]+", c.name.lower()))
        expanded = words | {w + "s" for w in words} | {w.rstrip("s") for w in words}
        scored[name] = len(tokens & expanded)

    kept = {n for n, s in scored.items() if s > 0}
    for n in list(kept):
        for fk in tables[n].foreign_keys:
            ref = fk.split("->")[1].strip().split(".")[0]
            kept.add(ref)
    if not kept:
        return tables
    return {n: t for n, t in tables.items() if n in kept}


def schema_to_markdown(tables: dict[str, Table]) -> str:
    """Render the schema as compact markdown for the LLM prompt."""
    parts = []
    for t in tables.values():
        lines = [f"### {t.name}  ({t.row_count} rows)"]
        for c in t.columns:
            tags = []
            if c.name in t.primary_keys:
                tags.append("PK")
            desc = f"- {c.name} {c.dtype}"
            if tags:
                desc += f" [{', '.join(tags)}]"
            if c.samples:
                desc += f"  values: {', '.join(c.samples[:12])}"
            lines.append(desc)
        for fk in t.foreign_keys:
            lines.append(f"- FK: {fk}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def schema_to_dict(tables: dict[str, Table]) -> dict:
    return {
        name: {
            "columns": [{"name": c.name, "type": c.dtype, "samples": c.samples} for c in t.columns],
            "primary_keys": t.primary_keys,
            "foreign_keys": t.foreign_keys,
            "row_count": t.row_count,
        }
        for name, t in tables.items()
    }
