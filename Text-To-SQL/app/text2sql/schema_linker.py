"""Schema linking — turns a retrieved table subset into an explicit, inspectable
resolution: which join path connects these tables, and which of the semantic
layer's pre-agreed metrics/dimensions apply.

This is the step that makes join-path selection something you can look at and
verify, instead of something buried inside one LLM call that either gets the
join right or doesn't with no visibility into why.
"""
import re
from dataclasses import dataclass, field, replace

from ..db.connection import SecurityContext
from ..db.masking import MASKED_TABLE_VIEWS, resolve_table_name
from ..schema_extract import Table
from ..semantic.resolver import Dimension, Metric, Relationship, ResolvedFilter, SemanticLayer


@dataclass
class LinkedSchema:
    tables: dict[str, Table]
    join_edges: list[Relationship]
    metrics: list[Metric]
    dimensions: list[Dimension]
    filters: list[ResolvedFilter] = field(default_factory=list)
    unreachable: bool = False  # True if the retrieved tables couldn't all be connected


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def link(
    question: str,
    selected_tables: set[str],
    live_schema: dict[str, Table],
    semantic_layer: SemanticLayer,
    filters: list[ResolvedFilter] | None = None,
    security_context: SecurityContext | None = None,
) -> LinkedSchema:
    connected, edges = semantic_layer.connect_tables(selected_tables)
    unreachable = connected != selected_tables and bool(selected_tables - connected)
    final_tables = connected if not unreachable else selected_tables

    tables = {n: live_schema[n] for n in final_tables if n in live_schema}

    q_tokens = _tokenize(question)
    metrics = [
        m for name, m in semantic_layer.metrics.items()
        if any(t in final_tables for t in m.base_tables) and (
            set(name.split("_")) & q_tokens or any(t in final_tables for t in m.base_tables)
        )
    ]
    dimensions = [
        d for name, d in semantic_layer.dimensions.items()
        if d.table in final_tables
    ]

    tables, edges, metrics, dimensions = _apply_view_masking(
        tables, edges, metrics, dimensions, security_context,
    )

    return LinkedSchema(
        tables=tables, join_edges=edges, metrics=metrics, dimensions=dimensions,
        filters=filters or [], unreachable=unreachable,
    )


def _apply_view_masking(
    tables: dict[str, Table],
    edges: list[Relationship],
    metrics: list[Metric],
    dimensions: list[Dimension],
    security_context: SecurityContext | None,
) -> tuple[dict[str, Table], list[Relationship], list[Metric], list[Dimension]]:
    """Redirect a masked base table to the view a non-admin role actually
    holds a grant on (see app/db/masking.py), consistently across every part
    of the prompt that names a table — the schema section, the join path,
    and the dimension list all have to agree on one identity, or the model
    is shown a schema table it's then told to reference under a different
    name. The physical-to-view rename never changes which columns exist
    (customers_masked is column-identical to customers apart from the
    redacted email), so dimension/metric resolution against name/country/
    segment etc. is unaffected."""
    table_map = {
        physical: view for physical, view in MASKED_TABLE_VIEWS.items()
        if physical in tables and resolve_table_name(physical, security_context) == view
    }
    if not table_map:
        return tables, edges, metrics, dimensions

    new_tables = dict(tables)
    for physical, view in table_map.items():
        new_tables[view] = replace(new_tables.pop(physical), name=view)

    new_edges = [
        replace(e, from_table=table_map.get(e.from_table, e.from_table),
                to_table=table_map.get(e.to_table, e.to_table))
        for e in edges
    ]

    new_metrics = []
    for m in metrics:
        mapped = [table_map.get(t, t) for t in m.base_tables]
        new_metrics.append(replace(m, base_tables=mapped) if mapped != m.base_tables else m)

    new_dimensions = []
    for d in dimensions:
        physical, column = d.column.split(".", 1)
        if physical in table_map:
            new_dimensions.append(replace(d, column=f"{table_map[physical]}.{column}"))
        else:
            new_dimensions.append(d)

    return new_tables, new_edges, new_metrics, new_dimensions


def render_join_path(edges: list[Relationship]) -> str:
    if not edges:
        return "(single table — no join needed)"
    return "\n".join(
        f"- {e.from_table}.{e.from_column} = {e.to_table}.{e.to_column}" for e in edges
    )


def render_metrics(metrics: list[Metric]) -> str:
    if not metrics:
        return ""
    lines = ["Pre-defined metrics (use these expressions exactly — do not re-derive them):"]
    for m in metrics:
        lines.append(f"- {m.name}: {m.expression}  ({m.description})")
    return "\n".join(lines)


def render_dimensions(dimensions: list[Dimension]) -> str:
    if not dimensions:
        return ""
    lines = ["Pre-defined dimensions (use these columns for filtering/grouping):"]
    for d in dimensions:
        lines.append(f"- {d.name}: {d.column}")
    return "\n".join(lines)


def render_filters(filters: list[ResolvedFilter]) -> str:
    """Filters the semantic router + resolver already identified and
    grounded (app/semantic/value_resolver.py) — passed through so generation
    doesn't have to re-derive the filter value from question text."""
    if not filters:
        return ""
    lines = ["Filters already resolved for this question (apply these exactly):"]
    for f in filters:
        lines.append(f"- {f.column} {f.operator} '{f.value}'")
    return "\n".join(lines)
