"""Loads app/semantic/layer.yaml and exposes it as a queryable graph: which
tables exist, how they relate (for join-path resolution), and which metrics/
dimensions are pre-agreed rather than left for the LLM to reconstruct.

Consumed by:
  - app/semantic/router.py (candidate catalog for the semantic router's
    structured-output call)
  - SchemaResolver below (deterministic mapping from the router's
    RoutedIntent to physical tables/columns)
  - app/text2sql/schema_linker.py (join-path rendering once tables are
    selected)
  - app/text2sql/schema_retriever.py (offline/no-LLM-key fallback retrieval
    only — see that module)
"""
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import yaml

_LAYER_PATH = os.path.join(os.path.dirname(__file__), "layer.yaml")


@dataclass
class Metric:
    name: str
    description: str
    expression: str
    base_tables: list[str]
    synonyms: list[str] = field(default_factory=list)


@dataclass
class Dimension:
    name: str
    column: str  # "table.column" (or, in future, an expression — see layer.yaml)
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    value_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def table(self) -> str:
        return self.column.split(".")[0]


@dataclass
class Domain:
    name: str
    description: str
    metrics: list[str]
    dimensions: list[str]


@dataclass
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class ResolvedFilter:
    dimension: str
    column: str          # "table.column", resolved from the dimension
    operator: str
    value: str            # already value-resolved (see app/semantic/value_resolver.py)


@dataclass
class ResolvedSchema:
    tables: set[str]
    filters: list[ResolvedFilter] = field(default_factory=list)
    bridge_tables: set[str] = field(default_factory=set)


@dataclass
class AmbiguityCheck:
    """Unified shape for both router-level and resolver-level ambiguity — see
    app/text2sql/pipeline.py, which treats either the same way (short-circuit
    to status="ambiguous" before generation runs). `interpretations` matches
    llm.Interpretation.model_dump()'s shape so app/main.py can render either
    source identically."""
    is_ambiguous: bool
    explanation: str = ""
    interpretations: list[dict] = field(default_factory=list)


@dataclass
class SemanticLayer:
    tables: dict[str, str] = field(default_factory=dict)          # name -> description
    relationships: list[Relationship] = field(default_factory=list)
    metrics: dict[str, Metric] = field(default_factory=dict)
    dimensions: dict[str, Dimension] = field(default_factory=dict)
    domains: dict[str, Domain] = field(default_factory=dict)
    time_dimension: str | None = None

    def __post_init__(self):
        self._graph: dict[str, list[Relationship]] = defaultdict(list)
        for r in self.relationships:
            self._graph[r.from_table].append(r)
            self._graph[r.to_table].append(
                Relationship(r.to_table, r.to_column, r.from_table, r.from_column)
            )
        # Reverse index for defensive synonym-fallback lookup in SchemaResolver:
        # synonym text -> names of every metric/dimension that declares it.
        # A synonym owned by 2+ names is exactly the deterministic ambiguity
        # signal (see SchemaResolver._lookup) — built once here rather than
        # rescanned per request.
        self._synonym_owners: dict[str, list[str]] = defaultdict(list)
        for name, m in self.metrics.items():
            for syn in m.synonyms:
                self._synonym_owners[syn.lower()].append(name)
        for name, d in self.dimensions.items():
            for syn in d.synonyms:
                self._synonym_owners[syn.lower()].append(name)

    def join_path(self, start: str, end: str) -> list[Relationship] | None:
        """BFS over the FK graph (undirected) — shortest sequence of joins
        connecting two tables, or None if they're not connected at all."""
        if start == end:
            return []
        visited = {start}
        queue: deque[tuple[str, list[Relationship]]] = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for edge in self._graph.get(node, []):
                if edge.to_table in visited:
                    continue
                new_path = path + [edge]
                if edge.to_table == end:
                    return new_path
                visited.add(edge.to_table)
                queue.append((edge.to_table, new_path))
        return None

    def connect_tables(self, tables: set[str]) -> tuple[set[str], list[Relationship]]:
        """Given a candidate table set, find the join edges connecting them —
        pulling in any intermediate "bridge" table not originally selected but
        required for a valid join path (e.g. customers <-> products has no
        direct FK; the path runs through orders -> order_items).

        Shared by SchemaResolver.resolve() (decides the final table set to
        send to the prompt) and schema_linker (renders the explicit join
        path) so both agree on the same connectivity, computed once. This
        method is unchanged by the semantic-router rework — it already did
        exactly the bridge-table job a "schema graph" needs.
        """
        tables = set(tables)
        if len(tables) <= 1:
            return tables, []
        anchor, *rest = sorted(tables)
        connected = {anchor}
        edges: list[Relationship] = []
        for target in rest:
            if target in connected:
                continue
            path = self.join_path(anchor, target)
            if path is None:
                continue  # disconnected in this schema — caller decides how to handle
            for edge in path:
                if edge not in edges and Relationship(edge.to_table, edge.to_column,
                                                       edge.from_table, edge.from_column) not in edges:
                    edges.append(edge)
                connected.add(edge.from_table)
                connected.add(edge.to_table)
        return connected, edges

    def tables_for_terms(self, tokens: set[str]) -> set[str]:
        """Tables implied by metric/dimension names whose words overlap the
        given token set. Used ONLY by app/text2sql/schema_retriever.py's
        offline/no-LLM-key fallback path — the normal path is the semantic
        router + SchemaResolver below, which never does raw lexical
        matching against question text."""
        hit_tables: set[str] = set()
        for name, metric in self.metrics.items():
            if set(name.split("_")) & tokens:
                hit_tables.update(metric.base_tables)
        for name, dim in self.dimensions.items():
            if set(name.split("_")) & tokens:
                hit_tables.add(dim.table)
        return hit_tables


class SchemaResolver:
    """Deterministic mapping from a semantic router's RoutedIntent to
    physical tables (and, per filter, a physical column) — see
    app/semantic/router.py for how RoutedIntent is produced and
    app/text2sql/pipeline.py for how the two are wired together.

    Domain is NOT the table-selection mechanism here: resolve() only ever
    includes a table because a specifically-resolved metric/dimension/filter
    named it, never because "some table in the same domain" exists. Domain is
    used only to build clarification candidates (via domain_metrics/
    domain_dimensions) when a term can't be resolved.
    """

    def __init__(self, layer: SemanticLayer):
        self.layer = layer

    def _lookup(self, name: str, kind: str) -> tuple[str | None, list[str]]:
        """Resolve one router-supplied name to exactly one canonical
        metric/dimension name.

        Canonical-name lookup is the normal path (the router is expected to
        emit canonical IDs). Synonym lookup is a defensive fallback only, for
        the rare case a canonical ID wasn't emitted.

        Returns (resolved_name, candidates). resolved_name is set only on an
        unambiguous match; otherwise candidates lists every colliding name
        (empty list means zero matches, not one) — both are treated as the
        deterministic ambiguity gate by the caller.
        """
        catalog = self.layer.metrics if kind == "metric" else self.layer.dimensions
        if name in catalog:
            return name, [name]
        owners = self.layer._synonym_owners.get(name.lower(), [])
        owners = [o for o in owners if o in catalog]
        if len(owners) == 1:
            return owners[0], owners
        return None, owners

    def resolve(self, routed) -> ResolvedSchema | AmbiguityCheck:
        """`routed` is an llm.RoutedIntent. Import is deferred to avoid a
        module-level dependency cycle (app.llm has no reason to import
        app.semantic)."""
        required_tables: set[str] = set()
        unresolved: list[str] = []
        collisions: dict[str, list[str]] = {}

        for name in routed.metrics:
            resolved, candidates = self._lookup(name, "metric")
            if resolved is None:
                if candidates:
                    collisions[name] = candidates
                else:
                    unresolved.append(name)
                continue
            required_tables.update(self.layer.metrics[resolved].base_tables)

        for name in routed.dimensions:
            resolved, candidates = self._lookup(name, "dimension")
            if resolved is None:
                if candidates:
                    collisions[name] = candidates
                else:
                    unresolved.append(name)
                continue
            required_tables.add(self.layer.dimensions[resolved].table)

        resolved_filters: list[ResolvedFilter] = []
        for f in routed.filters:
            resolved, candidates = self._lookup(f.dimension, "dimension")
            if resolved is None:
                if candidates:
                    collisions[f.dimension] = candidates
                else:
                    unresolved.append(f.dimension)
                continue
            dim = self.layer.dimensions[resolved]
            required_tables.add(dim.table)
            resolved_filters.append(ResolvedFilter(
                dimension=resolved, column=dim.column, operator=f.operator, value=f.value,
            ))

        if unresolved or collisions:
            return self._ambiguity_for(routed, unresolved, collisions)

        connected, _ = self.layer.connect_tables(required_tables)
        bridges = connected - required_tables
        return ResolvedSchema(
            tables=connected, filters=resolved_filters, bridge_tables=bridges,
        )

    def _ambiguity_for(self, routed, unresolved: list[str],
                        collisions: dict[str, list[str]]) -> AmbiguityCheck:
        interpretations: list[dict] = []
        for term, candidates in collisions.items():
            options = ", ".join(candidates)
            interpretations.append({
                "description": f"\"{term}\" could mean any of: {options}. Which did you intend?",
                "example_sql": "",
            })
        for term in unresolved:
            # Offer the router's own selected domain's concepts as candidate
            # clarification options — never as an automatic broadening of the
            # physical schema (that's the whole point of this gate).
            domain_concepts: list[str] = []
            for domain_name in routed.domains:
                d = self.layer.domains.get(domain_name)
                if d:
                    domain_concepts.extend(d.metrics + d.dimensions)
            options = ", ".join(sorted(set(domain_concepts))) or "no known concept in this domain"
            interpretations.append({
                "description": f"\"{term}\" isn't a recognized business concept here. "
                                f"Did you mean one of: {options}?",
                "example_sql": "",
            })
        explanation = (
            "This question uses a term that maps to more than one business concept, "
            "or one this schema doesn't define — see the options below."
        )
        return AmbiguityCheck(is_ambiguous=True, explanation=explanation,
                               interpretations=interpretations)


def load(path: str = _LAYER_PATH) -> SemanticLayer:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    relationships = [
        Relationship(
            from_table=r["from"].split(".")[0], from_column=r["from"].split(".")[1],
            to_table=r["to"].split(".")[0], to_column=r["to"].split(".")[1],
        )
        for r in raw.get("relationships", [])
    ]
    metrics = {
        name: Metric(name=name, description=m["description"], expression=m["expression"],
                     base_tables=list(m["base_tables"]), synonyms=list(m.get("synonyms", [])))
        for name, m in raw.get("metrics", {}).items()
    }
    dimensions = {
        name: Dimension(name=name, column=d["column"], description=d.get("description", ""),
                         synonyms=list(d.get("synonyms", [])),
                         value_aliases=dict(d.get("value_aliases", {})))
        for name, d in raw.get("dimensions", {}).items()
    }
    domains = {
        name: Domain(name=name, description=dm.get("description", ""),
                     metrics=list(dm.get("metrics", [])), dimensions=list(dm.get("dimensions", [])))
        for name, dm in raw.get("domains", {}).items()
    }
    tables = {name: t.get("description", "") for name, t in raw.get("tables", {}).items()}

    return SemanticLayer(
        tables=tables,
        relationships=relationships,
        metrics=metrics,
        dimensions=dimensions,
        domains=domains,
        time_dimension=raw.get("time_dimension"),
    )


_cached: SemanticLayer | None = None


def get() -> SemanticLayer:
    global _cached
    if _cached is None:
        _cached = load()
    return _cached
