"""Filter-value grounding — the last mile of resolving a natural-language
filter into something Postgres will actually match.

The router/resolver can correctly identify the right table+column for a
filter (app/semantic/resolver.py:SchemaResolver) and still produce a query
that silently returns wrong/empty results if the user's wording doesn't match
what's actually stored (e.g. "UAE" vs. the stored "United Arab Emirates").
`ValueResolver.resolve()` grounds one filter value in two tiers, no
embeddings and no new LLM call:

  1. Controlled categorical dimensions — an exact/case-insensitive lookup
     against that dimension's `value_aliases` map (layer.yaml). This is the
     entire tier: no DB round-trip, so it's the cheap, deterministic path for
     dimensions where the whole value set is small and curated.
  2. High-cardinality dimensions with no `value_aliases` declared (customer
     names, product names, SKUs, ...) — a constrained, parameterized
     `SELECT DISTINCT ... WHERE col ILIKE '%value%' LIMIT N` against the
     already-resolved column, rather than maintaining an alias dictionary
     that could never be complete.

An unresolved value (no alias match, no DB match, or an ambiguous multi-row
DB match) is a resolver-level ambiguity outcome — never silently passed
through as literal filter text, which is exactly the failure mode this
module exists to prevent.

SecurityContext and RLS
------------------------
The DB lookup tier is threaded a `SecurityContext | None` and, when given,
runs through `ConnectionPool.scoped_connection()` — the same RLS-scoped path
every other query in this project uses (see app/db/connection.py) — rather
than an unscoped connection. This matters here specifically: several of this
schema's dimensions (customer_country, order_region, order_channel, ...) sit
on tables that carry region-scoped RLS policies (app/db/policies.sql). An
*unscoped* distinct-value lookup against `customers.country` would return
every country present across every region, including ones a scoped role
(e.g. sales_us) has no authority to see rows from at all — a cross-region
information leak through the value-grounding path even though the guardrail
layer and RLS itself never let that role's *generated SQL* touch those rows.
Threading the caller's SecurityContext through closes that gap: the lookup
sees only the distinct values a row-level-secured connection for that role
would ever be allowed to return.

`security_context=None` is accepted (and left unscoped) only for contexts
where that's genuinely safe — offline/unit tests with a fake `lookup_fn`, or
call sites that already run as the system/admin-equivalent context on
purpose (mirroring app/main.py's `_SYSTEM_CONTEXT` and
app/evaluation/evaluator.py's `EVAL_SECURITY_CONTEXT`, both of which
deliberately bypass RLS for their own already-documented reasons). Production
pipeline wiring (app/text2sql/pipeline.py) always forwards the request's real
SecurityContext.
"""
from dataclasses import dataclass, field

from ..db.connection import ConnectionPool, SecurityContext
from ..db.masking import resolve_table_name
from .resolver import Dimension

DEFAULT_LOOKUP_LIMIT = 10


@dataclass
class ResolvedValue:
    resolved: bool
    value: str = ""
    method: str = ""              # "alias" | "db_lookup" | "unresolved"
    candidates: list[str] = field(default_factory=list)  # populated on an ambiguous DB match


async def _default_lookup_fn(
    pool: ConnectionPool,
    dimension: Dimension,
    value: str,
    security_context: SecurityContext | None,
    limit: int,
) -> list[str]:
    """Real implementation of the bounded DB lookup — injectable (see
    ValueResolver.__init__) so this is unit-testable with a fake and no live
    Postgres, matching this repo's established DI convention
    (app/text2sql/generator.py's `parse_structured`, app/text2sql/pipeline.py's
    `*_fn` parameters)."""
    table, column = dimension.column.split(".", 1)
    # A non-admin role has no grant on some base tables at all (e.g.
    # `customers` — see app/db/policies.sql section 4 and
    # app/db/masking.py) — redirect to the view it's actually allowed to
    # read before this ever reaches Postgres, or the lookup itself fails
    # with a permission error for column/value combinations the masked view
    # exposes identically (name, country, segment; only email is redacted).
    table = resolve_table_name(table, security_context)
    # Cast to text before ILIKE: several dimensions resolve to a non-text
    # column (order_id, rating, campaign_id, …), and Postgres has no ILIKE
    # operator for integer/numeric types — this lookup needs to work for any
    # dimension's stored type, not just the string ones.
    sql = f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}"::text ILIKE $1 LIMIT $2'
    async with pool.scoped_connection(security_context) as conn:
        rows = await conn.fetch(sql, f"%{value}%", limit)
    return [r[0] for r in rows if r[0] is not None]


class ValueResolver:
    def __init__(self, pool: ConnectionPool | None = None, lookup_fn=None,
                 lookup_limit: int = DEFAULT_LOOKUP_LIMIT):
        self.pool = pool
        self.lookup_fn = lookup_fn or _default_lookup_fn
        self.lookup_limit = lookup_limit

    async def resolve(
        self,
        dimension: Dimension,
        value: str,
        security_context: SecurityContext | None = None,
    ) -> ResolvedValue:
        stripped = value.strip()

        # Tier 1: controlled categorical dimensions — exact/case-insensitive
        # lookup against the curated alias map. Also accepts a value that
        # already matches one of the *stored* (right-hand-side) values
        # case-insensitively, so a user typing the canonical value verbatim
        # doesn't fall through to a DB round-trip unnecessarily.
        for alias, stored in dimension.value_aliases.items():
            if alias.strip().lower() == stripped.lower():
                return ResolvedValue(resolved=True, value=stored, method="alias")
        for stored in dimension.value_aliases.values():
            if stored.strip().lower() == stripped.lower():
                return ResolvedValue(resolved=True, value=stored, method="alias")

        if dimension.value_aliases:
            # A controlled dimension declares its full known vocabulary via
            # value_aliases; a value that matches none of it is genuinely
            # unresolved rather than something to guess at via a DB scan.
            return ResolvedValue(resolved=False, method="unresolved")

        # Tier 2: high-cardinality dimension, no declared aliases — bounded,
        # parameterized DB lookup.
        if self.pool is None:
            return ResolvedValue(resolved=False, method="unresolved")
        matches = await self.lookup_fn(
            self.pool, dimension, stripped, security_context, self.lookup_limit,
        )
        if len(matches) == 1:
            return ResolvedValue(resolved=True, value=matches[0], method="db_lookup")
        if len(matches) > 1:
            return ResolvedValue(resolved=False, method="db_lookup", candidates=matches)
        return ResolvedValue(resolved=False, method="unresolved")
