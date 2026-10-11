"""Unit tests for app/semantic/resolver.py's SchemaResolver — the
deterministic mapping from a semantic router's RoutedIntent to physical
tables/columns.

Covers the exact bug this whole rework started from: the word "spend" used
to steer table selection to the wrong metric (campaign_spend) via lexical
substring matching. SchemaResolver never does that kind of matching — these
tests build RoutedIntent objects directly (as a router would produce them)
and confirm resolution is purely canonical-id / curated-synonym driven, with
a collision or an unknown id always producing a clarification, never a
silent guess or an automatic domain-wide broadening.

Fully offline — no live Postgres, no LLM API key. Uses the real
app/semantic/layer.yaml via resolver.get() so this also doubles as a
regression test on the layer file's own shape.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.semantic.resolver import AmbiguityCheck, ResolvedSchema, SchemaResolver, get as get_layer


def _routed(metrics=None, dimensions=None, filters=None, domains=None):
    return llm.RoutedIntent(
        domains=domains or [], metrics=metrics or [], dimensions=dimensions or [],
        filters=filters or [], needs_clarification=False, clarification_reason="",
        interpretations=[],
    )


def _resolver():
    return SchemaResolver(get_layer())


def test_canonical_metric_name_resolves_to_its_base_tables():
    result = _resolver().resolve(_routed(metrics=["revenue"]))
    assert isinstance(result, ResolvedSchema)
    assert "order_items" in result.tables


def test_canonical_dimension_name_resolves_without_pulling_in_domain_siblings():
    """The core design principle: resolving customer_country must not also
    pull in orders/order_items just because customer_analytics (the domain
    that owns customer_country) also lists revenue-related metrics."""
    result = _resolver().resolve(_routed(dimensions=["customer_country"]))
    assert isinstance(result, ResolvedSchema)
    assert result.tables == {"customers"}


def test_time_dimension_is_declared_and_resolves_to_the_orders_table():
    """Guards the fix for a real bug: a question with a date/time condition
    (e.g. "top products by revenue in 2025") never pulled the orders table —
    and so never its order_date column — into scope at all, because nothing
    in the catalog mapped a time condition to any table. app/agent/graph.py's
    build_routed_intent_from_state now appends get_layer().time_dimension
    whenever a time_range is resolved; this just confirms the catalog side
    of that fix is wired: the designated time dimension exists and resolves
    to orders."""
    layer = get_layer()
    assert layer.time_dimension == "order_date"
    result = _resolver().resolve(_routed(dimensions=[layer.time_dimension]))
    assert isinstance(result, ResolvedSchema)
    assert "orders" in result.tables


def test_unknown_term_is_ambiguity_not_a_silent_domain_broadening():
    """A zero-match term must produce a clarification request, never an
    automatic fallback to every metric/dimension in the router's selected
    domain (the exact failure mode this project's docs warn against)."""
    result = _resolver().resolve(_routed(
        metrics=["totally_unrecognized_metric"], domains=["customer_analytics"],
    ))
    assert isinstance(result, AmbiguityCheck)
    assert result.is_ambiguous
    assert result.interpretations


def test_synonym_shared_by_two_metrics_is_a_collision_not_a_guess():
    """layer.yaml's revenue/campaign_spend both declare synonyms that could
    plausibly collide on a badly-chosen curated phrase; this test forces that
    scenario directly via the resolver's synonym-fallback path rather than
    relying on a specific current phrase in layer.yaml (which could change),
    by monkeypatching the synonym index to simulate exactly that collision."""
    layer = get_layer()
    r = SchemaResolver(layer)
    original = dict(layer._synonym_owners)
    try:
        layer._synonym_owners["ambiguous phrase"] = ["revenue", "campaign_spend"]
        result = r.resolve(_routed(metrics=["ambiguous phrase"]))
        assert isinstance(result, AmbiguityCheck)
        assert result.is_ambiguous
    finally:
        layer._synonym_owners.clear()
        layer._synonym_owners.update(original)


def test_the_original_spend_bug_no_longer_reproduces():
    """Historical regression: "spend" used to lexically match campaign_spend
    via substring overlap on the metric's own name. The router (not tested
    here) is responsible for emitting the *canonical* id "revenue" for a
    customer-spend question; this test confirms that once it does, the
    resolver correctly includes order_items — it does not confirm the
    router's own classification, which is covered by test_router.py."""
    result = _resolver().resolve(_routed(metrics=["revenue"], dimensions=["customer_country"]))
    assert isinstance(result, ResolvedSchema)
    assert {"order_items", "customers"} <= result.tables


def test_filter_resolves_to_grounded_column_not_a_free_form_one():
    result = _resolver().resolve(_routed(
        dimensions=[], filters=[llm.FilterSpec(dimension="customer_country", operator="=", value="United Arab Emirates")],
    ))
    assert isinstance(result, ResolvedSchema)
    assert result.filters[0].column == "customers.country"
    assert result.tables == {"customers"}


def test_bridge_tables_added_via_connect_tables_for_a_two_hop_selection():
    """customers <-> products has no direct FK — the path runs through
    orders -> order_items. connect_tables() (unchanged) must still supply
    that bridge when the resolver picks customer_country + revenue."""
    result = _resolver().resolve(_routed(
        metrics=["revenue"], dimensions=["customer_country", "product_category"],
    ))
    assert isinstance(result, ResolvedSchema)
    assert {"customers", "orders", "order_items", "products"} <= result.tables
