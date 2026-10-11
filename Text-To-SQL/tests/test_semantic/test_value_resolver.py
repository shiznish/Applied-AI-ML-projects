"""Unit tests for app/semantic/value_resolver.py's ValueResolver — filter
value grounding (e.g. "UAE" -> the actually-stored "United Arab Emirates").

The DB lookup tier is injectable (`lookup_fn`, same DI pattern as this
repo's other real-implementation-defaulted parameters) so these tests never
touch a live Postgres instance.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.connection import SecurityContext
from app.semantic.resolver import get as get_layer
from app.semantic.value_resolver import ValueResolver, _default_lookup_fn


def _dimension(name):
    return get_layer().dimensions[name]


def test_alias_tier_resolves_a_curated_abbreviation():
    resolver = ValueResolver()
    result = asyncio.run(resolver.resolve(_dimension("customer_country"), "UAE"))
    assert result.resolved
    assert result.value == "United Arab Emirates"
    assert result.method == "alias"


def test_alias_tier_is_case_insensitive():
    resolver = ValueResolver()
    result = asyncio.run(resolver.resolve(_dimension("customer_country"), "uae"))
    assert result.resolved
    assert result.value == "United Arab Emirates"


def test_value_already_matching_stored_form_resolves_without_db_lookup():
    async def unused_lookup(pool, dimension, value, security_context, limit):
        raise AssertionError("should not hit the DB — value already matches a stored alias target")

    resolver = ValueResolver(pool=object(), lookup_fn=unused_lookup)
    result = asyncio.run(resolver.resolve(_dimension("customer_country"), "United Arab Emirates"))
    assert result.resolved
    assert result.value == "United Arab Emirates"


def test_controlled_dimension_value_not_in_alias_map_is_unresolved():
    """customer_country declares a full curated vocabulary — a value that
    matches none of it should be treated as genuinely unresolved, not sent
    to a DB scan (see value_resolver.py's module docstring)."""
    resolver = ValueResolver()
    result = asyncio.run(resolver.resolve(_dimension("customer_country"), "Narnia"))
    assert not result.resolved


def test_high_cardinality_dimension_uses_bounded_db_lookup():
    captured = {}

    async def fake_lookup(pool, dimension, value, security_context, limit):
        captured["value"] = value
        captured["dimension"] = dimension.name
        return ["Completed"]

    resolver = ValueResolver(pool=object(), lookup_fn=fake_lookup)
    result = asyncio.run(resolver.resolve(_dimension("order_status"), "completed"))
    assert result.resolved
    assert result.value == "Completed"
    assert result.method == "db_lookup"
    assert captured["dimension"] == "order_status"


def test_ambiguous_db_match_returns_candidates_not_a_guess():
    async def fake_lookup(pool, dimension, value, security_context, limit):
        return ["Cancelled", "Cancelled by customer"]

    resolver = ValueResolver(pool=object(), lookup_fn=fake_lookup)
    result = asyncio.run(resolver.resolve(_dimension("order_status"), "cancel"))
    assert not result.resolved
    assert result.candidates == ["Cancelled", "Cancelled by customer"]


def test_no_db_matches_is_unresolved_with_no_candidates():
    async def fake_lookup(pool, dimension, value, security_context, limit):
        return []

    resolver = ValueResolver(pool=object(), lookup_fn=fake_lookup)
    result = asyncio.run(resolver.resolve(_dimension("order_status"), "nonexistent"))
    assert not result.resolved
    assert result.candidates == []


def test_high_cardinality_dimension_with_no_pool_is_unresolved_not_a_crash():
    resolver = ValueResolver(pool=None)
    result = asyncio.run(resolver.resolve(_dimension("order_status"), "completed"))
    assert not result.resolved


class _FakeConn:
    def __init__(self, capture):
        self._capture = capture

    async def fetch(self, sql, *args):
        self._capture["sql"] = sql
        return []


class _FakeScopedConn:
    def __init__(self, capture):
        self._capture = capture

    async def __aenter__(self):
        return _FakeConn(self._capture)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    """Stands in for ConnectionPool — only scoped_connection() is exercised
    by _default_lookup_fn, so nothing else needs faking."""

    def __init__(self, capture):
        self._capture = capture

    def scoped_connection(self, security_context=None):
        return _FakeScopedConn(self._capture)


def _security_context(role: str) -> SecurityContext:
    return SecurityContext(username="u", role=role, region="us", permissions=frozenset())


def test_default_lookup_redirects_masked_table_for_non_admin():
    """customer_segment has no value_aliases, so it hits the Tier-2 DB
    lookup (app/semantic/value_resolver.py's _default_lookup_fn) against
    `customers.segment` — a non-admin role has no grant on the base
    `customers` table at all (app/db/policies.sql), so the lookup must be
    redirected to `customers_masked`, which is column-identical apart from
    the redacted email."""
    capture = {}
    pool = _FakePool(capture)
    asyncio.run(_default_lookup_fn(
        pool, _dimension("customer_segment"), "enterprise",
        _security_context("sales_us"), 10,
    ))
    assert 'FROM "customers_masked"' in capture["sql"]


def test_default_lookup_uses_real_table_for_admin():
    capture = {}
    pool = _FakePool(capture)
    asyncio.run(_default_lookup_fn(
        pool, _dimension("customer_segment"), "enterprise",
        _security_context("admin"), 10,
    ))
    assert 'FROM "customers"' in capture["sql"]


def test_default_lookup_uses_real_table_with_no_security_context():
    capture = {}
    pool = _FakePool(capture)
    asyncio.run(_default_lookup_fn(
        pool, _dimension("customer_segment"), "enterprise", None, 10,
    ))
    assert 'FROM "customers"' in capture["sql"]
