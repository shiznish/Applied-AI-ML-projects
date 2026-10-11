"""Offline unit tests for schema linking against the real semantic layer plus
a hand-built disconnected layer for the unreachable-path edge case.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.connection import SecurityContext
from app.schema_extract import Table
from app.semantic.resolver import Relationship, SemanticLayer
from app.semantic import resolver
from app.text2sql import schema_linker as linker


def _ctx(role: str, region: str = "us") -> SecurityContext:
    return SecurityContext(username="u", role=role, region=region, permissions=frozenset())


def _table(name: str) -> Table:
    return Table(name=name, columns=[], primary_keys=[], foreign_keys=[], row_count=0)


def test_links_directly_connected_pair():
    sl = resolver.get()
    live_schema = {n: _table(n) for n in ["customers", "orders"]}
    linked = linker.link("orders placed by customers", {"customers", "orders"}, live_schema, sl)

    assert not linked.unreachable
    assert set(linked.tables) == {"customers", "orders"}
    assert len(linked.join_edges) == 1
    edge = linked.join_edges[0]
    assert {edge.from_table, edge.to_table} == {"customers", "orders"}


def test_metrics_scoped_to_final_tables():
    sl = resolver.get()
    names = ["orders", "order_items", "products"]
    live_schema = {n: _table(n) for n in names}
    linked = linker.link("total revenue by product", set(names), live_schema, sl)

    metric_names = {m.name for m in linked.metrics}
    assert "revenue" in metric_names
    # refund_amount's base_table (returns) isn't part of this selection.
    assert "refund_amount" not in metric_names


def test_dimensions_scoped_to_final_tables():
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: _table(n) for n in names}
    linked = linker.link("orders by status for customers", set(names), live_schema, sl)

    dim_names = {d.name for d in linked.dimensions}
    assert "order_status" in dim_names
    assert "customer_country" in dim_names
    assert "product_category" not in dim_names


def test_unreachable_tables_flagged_when_no_path_exists():
    # A deliberately disconnected two-island layer: {a, b} and {c, d} with no
    # relationship bridging them — connect_tables can't join across islands,
    # so link() must say so rather than silently dropping one side.
    layer = SemanticLayer(
        tables={"a": "", "b": "", "c": "", "d": ""},
        relationships=[
            Relationship("a", "id", "b", "a_id"),
            Relationship("c", "id", "d", "c_id"),
        ],
    )
    live_schema = {n: _table(n) for n in ["a", "b", "c", "d"]}
    linked = linker.link("a question", {"a", "c"}, live_schema, layer)

    assert linked.unreachable is True


def test_render_helpers_handle_empty_input():
    assert linker.render_join_path([]) == "(single table — no join needed)"
    assert linker.render_metrics([]) == ""
    assert linker.render_dimensions([]) == ""


def test_non_admin_security_context_redirects_customers_to_masked_view():
    """Non-admin roles have no grant on the base `customers` table at all
    (see app/db/policies.sql) — the schema shown to the model, the join
    path, and the dimension list must all consistently reference
    `customers_masked` instead, or the generated SQL will name a table the
    role can't read."""
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: _table(n) for n in names}
    linked = linker.link(
        "orders by customer country", set(names), live_schema, sl,
        security_context=_ctx("sales_us"),
    )

    assert "customers_masked" in linked.tables
    assert "customers" not in linked.tables
    assert linked.tables["customers_masked"].name == "customers_masked"

    edge = linked.join_edges[0]
    assert {edge.from_table, edge.to_table} == {"orders", "customers_masked"}

    dims = {d.name: d.column for d in linked.dimensions}
    assert dims["customer_country"] == "customers_masked.country"


def test_admin_security_context_keeps_real_customers_table():
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: _table(n) for n in names}
    linked = linker.link(
        "orders by customer country", set(names), live_schema, sl,
        security_context=_ctx("admin", region="*"),
    )

    assert "customers" in linked.tables
    assert "customers_masked" not in linked.tables
    dims = {d.name: d.column for d in linked.dimensions}
    assert dims["customer_country"] == "customers.country"


def test_no_security_context_keeps_real_customers_table():
    """security_context=None mirrors the rest of this codebase's convention
    for an internal/system call already running as the admin-equivalent
    context — it must not be treated as a reason to mask."""
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: _table(n) for n in names}
    linked = linker.link("orders by customer country", set(names), live_schema, sl)

    assert "customers" in linked.tables
    assert "customers_masked" not in linked.tables
