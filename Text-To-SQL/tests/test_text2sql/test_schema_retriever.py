"""Offline unit tests for schema retrieval against the real 13-table semantic
layer, using hand-built Table objects (no live Postgres needed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.schema_extract import Table
from app.semantic import resolver
from app.text2sql import schema_retriever as sr


def _table(name: str) -> Table:
    return Table(name=name, columns=[], primary_keys=[], foreign_keys=[], row_count=0)


def test_small_schema_returns_everything():
    sl = resolver.get()
    live_schema = {n: _table(n) for n in ["customers", "orders"]}
    result = sr.retrieve("anything at all", live_schema, sl, max_tables=6)
    assert result.selected_tables == {"customers", "orders"}


def test_retrieval_pulls_in_bridge_table_for_disconnected_pair():
    """customers and warehouses have no direct FK — the only path in
    layer.yaml runs customers -> orders -> shipments -> warehouses. When the
    question's keyword overlap ranks customers/warehouses above the bridge
    tables, retrieve() must still pull orders/shipments in so the pair is
    joinable — an unjoinable selection would be useless downstream."""
    sl = resolver.get()
    names = ["customers", "warehouses", "suppliers", "employees", "orders",
             "shipments", "products"]
    live_schema = {n: _table(n) for n in names}

    result = sr.retrieve("customers and warehouses", live_schema, sl, max_tables=2)

    assert {"customers", "warehouses"} <= result.selected_tables
    assert result.bridge_tables  # something was pulled in purely for connectivity
    assert result.bridge_tables <= {"orders", "shipments"}
    # The final selection must be genuinely connected — connect_tables should
    # find a path between every pair once bridges are included.
    connected, _ = sl.connect_tables(result.selected_tables)
    assert connected == result.selected_tables or not (result.selected_tables - connected)


def test_realistic_multitable_question_selects_relevant_subset():
    """From the module's own docstring: 'revenue by product category from UAE
    customers' should retrieve a handful of relevant tables out of 13, not
    the whole catalog."""
    sl = resolver.get()
    live_schema = {n: _table(n) for n in sl.tables}
    assert len(live_schema) == 13

    result = sr.retrieve(
        "revenue by product category from UAE customers last quarter",
        live_schema, sl, max_tables=6,
    )

    assert len(result.selected_tables) <= 6
    assert {"customers", "products", "order_items"} <= result.selected_tables


def test_no_keyword_overlap_falls_back_to_something_small():
    sl = resolver.get()
    live_schema = {n: _table(n) for n in sl.tables}
    result = sr.retrieve("zzzzz qqqqq xxxxx", live_schema, sl, max_tables=4)
    # A bridge table can push the final set slightly past max_tables (an
    # unjoinable selection is worse than a marginally larger one — see
    # schema_retriever.retrieve's docstring) but it must stay small relative
    # to the full 13-table catalog, not silently degrade to "everything".
    assert 0 < len(result.selected_tables) < len(live_schema)
