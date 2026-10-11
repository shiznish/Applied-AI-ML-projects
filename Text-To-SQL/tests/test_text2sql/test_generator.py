"""Offline unit tests for app/text2sql/generator.py. The provider call
(app.llm._parse_structured) is injected as a fake — no real API key/network
access is available in this environment, and none should be required here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import llm
from app.schema_extract import Table
from app.semantic import resolver
from app.text2sql import generator, schema_linker


def _linked_schema():
    sl = resolver.get()
    names = ["orders", "customers"]
    live_schema = {n: Table(name=n, columns=[], primary_keys=[], foreign_keys=[], row_count=0)
                   for n in names}
    return schema_linker.link("orders by customer", set(names), live_schema, sl)


def _fake_generated(sql="SELECT 1", confidence=0.9, needs_clarification=False):
    return llm.GeneratedSQL(
        needs_clarification=needs_clarification,
        interpretations=[],
        sql=sql,
        explanation="a fake explanation",
        confidence=confidence,
        tables_used=["orders"],
        columns_used=["orders.order_id"],
    )


def test_generate_sql_uses_injected_parse_structured():
    captured = {}

    def fake_parse_structured(system, user, model_cls, max_tokens):
        captured["system"] = system
        captured["user"] = user
        captured["model_cls"] = model_cls
        return _fake_generated()

    linked = _linked_schema()
    result = generator.generate_sql(
        "How many orders per customer?", linked, [], parse_structured=fake_parse_structured
    )

    assert result.sql == "SELECT 1"
    assert captured["model_cls"] is llm.GeneratedSQL
    assert captured["user"] == "How many orders per customer?"
    # The system prompt should be built from the linked schema, not a raw dump.
    assert "orders" in captured["system"]
    assert "customers" in captured["system"]


def test_generate_sql_clamps_confidence_above_one():
    def fake_parse_structured(system, user, model_cls, max_tokens):
        return _fake_generated(confidence=5.0)

    linked = _linked_schema()
    result = generator.generate_sql("q", linked, [], parse_structured=fake_parse_structured)
    assert result.confidence == 1.0


def test_generate_sql_clamps_confidence_below_zero():
    def fake_parse_structured(system, user, model_cls, max_tokens):
        return _fake_generated(confidence=-3.0)

    linked = _linked_schema()
    result = generator.generate_sql("q", linked, [], parse_structured=fake_parse_structured)
    assert result.confidence == 0.0


def test_generate_sql_defaults_to_real_llm_parse_structured():
    """The injectable parameter must default to app.llm._parse_structured so
    production behavior is unchanged when nothing is overridden. (Identity
    comparison is avoided here since duplicate module loading under pytest's
    path manipulation can yield two distinct-but-equivalent function objects;
    name/module equality is what actually matters.)"""
    import inspect

    default = inspect.signature(generator.generate_sql).parameters["parse_structured"].default
    assert default.__name__ == "_parse_structured"
    assert default.__module__.endswith("llm")
