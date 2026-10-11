from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app
from ledger_copilot.tools.po_status import GetPoStatusInput, PoStatusFound, PoStatusNotFound


@pytest.mark.asyncio
async def test_get_po_status_schema_matches_declared_models() -> None:
    app = create_app()
    async with Client(app) as client:
        tools = await client.list_tools()

    tool = next(t for t in tools if t.name == "get_po_status")

    input_properties = tool.input_schema["properties"]["input"]
    assert set(input_properties["required"]) == set(GetPoStatusInput.model_fields.keys())
    assert set(input_properties["properties"]) == set(GetPoStatusInput.model_fields.keys())

    output_variants = tool.output_schema["properties"]["result"]["anyOf"]
    assert len(output_variants) == 2

    variants_by_property_set = {frozenset(variant["properties"]) for variant in output_variants}
    assert frozenset(PoStatusFound.model_fields.keys()) in variants_by_property_set
    assert frozenset(PoStatusNotFound.model_fields.keys()) in variants_by_property_set
