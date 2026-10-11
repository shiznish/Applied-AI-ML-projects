from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app
from ledger_copilot.tools.spend_report import (
    GenerateSpendReportInput,
    GenerateSpendReportOutput,
)


@pytest.mark.asyncio
async def test_generate_spend_report_schema_matches_declared_models() -> None:
    app = create_app()
    async with Client(app) as client:
        tools = await client.list_tools()

    tool = next(t for t in tools if t.name == "generate_spend_report")

    input_properties = tool.input_schema["properties"]["input"]
    required_fields = {
        name
        for name, field in GenerateSpendReportInput.model_fields.items()
        if field.is_required()
    }
    assert required_fields <= set(input_properties["required"])
    assert set(input_properties["properties"]) == set(
        GenerateSpendReportInput.model_fields.keys()
    )

    output_properties = tool.output_schema["properties"]
    assert set(output_properties) == set(GenerateSpendReportOutput.model_fields.keys())
