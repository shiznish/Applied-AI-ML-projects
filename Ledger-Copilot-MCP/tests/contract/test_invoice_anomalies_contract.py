from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app
from ledger_copilot.tools.invoice_anomalies import (
    DetectInvoiceAnomaliesInput,
    DetectInvoiceAnomaliesOutput,
)


@pytest.mark.asyncio
async def test_detect_invoice_anomalies_schema_matches_declared_models() -> None:
    app = create_app()
    async with Client(app) as client:
        tools = await client.list_tools()

    tool = next(t for t in tools if t.name == "detect_invoice_anomalies")

    input_properties = tool.input_schema["properties"]["input"]
    required_fields = {
        name
        for name, field in DetectInvoiceAnomaliesInput.model_fields.items()
        if field.is_required()
    }
    assert required_fields <= set(input_properties["required"])
    assert set(input_properties["properties"]) == set(
        DetectInvoiceAnomaliesInput.model_fields.keys()
    )

    output_properties = tool.output_schema["properties"]
    assert set(output_properties) == set(DetectInvoiceAnomaliesOutput.model_fields.keys())
