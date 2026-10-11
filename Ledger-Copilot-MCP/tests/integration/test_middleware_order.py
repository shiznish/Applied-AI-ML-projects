import logging

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app


@pytest.mark.asyncio
async def test_middleware_chain_runs_in_documented_order(caplog) -> None:  # type: ignore[no-untyped-def]
    app = create_app()

    @app.tool
    def dummy_tool() -> str:
        return "ok"

    with caplog.at_level(logging.INFO, logger="ledger_copilot.middleware"):
        async with Client(app) as client:
            result = await client.call_tool("dummy_tool", {})

    assert result.data == "ok"

    events = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("ledger_copilot.middleware")
    ]
    assert events == [
        "auth",
        "rate_limit",
        "tracing",
        "audit.pre",
        "audit.post",
        "error_handling",
    ]
