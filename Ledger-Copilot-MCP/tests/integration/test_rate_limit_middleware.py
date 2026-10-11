from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from ledger_copilot.middleware.rate_limit import RateLimitMiddleware
from ledger_copilot.server import create_app


@pytest.mark.asyncio
async def test_rate_limit_trips_then_recovers_after_refill() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    rate_limit = RateLimitMiddleware(capacity=2, refill_per_second=1.0, clock=clock)
    app = create_app(rate_limit_middleware=rate_limit)

    @app.tool
    def dummy_tool() -> str:
        return "ok"

    async with Client(app) as client:
        assert (await client.call_tool("dummy_tool", {})).data == "ok"
        assert (await client.call_tool("dummy_tool", {})).data == "ok"

        with pytest.raises(ToolError, match="rate limit exceeded"):
            await client.call_tool("dummy_tool", {})

        clock_value = 1.0
        assert (await client.call_tool("dummy_tool", {})).data == "ok"
