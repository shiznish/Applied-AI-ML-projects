from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app


def _expense_args(idempotency_key: str) -> dict[str, object]:
    return {
        "input": {
            "idempotency_key": idempotency_key,
            "requested_by": "alice",
            "cost_center": "CC-100",
            "amount": 42.5,
            "currency": "USD",
            "description": "team lunch",
        }
    }


@pytest.mark.asyncio
async def test_first_call_creates_expense() -> None:
    app = create_app()
    async with Client(app) as client:
        result = await client.call_tool("create_expense_request", _expense_args("key-first-call"))

    assert result.data.cost_center == "CC-100"
    assert result.data.expense_id


@pytest.mark.asyncio
async def test_duplicate_key_returns_cached_result() -> None:
    app = create_app()
    async with Client(app) as client:
        first = await client.call_tool("create_expense_request", _expense_args("key-duplicate"))
        second = await client.call_tool("create_expense_request", _expense_args("key-duplicate"))

    assert first.data.expense_id == second.data.expense_id


@pytest.mark.asyncio
async def test_different_key_creates_again() -> None:
    app = create_app()
    async with Client(app) as client:
        first = await client.call_tool("create_expense_request", _expense_args("key-a"))
        second = await client.call_tool("create_expense_request", _expense_args("key-b"))

    assert first.data.expense_id != second.data.expense_id
