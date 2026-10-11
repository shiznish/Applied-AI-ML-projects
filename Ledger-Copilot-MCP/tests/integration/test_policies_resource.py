from __future__ import annotations

import json

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app


@pytest.mark.asyncio
async def test_list_resources_includes_policy_template() -> None:
    app = create_app()
    async with Client(app) as client:
        templates = await client.list_resource_templates()

    uri_templates = [template.uri_template for template in templates]
    assert "procurement://policies/{doc_id}" in uri_templates


@pytest.mark.asyncio
async def test_read_policy_by_uri() -> None:
    app = create_app()
    async with Client(app) as client:
        contents = await client.read_resource("procurement://policies/expense-approval-limits")

    assert len(contents) == 1
    payload = json.loads(contents[0].text)
    assert payload["doc_id"] == "expense-approval-limits"
    assert payload["title"] == "Expense Approval Limits Policy"
    assert "segregation of duties" in payload["body"]
