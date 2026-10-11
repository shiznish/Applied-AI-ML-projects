from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app


@pytest.mark.asyncio
async def test_list_prompts_includes_quarterly_review() -> None:
    app = create_app()
    async with Client(app) as client:
        prompts = await client.list_prompts()

    assert "quarterly-review" in [prompt.name for prompt in prompts]


@pytest.mark.asyncio
async def test_render_quarterly_review_prompt() -> None:
    app = create_app()
    async with Client(app) as client:
        result = await client.get_prompt("quarterly-review", {"cost_center": "CC-100"})

    assert len(result.messages) == 1
    text = result.messages[0].content.text
    assert "CC-100" in text
    assert "generate_spend_report" in text
