from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

Decision = Literal["allowed", "denied"]
ResultStatus = Literal["success", "error"]


class AuditRecord(BaseModel):
    """One append-only entry per tool invocation."""

    record_id: str
    timestamp: datetime
    trace_id: str
    actor_user_id: str | None
    on_behalf_of_chain: list[str]
    tool_name: str
    arguments: dict[str, Any]
    scopes_presented: list[str]
    decision: Decision
    approval_id: str | None
    idempotency_key: str | None
    latency_ms: float
    result_status: ResultStatus
