from __future__ import annotations

import uuid

from fastmcp import FastMCP
from fastmcp.server.context import _current_transport
from fastmcp.server.dependencies import get_access_token
from pydantic import BaseModel

from ledger_copilot.domain.idempotency import IdempotencyStore

TOOL_SCOPE = "finance:expense:write"
IDEMPOTENCY_WINDOW_SECONDS = 24 * 60 * 60

# Matches middleware/auth.py's _ENFORCED_TRANSPORTS: a verified actor subject
# only exists as a concept over real HTTP transport. Scoping the idempotency
# key by actor here prevents one caller from replaying or reading back
# another caller's cached write by guessing/reusing their idempotency_key —
# the store itself stays a plain string-keyed cache (domain/idempotency.py),
# so this composes the key rather than changing that module.
_ENFORCED_TRANSPORTS = {"streamable-http", "sse"}


def _scoped_key(raw_key: str) -> str:
    if _current_transport.get() in _ENFORCED_TRANSPORTS:
        token = get_access_token()
        if token is not None and token.subject is not None:
            return f"{token.subject}:{raw_key}"
    return raw_key


class CreateExpenseRequestInput(BaseModel):
    idempotency_key: str
    requested_by: str
    cost_center: str
    amount: float
    currency: str
    description: str


class CreateExpenseRequestOutput(BaseModel):
    expense_id: str
    requested_by: str
    cost_center: str
    amount: float
    currency: str
    description: str


_idempotency_store: IdempotencyStore[CreateExpenseRequestOutput] = IdempotencyStore(
    window_seconds=IDEMPOTENCY_WINDOW_SECONDS
)


def register(app: FastMCP) -> None:
    @app.tool(
        name="create_expense_request",
        description=(
            "Create a new expense request for a cost center. Safe to retry: "
            "calling again with the same idempotency_key returns the "
            "original request instead of creating a duplicate."
        ),
        meta={"scope": TOOL_SCOPE},
    )
    def create_expense_request_tool(input: CreateExpenseRequestInput) -> CreateExpenseRequestOutput:
        def create() -> CreateExpenseRequestOutput:
            return CreateExpenseRequestOutput(
                expense_id=str(uuid.uuid4()),
                requested_by=input.requested_by,
                cost_center=input.cost_center,
                amount=input.amount,
                currency=input.currency,
                description=input.description,
            )

        scoped_key = _scoped_key(input.idempotency_key)
        result, _created = _idempotency_store.get_or_create(scoped_key, create)
        return result
