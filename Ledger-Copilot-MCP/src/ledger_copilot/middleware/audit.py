from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastmcp.exceptions import AuthorizationError, InsufficientScopeError
from fastmcp.server.context import _current_transport
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp_types import CallToolRequestParams

from ledger_copilot.audit.schema import AuditRecord, Decision, ResultStatus
from ledger_copilot.audit.store import AuditStore

logger = logging.getLogger(__name__)

# Matches middleware/auth.py's _ENFORCED_TRANSPORTS: actor identity and
# presented scopes only exist as OAuth concepts over real HTTP transport, so
# they're simply absent for the in-memory `fastmcp.Client` transport most
# tests use.
_ENFORCED_TRANSPORTS = {"streamable-http", "sse"}

# Field names that identify a specific human actor (as opposed to a business
# record id like po_id/invoice_id, which stays in the clear so the record is
# still useful for audit review) or that carry free-text a caller could put
# arbitrary PII into. The audit policy is "sensitive fields hashed, financial
# amounts kept" without enumerating which fields count — this is the
# judgment call for this codebase's actual tool inputs.
_SENSITIVE_ARGUMENT_KEYS = frozenset({"requested_by", "approver_id", "description"})


def _hash_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_hash_value(val) if key in _SENSITIVE_ARGUMENT_KEYS else _redact(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _on_behalf_of_chain(claims: dict[str, Any] | None) -> list[str]:
    """Walk the RFC 8693 `act` (actor) delegation chain, innermost first."""
    chain: list[str] = []
    actor = (claims or {}).get("act")
    while isinstance(actor, dict):
        subject = actor.get("sub")
        if subject is not None:
            chain.append(str(subject))
        actor = actor.get("act")
    return chain


def _find_str_field(payload: Any, key: str, *, max_depth: int = 3) -> str | None:
    """Recursively look for a string value at `key`.

    Tool outputs are either a single Pydantic model's fields directly, or
    (for `X | Y` union outputs) that same shape nested one level under a
    `result` key — see tests/contract/*. Searching structurally instead of
    hardcoding "result" keeps this working for both without coupling the
    audit middleware to any one tool's schema.
    """
    if max_depth < 0 or not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, str):
        return value
    for nested in payload.values():
        found = _find_str_field(nested, key, max_depth=max_depth - 1)
        if found is not None:
            return found
    return None


class AuditMiddleware(Middleware):
    """Writes one append-only `AuditRecord` per tool call to `AuditStore`.

    Still emits the "audit.pre"/"audit.post" log lines
    `tests/integration/test_middleware_order.py` asserts on, alongside the
    real audit record — they're cheap, useful call markers independent of
    the structured record.
    """

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        logger.info("audit.pre", extra={"tool": tool_name})

        arguments = _redact(context.message.arguments or {})
        idempotency_key = _find_str_field(arguments, "idempotency_key")

        actor_user_id: str | None = None
        scopes_presented: list[str] = []
        on_behalf_of_chain: list[str] = []
        if _current_transport.get() in _ENFORCED_TRANSPORTS:
            token = get_access_token()
            if token is not None:
                actor_user_id = token.subject
                scopes_presented = list(token.scopes)
                on_behalf_of_chain = _on_behalf_of_chain(token.claims)

        trace_id = (
            context.fastmcp_context.request_id
            if context.fastmcp_context is not None
            else str(uuid.uuid4())
        )

        started_at = time.monotonic()

        def record(
            *, decision: Decision, result_status: ResultStatus, result: ToolResult | None
        ) -> None:
            self._store.append(
                AuditRecord(
                    record_id=str(uuid.uuid4()),
                    timestamp=datetime.now(UTC),
                    trace_id=trace_id,
                    actor_user_id=actor_user_id,
                    on_behalf_of_chain=on_behalf_of_chain,
                    tool_name=tool_name,
                    arguments=arguments,
                    scopes_presented=scopes_presented,
                    decision=decision,
                    approval_id=_find_str_field(
                        result.structured_content if result is not None else None,
                        "approval_id",
                    ),
                    idempotency_key=idempotency_key,
                    latency_ms=(time.monotonic() - started_at) * 1000,
                    result_status=result_status,
                )
            )

        try:
            result = await call_next(context)
        except (AuthorizationError, InsufficientScopeError):
            record(decision="denied", result_status="error", result=None)
            raise
        except Exception:
            record(decision="allowed", result_status="error", result=None)
            raise

        logger.info("audit.post", extra={"tool": tool_name})
        record(decision="allowed", result_status="success", result=result)
        return result
