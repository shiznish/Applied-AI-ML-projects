from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastmcp.exceptions import ToolError
from fastmcp.server.context import _current_transport
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp_types import CallToolRequestParams

logger = logging.getLogger(__name__)

# Matches middleware/auth.py's _ENFORCED_TRANSPORTS: a verified OAuth client
# identity only exists as a concept over real HTTP transport.
_ENFORCED_TRANSPORTS = {"streamable-http", "sse"}

# Fallback bucket key for transports with no verified per-caller identity
# (stdio, the in-memory `fastmcp.Client` transport most tests use) — every
# such call shares one bucket rather than being unlimited, matching this
# codebase's general "absent over unenforced transport" convention.
_UNAUTHENTICATED_CLIENT_ID = "__unauthenticated__"


class TokenBucket:
    """Continuous-refill token bucket: `capacity` tokens max, refilled at
    `refill_per_second`. An injectable clock makes tripping and recovery
    testable without real sleeps, mirroring domain/idempotency.py's
    injectable-clock pattern."""

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._clock = clock
        self._tokens = capacity
        self._last_refill = clock()

    def try_consume(self, cost: float = 1.0) -> bool:
        now = self._clock()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._last_refill = now
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True


def _client_id() -> str:
    if _current_transport.get() in _ENFORCED_TRANSPORTS:
        token = get_access_token()
        if token is not None:
            return token.client_id
    return _UNAUTHENTICATED_CLIENT_ID


class RateLimitMiddleware(Middleware):
    """Token-bucket rate limiting, one bucket per client_id.

    client_id is the verified access token's `client_id` when auth is
    enforced (the same transport gate middleware/auth.py and
    middleware/audit.py use) — that's the only caller identity FastMCP's
    OAuth layer ties to a specific registered client rather than a shared
    connection. Every other transport has no such identity, so those calls
    fall back to one shared bucket.
    """

    def __init__(
        self,
        capacity: float = 20.0,
        refill_per_second: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._clock = clock
        # In-memory and per-process, so this is correct only for a single
        # replica — N replicas behind a load balancer each enforce their own
        # limit, multiplying effective capacity by N. Scaling beyond one
        # replica needs this moved to a shared store (e.g. Redis) so all
        # replicas decrement the same counter per client_id.
        self._buckets: dict[str, TokenBucket] = {}

    def _bucket_for(self, client_id: str) -> TokenBucket:
        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = TokenBucket(self._capacity, self._refill_per_second, self._clock)
            self._buckets[client_id] = bucket
        return bucket

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        logger.info("rate_limit", extra={"tool": tool_name})

        client_id = _client_id()
        if not self._bucket_for(client_id).try_consume():
            raise ToolError(f"rate limit exceeded for client '{client_id}'")

        return await call_next(context)
