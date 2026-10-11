from __future__ import annotations

import logging

from fastmcp.exceptions import AuthorizationError, InsufficientScopeError
from fastmcp.server.context import _current_transport
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp_types import CallToolRequestParams

logger = logging.getLogger(__name__)

# Bearer-token semantics (missing/invalid/expired/wrong-audience -> 401 with a
# WWW-Authenticate header) are already enforced by FastMCP's own
# RequireAuthMiddleware/BearerAuthBackend at the HTTP transport layer, wired
# in server.py by passing a RemoteAuthProvider(JWTVerifier(...)) as
# FastMCP(auth=...). That happens before a request ever reaches this
# middleware, so a call only gets here already authenticated (or over a
# transport that has no bearer-token concept at all, e.g. stdio or an
# in-memory `fastmcp.Client` used by tests). `_current_transport` is only
# ever set to a value in `_ENFORCED_TRANSPORTS` while driving the real HTTP
# app, matching FastMCP's own "auth is an HTTP transport concept" convention
# (see fastmcp.server.middleware.authorization.AuthMiddleware's identical
# stdio skip). Per-tool scope enforcement below additionally requires
# `fastmcp.auth` to be configured, so a server deliberately run with no auth
# provider (no JWKS URI set) stays unauthenticated over real HTTP too, not
# just over the in-memory test transport.
_ENFORCED_TRANSPORTS = {"streamable-http", "sse"}


class AuthMiddleware(Middleware):
    """Enforces per-tool OAuth scopes for HTTP-transport tool calls, when
    the app has an auth provider configured.

    Bearer token validation (signature, expiry, issuer, audience) happens
    upstream via the JWTVerifier configured in server.py; this middleware
    only checks that the already-verified token carries the scope the
    called tool declares via `meta={"scope": ...}`. If no auth provider is
    configured (`fastmcp.auth is None`), this middleware is a no-op."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        logger.info("auth", extra={"tool": tool_name})

        fastmcp_context = context.fastmcp_context
        if (
            fastmcp_context is not None
            and fastmcp_context.fastmcp.auth is not None
            and _current_transport.get() in _ENFORCED_TRANSPORTS
        ):
            token = get_access_token()
            if token is None:
                raise AuthorizationError(
                    f"Authorization failed for tool '{tool_name}': "
                    "missing or invalid bearer token"
                )

            required_scope = None
            tool = await fastmcp_context.fastmcp.get_tool(tool_name)
            if tool is not None and tool.meta:
                required_scope = tool.meta.get("scope")

            if required_scope and required_scope not in token.scopes:
                raise InsufficientScopeError(
                    [required_scope],
                    message=(
                        f"Authorization failed for tool '{tool_name}': "
                        f"insufficient scope (required: {required_scope})"
                    ),
                )

        return await call_next(context)
