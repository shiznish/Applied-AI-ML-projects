import logging

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp_types import CallToolRequestParams

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(Middleware):
    """Catches anything unhandled from audit/the tool handler and logs it so
    nothing throws unhandled into the transport layer. Currently re-raises
    after logging; a normalized error response is a candidate for later,
    once more tool logic exists to normalize against."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            result = await call_next(context)
        except Exception:
            logger.exception("unhandled tool error", extra={"tool": context.message.name})
            raise
        logger.info("error_handling", extra={"tool": context.message.name})
        return result
