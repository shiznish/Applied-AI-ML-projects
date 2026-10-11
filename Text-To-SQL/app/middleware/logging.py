"""Structured (JSON-lines) request logging, matching this project's existing
logs/*.jsonl convention (see app/main.py's _log_query/_QUERY_LOG pattern).

Username extraction is best-effort: it decodes the bearer token (if any)
purely to attribute the log line, and never fails or blocks the request if
the token is missing, invalid, or the route doesn't require auth at all —
enforcement is entirely app.auth.rbac's job, not this middleware's.
"""
import json
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from ..config import settings

_REQUEST_LOG = os.path.join(settings.log_dir, "requests.jsonl")


def _best_effort_username(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[len("bearer "):].strip()
    try:
        from ..auth.jwt import decode_token  # local import: keep middleware DB/auth-optional

        claims = decode_token(token)
        return claims.get("sub")
    except Exception:
        return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, log_path: str | None = None):
        super().__init__(app)
        self._log_path = log_path or _REQUEST_LOG

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            record = {
                "ts": time.time(),
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "correlation_id": getattr(request.state, "correlation_id", None),
                "username": _best_effort_username(request),
            }
            try:
                os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except OSError:
                pass  # logging must never break the request path
