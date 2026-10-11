from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider, JWTVerifier, RemoteAuthProvider
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

from ledger_copilot.audit.store import AuditStore
from ledger_copilot.config import Settings
from ledger_copilot.middleware.audit import AuditMiddleware
from ledger_copilot.middleware.auth import AuthMiddleware
from ledger_copilot.middleware.error_handling import ErrorHandlingMiddleware
from ledger_copilot.middleware.rate_limit import RateLimitMiddleware
from ledger_copilot.middleware.tracing import TracingMiddleware
from ledger_copilot.prompts import quarterly_review
from ledger_copilot.resources import policies
from ledger_copilot.tools import (
    expense_request,
    invoice_anomalies,
    po_status,
    spend_report,
    vendor_risk,
)


def build_auth_provider(settings: Settings) -> AuthProvider | None:
    """Build the OAuth 2.1 resource-server auth provider from env config.

    Returns None (no bearer-token requirement) when no JWKS endpoint is
    configured, so `create_app()` stays zero-config for local/dev use and
    for the in-memory `fastmcp.Client` tests — those never exercise the real
    HTTP transport this provider protects (see middleware/auth.py's
    `_ENFORCED_TRANSPORTS` comment).
    """
    if not settings.auth_jwks_uri:
        return None

    verifier = JWTVerifier(
        jwks_uri=settings.auth_jwks_uri,
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        algorithm=settings.auth_algorithm,
    )
    authorization_servers = (
        [AnyHttpUrl(settings.auth_issuer)] if settings.auth_issuer else []
    )
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=authorization_servers,
        base_url=f"http://{settings.host}:{settings.port}",
    )


def create_app(
    auth: AuthProvider | None = None,
    audit_store: AuditStore | None = None,
    rate_limit_middleware: RateLimitMiddleware | None = None,
) -> FastMCP:
    settings = Settings()
    resolved_auth = auth if auth is not None else build_auth_provider(settings)
    resolved_audit_store = (
        audit_store if audit_store is not None else AuditStore(settings.audit_database_path)
    )
    resolved_rate_limit = rate_limit_middleware or RateLimitMiddleware(
        capacity=settings.rate_limit_capacity,
        refill_per_second=settings.rate_limit_refill_per_second,
    )
    app = FastMCP(name="ledger-copilot", auth=resolved_auth)

    # FastMCP wraps middleware onion-style: the first one added is outermost
    # and runs its pre-logic first. To get the intended log order
    # (auth -> rate_limit -> tracing -> audit.pre -> handler -> audit.post ->
    # error_handling), error_handling must be registered *before* audit so it
    # sits just inside tracing but outside audit, and audit ends up innermost,
    # directly wrapping the handler.
    app.add_middleware(AuthMiddleware())
    app.add_middleware(resolved_rate_limit)
    app.add_middleware(TracingMiddleware())
    app.add_middleware(ErrorHandlingMiddleware())
    app.add_middleware(AuditMiddleware(resolved_audit_store))

    po_status.register(app)
    vendor_risk.register(app)
    expense_request.register(app)
    invoice_anomalies.register(app)
    spend_report.register(app)
    policies.register(app)
    quarterly_review.register(app)

    @app.custom_route("/healthz", methods=["GET"])
    async def liveness(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.custom_route("/readyz", methods=["GET"])
    async def readiness(request: Request) -> JSONResponse:
        if resolved_audit_store.ping():
            return JSONResponse({"status": "ok"})
        return JSONResponse(
            {"status": "unhealthy", "reason": "audit_store unreachable"},
            status_code=503,
        )

    return app


app = create_app()


def configure_tracing() -> None:
    """Install the process-wide OpenTelemetry SDK TracerProvider.

    Deliberately not called from `create_app()`: `set_tracer_provider` may
    only take effect once per process, and tests install their own
    in-memory exporter's provider before driving the app to assert on
    spans — calling this here (create_app() runs many times across the
    test suite) would race that and silently lose to whichever runs first.
    `main()` is the one real entrypoint, so it's the right place to wire a
    real exporter for an actual running server.
    """
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def main() -> None:
    configure_tracing()
    settings = Settings()
    app.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
